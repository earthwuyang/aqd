#!/usr/bin/env python3
# train_diff_rowcol.py  —  Differentiable-programming hybrid cost model
# Compatible with your existing data folders
# ---------------------------------------------------------------
import argparse, json, math, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def log(msg, lvl="INFO"): print(f"[{lvl}] {msg}", file=sys.stderr if lvl!="INFO" else sys.stdout)
def safe_float(x):   # robust float converter
    try: return float(x)
    except (ValueError, TypeError): return float("nan")

def safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, x))

def parse_number(j, key: str) -> float:
    if key not in j:               return 0.0
    if isinstance(j[key], (int,float)): return float(j[key])
    if isinstance(j[key], str):
        try:    return float(j[key])
        except: return 0.0
    return 0.0

def bytes_from_human(txt: str) -> float:          # "2.5G" → 2.5e9
    if not txt: return 0.0
    txt = txt.strip()
    if not txt: return 0.0
    suffix = txt[-1].upper()
    factor = {"K":1e3,"M":1e6,"G":1e9}.get(suffix,1.0)
    if suffix in "KMG": txt = txt[:-1]
    try:    val = float(txt)
    except: val = 0.0
    return val * factor

# --------------------------------------------------------------------------- #
# === Row-plan JSON → 8-D feature vector  (exactly like your C++ extractor)   #
# --------------------------------------------------------------------------- #
def row_json_to_feat(path: Path) -> Tuple[np.ndarray, float]:
    feat = np.zeros(8, dtype=np.float32)
    try:    j = json.loads(path.read_text())
    except Exception as e:
        log(f"bad JSON {path}: {e}", "WARN")
        return feat, 0.0

    blk = j.get("query_block", {})
    sums = dict(Re=0,Rp=0,F=0,Rc=0,Ec=0,Pc=0,Dr=0,n=0)

    def dfs(node):
        if isinstance(node, dict):
            if "table" in node:
                t = node["table"]; ci = t.get("cost_info",{})
                sums["Re"] += parse_number(t,"rows_examined_per_scan")
                sums["Rp"] += parse_number(t,"rows_produced_per_join")
                sums["F"]  += parse_number(t,"filtered")
                sums["Rc"] += parse_number(ci,"read_cost")
                sums["Ec"] += parse_number(ci,"eval_cost")
                sums["Pc"] += parse_number(ci,"prefix_cost")
                sums["Dr"] += bytes_from_human(ci.get("data_read_per_join","0"))
                sums["n"]  += 1
            for v in node.values(): dfs(v)
        elif isinstance(node, list):
            for v in node: dfs(v)
    dfs(blk)

    qc = parse_number(blk.get("cost_info",{}), "query_cost")

    if sums["n"] > 0:
        vals = [sums["Re"],sums["Rp"],sums["F"],sums["Rc"],
                sums["Ec"],sums["Pc"],sums["Dr"], qc]
        feat[:] = [safe_log1p(v/sums["n"]) for v in vals]
    return feat, qc

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class Sample:
    __slots__=("row_feat","label","row_time","col_time","orig_cost","hybrid_flag")
    def __init__(self, row_feat, label, row_t, col_t, orig_cost, hybrid_flag):
        self.row_feat=row_feat; self.label=label
        self.row_time=row_t; self.col_time=col_t
        self.orig_cost=orig_cost; self.hybrid_flag=hybrid_flag

class RowColDataset(Dataset):
    def __init__(self, base: Path, dirs: List[str]):
        self.s: List[Sample] = []
        for d in dirs:
            wdir = base/d
            csv  = wdir/"query_costs.csv"
            if not csv.exists():
                log(f"missing {csv}", "WARN"); continue
            row_dir = wdir/"row_plans"

            lines = csv.read_text().splitlines()
            for ln in tqdm(lines[1:], desc=f"{d}"):
                qid, lab, row_s, col_s, hyb = ln.strip().split(",")[:5]
                row_json = row_dir/f"{qid}.json"
                if not row_json.exists():   continue
                feat, qc = row_json_to_feat(row_json)
                self.s.append(Sample(
                    feat,
                    int(lab) if lab else -1,
                    safe_float(row_s),
                    safe_float(col_s),
                    qc,
                    safe_float(hyb)))
        log(f"Loaded {len(self.s)} samples")

    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]

# --------------------------------------------------------------------------- #
# Differentiable-programming Cost Model                                       #
# --------------------------------------------------------------------------- #
class DiffCostModel(nn.Module):
    """
    row_cost  = 〈w_row , x〉 + MLP_row(x)
    col_cost  = 〈w_col , x〉 + MLP_col(x)
    prob_col  = σ(col_cost - row_cost)
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.w_row = nn.Linear(8,1, bias=False)      # analytic weights
        self.w_col = nn.Linear(8,1, bias=False)
        self.mlp_row = nn.Sequential(
            nn.Linear(8,hidden), nn.ReLU(), nn.Linear(hidden,1))
        self.mlp_col = nn.Sequential(
            nn.Linear(8,hidden), nn.ReLU(), nn.Linear(hidden,1))

    def forward(self, x):                # x: [B,8]
        row_cost = self.w_row(x) + self.mlp_row(x)   # [B,1]
        col_cost = self.w_col(x) + self.mlp_col(x)
        return torch.sigmoid(col_cost - row_cost).squeeze(1)   # [B]

# --------------------------------------------------------------------------- #
# Early stopping helper                                                       #
# --------------------------------------------------------------------------- #
class Stopper:
    def __init__(self, patience=12): self.pat, self.best, self.stale = patience, None, 0
    def step(self, val):
        if self.best is None or val < self.best:
            self.best, self.stale = val, 0; return False
        self.stale += 1; return self.stale >= self.pat

# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/wuy/query_costs")
    ap.add_argument("--data_dirs", action="append")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--ckpt", default="checkpoints/hybrid_diff_model.pt")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    dirs = args.data_dirs or ["carcinogenesis"]
    ds   = RowColDataset(Path(args.base), dirs)
    if len(ds)==0: log("empty dataset", "ERROR"); sys.exit(1)

    idxs = list(range(len(ds))); random.shuffle(idxs)
    split = int(0.8*len(idxs))
    train_idx, val_idx = idxs[:split], idxs[split:]
    log(f"Train={len(train_idx)}  Val={len(val_idx)}")

    # μ/σ for row features (train set only)
    feats = np.stack([ds[i].row_feat for i in train_idx])
    mu, sig = feats.mean(0), feats.std(0)+1e-8
    mu = torch.tensor(mu, dtype=torch.float32)
    sig= torch.tensor(sig,dtype=torch.float32)
    def norm(x): return (torch.tensor(x,dtype=torch.float32)-mu)/sig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DiffCostModel(hidden=args.hidden).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit   = nn.BCELoss()
    stopper= Stopper()

    # class balancing
    lbls = [max(0,ds[i].label) for i in train_idx]
    pos, neg = sum(lbls), len(lbls)-sum(lbls)
    wts = [(1/pos if l==1 else 1/neg) for l in lbls]
    sampler = WeightedRandomSampler(wts, len(wts), replacement=True)
    train_loader = DataLoader(train_idx, batch_size=args.batch,
                              sampler=sampler, drop_last=False)

    # ------------------------------------------------------------------- #
    # Training loop                                                       #
    # ------------------------------------------------------------------- #
    if not args.skip_train:
        best=1e9
        for ep in range(1, args.epochs+1):
            model.train(); tr_loss=0.0
            for batch in tqdm(train_loader, desc=f"Ep{ep}", leave=False):
                opt.zero_grad(set_to_none=True)
                lbl = torch.tensor([ds[i].label==1 for i in batch],
                                   dtype=torch.float32, device=device)
                feats = torch.stack([norm(ds[i].row_feat) for i in batch]).to(device)
                p = model(feats)
                loss = crit(p, lbl); loss.backward(); opt.step()
                tr_loss += loss.item()*len(batch)
            tr_loss /= len(train_idx)
            # validation
            model.eval(); val_loss=0.0
            with torch.no_grad():
                for i in val_idx:
                    lbl = torch.tensor([ds[i].label==1], dtype=torch.float32,device=device)
                    p   = model(norm(ds[i].row_feat).to(device).unsqueeze(0))
                    val_loss += crit(p,lbl).item()
            val_loss/=len(val_idx)
            log(f"Ep{ep}  train={tr_loss:.6f}  val={val_loss:.6f}")
            if val_loss<best:
                best=val_loss
                Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "mu": mu, "sig": sig
                }, args.ckpt)
                log("  • saved best ckpt")
            if stopper.step(val_loss):
                log("early stop"); break

    # ------------------------------------------------------------------- #
    # Inference metrics                                                   #
    # ------------------------------------------------------------------- #
    # ckpt = torch.load(args.ckpt, map_location=device)
    # model.load_state_dict(ckpt["model"]); model.eval()
    # mu, sig = ckpt["mu"], ckpt["sig"]
            
    if args.ckpt.endswith(".pt") and not args.ckpt.endswith(".pth"):
        # it's a scripted TorchScript model
        model = torch.jit.load(args.ckpt, map_location=device)
        mu = torch.tensor([0.0]*8)   # dummy if not saved
        sig= torch.tensor([1.0]*8)
        log("loaded TorchScript model")
    else:
        # it's a Python checkpoint
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"]); model.eval()
        mu, sig = ckpt["mu"], ckpt["sig"]
        log(f"loaded checkpoint {args.ckpt}")


    agg = {k:0.0 for k in ("row","col","cost","hyb","opt","ai")}
    cnt = {k:0   for k in agg}
    with torch.no_grad():
        for s in tqdm(ds, desc="Metrics"):
            # row/col times
            if not math.isnan(s.row_time): agg["row"]+=s.row_time; cnt["row"]+=1
            if not math.isnan(s.col_time): agg["col"]+=s.col_time; cnt["col"]+=1
            # simple cost rule
            rule_rt = s.col_time if s.orig_cost>5e4 else s.row_time
            if not math.isnan(rule_rt):    agg["cost"]+=rule_rt; cnt["cost"]+=1
            # hybrid flag from MySQL kernel
            hyb_rt  = s.col_time if s.hybrid_flag else s.row_time
            if not math.isnan(hyb_rt):     agg["hyb"]+=hyb_rt; cnt["hyb"]+=1
            # oracle
            best_rt = min(s.row_time, s.col_time)
            if not math.isnan(best_rt):    agg["opt"]+=best_rt; cnt["opt"]+=1
            # our diff-model
            feat = ((torch.tensor(s.row_feat)-mu)/sig).to(device)
            choose_col = model(feat.unsqueeze(0)).item() >= 0.5
            ai_rt = s.col_time if choose_col else s.row_time
            if not math.isnan(ai_rt):      agg["ai"]+=ai_rt;  cnt["ai"]+=1

    for k in agg:
        agg[k] = agg[k]/cnt[k] if cnt[k] else float("nan")
    print(f"Row={agg['row']:.6f}"
          f"  Col={agg['col']:.6f}"
          f"  CostRule={agg['cost']:.6f}"
          f"  Hybrid={agg['hyb']:.6f}"
          f"  Optimal={agg['opt']:.6f}"
          f"  AI={agg['ai']:.6f}")

    # ------------------------------------------------------------------- #
    # Export TorchScript for C++ inference                                #
    # ------------------------------------------------------------------- #
    scripted = torch.jit.script(model.cpu())
    out_pt = Path(args.ckpt).with_suffix(".pt")
    scripted.save(out_pt)
    log(f"TorchScript saved → {out_pt}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
