#!/usr/bin/env python3
# differential_model_routing.py
# ---------------------------------------------------------------
# Row/Column hybrid cost model with differentiable programming +
# a GIN encoder over the column plan graph.
# ---------------------------------------------------------------
import argparse, json, math, os, random, sys, time
from pathlib import Path
from typing  import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.nn   import GINConv
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def log(msg, lvl="INFO"):
    tgt = sys.stderr if lvl != "INFO" else sys.stdout
    print(f"[{lvl}] {msg}", file=tgt)

def safe_float(x):   # robust float converter
    try: return float(x)
    except (ValueError, TypeError): return float("nan")

def safe_log1p(x: float) -> float:               # log1p ∘ clamp(≥0)
    return math.log1p(max(0.0, x))

def parse_num(j, k: str) -> float:               # robust numeric getter
    if k not in j:               return 0.0
    v = j[k]
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try:    return float(v)
        except: return 0.0
    return 0.0

def bytes_human(txt: str) -> float:              # "86G" → 8.6e10
    if not txt: return 0.0
    txt = txt.strip();  factor = 1.0
    if txt[-1] in "KMGkmg":
        factor = dict(K=1e3, M=1e6, G=1e9,
                      k=1e3, m=1e6, g=1e9)[txt[-1]]
        txt = txt[:-1]
    try:    return float(txt) * factor
    except: return 0.0

# --------------------------------------------------------------------------- #
# Row-plan → 8-D vector                                                       #
# --------------------------------------------------------------------------- #
def row_json_to_feat(path: Path) -> Tuple[np.ndarray, float]:
    f  = np.zeros(8, dtype=np.float32)
    qc = 0.0
    try:        js = json.loads(path.read_text())
    except:     return f, qc
    blk = js.get("query_block", {})
    agg = dict(Re=0,Rp=0,F=0,Rc=0,Ec=0,Pc=0,Dr=0,n=0)
    def dfs(nd):
        if isinstance(nd, dict):
            if "table" in nd:
                tb  = nd["table"]; ci = tb.get("cost_info",{})
                agg["Re"] += parse_num(tb,"rows_examined_per_scan")
                agg["Rp"] += parse_num(tb,"rows_produced_per_join")
                agg["F"]  += parse_num(tb,"filtered")
                agg["Rc"] += parse_num(ci,"read_cost")
                agg["Ec"] += parse_num(ci,"eval_cost")
                agg["Pc"] += parse_num(ci,"prefix_cost")
                agg["Dr"] += bytes_human(ci.get("data_read_per_join","0"))
                agg["n"]  += 1
            for v in nd.values(): dfs(v)
        elif isinstance(nd, list):
            for v in nd: dfs(v)
    dfs(blk)
    qc = parse_num(blk.get("cost_info", {}), "query_cost")
    if agg["n"]:
        vals = [agg["Re"],agg["Rp"],agg["F"],
                agg["Rc"],agg["Ec"],agg["Pc"],agg["Dr"], qc]
        f[:] = [safe_log1p(v/agg["n"]) for v in vals]
    return f, qc

# --------------------------------------------------------------------------- #
# Column-plan normalisation stats                                             #
# --------------------------------------------------------------------------- #
class ColStats:
    def __init__(self, rows_c, rows_s, cost_c, cost_s, op2id):
        self.rows_c, self.rows_s = rows_c, rows_s or 1.0
        self.cost_c, self.cost_s = cost_c, cost_s or 1.0
        self.op2id = op2id                      # local mapping str→int

def build_col_stats(col_dir: Path) -> ColStats:
    cache = col_dir.parent / "column_plan_statistics.json"
    if cache.exists():
        try:
            js = json.loads(cache.read_text())
            if "rows_c" in js:      # new (Python) cache
                return ColStats(js["rows_c"], js["rows_s"],
                                js["cost_c"], js["cost_s"],
                                {k:int(v) for k,v in js["op2id"].items()})
        except: log("bad stats cache "+str(cache),"WARN")
    # otherwise scan and compute
    rows, cost, ops = [], [], set()
    for p in col_dir.glob("*.json"):
        try: plan = json.loads(p.read_text())
        except: continue
        for nd in plan.get("plan", []):
            stack=[nd]
            while stack:
                n = stack.pop()
                rows.append(safe_log1p(parse_num(n,"esti_rows")))
                cost.append(safe_log1p(parse_num(n,"esti_cost")))
                ops.add(n.get("operator","UNK"))
                stack.extend(n.get("children",[]))
    rows_c = float(np.median(rows)); cost_c = float(np.median(cost))
    rows_s = float((np.percentile(rows,75)-np.percentile(rows,25))/2.0) or 1.0
    cost_s = float((np.percentile(cost,75)-np.percentile(cost,25))/2.0) or 1.0
    op2id  = {op:i for i,op in enumerate(sorted(ops))}
    cache.write_text(json.dumps({"rows_c":rows_c,"rows_s":rows_s,
                                 "cost_c":cost_c,"cost_s":cost_s,
                                 "op2id":op2id},indent=2))
    return ColStats(rows_c,rows_s,cost_c,cost_s,op2id)

# --------------------------------------------------------------------------- #
# Column-plan → torch_geometric.Data                                          #
# --------------------------------------------------------------------------- #
def col_json_to_graph(js, stats: ColStats,
                      global_op2id: Dict[str,int]) -> Data:
    empty = Data(x_num=torch.zeros((0,2)), op_idx=torch.zeros((0,),dtype=torch.long),
                 edge_index=torch.zeros((2,0),dtype=torch.long))
    if "plan" not in js or not js["plan"]: return empty
    nodes, edges = [], []
    def dfs(nd, parent=-1):
        idx = len(nodes)
        nodes.append(nd)
        if parent>=0:
            edges.append((parent,idx)); edges.append((idx,parent))
        for ch in nd.get("children", []): dfs(ch, idx)
    dfs(js["plan"][0])
    N = len(nodes)
    rows = [(safe_log1p(parse_num(n,"esti_rows")) - stats.rows_c)/stats.rows_s
            for n in nodes]
    cost = [(safe_log1p(parse_num(n,"esti_cost")) - stats.cost_c)/stats.cost_s
            for n in nodes]
    x_num  = torch.tensor(list(zip(rows,cost)), dtype=torch.float32)
    op_idx = torch.tensor([global_op2id.get(n.get("operator","UNK"),0)
                           for n in nodes], dtype=torch.long)
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        ei = torch.zeros((2,0), dtype=torch.long)
    return Data(x_num=x_num, op_idx=op_idx, edge_index=ei)

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class Sample:
    __slots__=("row_feat","col_graph","label","row_time","col_time",
               "orig_cost","hybrid_flag")
    def __init__(self,row_feat,col_graph,label,row_t,col_t,orig_cost,hyb):
        self.row_feat=row_feat; self.col_graph=col_graph
        self.label=label; self.row_time=row_t; self.col_time=col_t
        self.orig_cost=orig_cost; self.hybrid_flag=hyb

class RowColDataset(Dataset):
    def __init__(self, base:Path, dirs:List[str]):
        self.samples:List[Sample]=[]
        global_ops=set(["UNK"])
        dir_stats:Dict[str,ColStats]={}
        # pass-1: build per-dir stats & global op set
        for d in dirs:
            cdir=(base/d/"column_plans")
            if cdir.exists():
                st=build_col_stats(cdir)
                dir_stats[d]=st
                global_ops.update(st.op2id.keys())
        global_op2id={op:i for i,op in enumerate(sorted(global_ops))}
        # pass-2: load samples
        for d in dirs:
            wdir=base/d; csv=wdir/"query_costs.csv"
            if not csv.exists(): log(f"missing {csv}","WARN"); continue
            row_dir=wdir/"row_plans"; col_dir=wdir/"column_plans"
            st=dir_stats.get(d)
            lines=csv.read_text().splitlines()
            for ln in tqdm(lines[1:],desc=d):
                qid,lab,row_s,col_s,hyb=ln.strip().split(",")[:5]
                row_json=row_dir/f"{qid}.json"
                col_json=col_dir/f"{qid}.json"
                if not row_json.exists(): continue
                rfeat,qc=row_json_to_feat(row_json)
                cgraph=Data()
                if st and col_json.exists():
                    try: cgraph=col_json_to_graph(
                                json.loads(col_json.read_text()),
                                st,global_op2id)
                    except: cgraph=Data()
                self.samples.append(Sample(
                    rfeat, cgraph,
                    int(lab) if lab else -1,
                    safe_float(row_s), safe_float(col_s),
                    qc, safe_float(hyb)))
        log(f"Loaded {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

# --------------------------------------------------------------------------- #
# GIN encoder                                                                 #
# --------------------------------------------------------------------------- #
class GINEncoder(nn.Module):
    def __init__(self,n_ops:int,hidden:int,dropout:float=.2):
        super().__init__()
        self.op_emb=nn.Embedding(n_ops,8)
        self.gin=GINConv(nn.Sequential(
            nn.Linear(10,hidden), nn.ReLU(), nn.Linear(hidden,hidden)))
        self.norm=nn.LayerNorm(hidden); self.dp=nn.Dropout(dropout)
    def forward(self,data:Data):
        if data.x_num.numel()==0:
            return torch.zeros(self.norm.normalized_shape,
                               device=data.x_num.device)
        x=torch.cat([data.x_num,self.op_emb(data.op_idx)],1)
        h=self.gin(x,data.edge_index)
        h=self.dp(self.norm(h))
        return h.mean(0)                # graph embedding

# --------------------------------------------------------------------------- #
# Differential cost head                                                     #
# --------------------------------------------------------------------------- #
class DiffHead(nn.Module):
    def __init__(self,in_dim:int,hidden:int=64):
        super().__init__()
        self.w_row=nn.Linear(in_dim,1,bias=False)
        self.w_col=nn.Linear(in_dim,1,bias=False)
        self.mlp_row=nn.Sequential(nn.Linear(in_dim,hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden,1))
        self.mlp_col=nn.Sequential(nn.Linear(in_dim,hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden,1))
    def forward(self,feat):          # feat [B,in_dim]
        r=self.w_row(feat)+self.mlp_row(feat)
        c=self.w_col(feat)+self.mlp_col(feat)
        return torch.sigmoid(c-r).squeeze(1)

# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="/home/wuy/query_costs")
    p.add_argument("--data_dirs",action="append")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--ckpt",default="checkpoints/diff_rowcol_gin.pth")
    p.add_argument("--skip_train",action="store_true")
    args=p.parse_args()

    dirs=args.data_dirs or ["carcinogenesis"]
    ds=RowColDataset(Path(args.base), dirs)
    if not len(ds): log("empty dataset","ERROR"); sys.exit(1)

    idxs=list(range(len(ds))); random.shuffle(idxs)
    split=int(.8*len(idxs))
    train_idx, val_idx = idxs[:split], idxs[split:]
    log(f"Train={len(train_idx)}  Val={len(val_idx)}")

    feats = np.stack([ds[i].row_feat for i in train_idx])
    mu, sig = feats.mean(0), feats.std(0)+1e-8
    mu, sig = torch.tensor(mu), torch.tensor(sig)
    def norm(r): return (torch.tensor(r)-mu)/sig

    n_ops = max((int(s.col_graph.op_idx.max()) if s.col_graph.op_idx.numel() else 0)
                for s in ds)+1
    enc  = GINEncoder(n_ops, args.hidden).to('cuda' if torch.cuda.is_available() else 'cpu')
    head = DiffHead(args.hidden+8, args.hidden).to(enc.op_emb.weight.device)

    opt  = torch.optim.Adam(list(enc.parameters())+list(head.parameters()), lr=args.lr)
    crit = nn.BCELoss()

    lbls=[max(0, ds[i].label) for i in train_idx]
    pos, neg = sum(lbls), len(lbls)-sum(lbls)
    wts = [(1/pos if l==1 else 1/neg) for l in lbls]
    sampler = WeightedRandomSampler(wts,len(wts),replacement=True)
    loader = DataLoader(train_idx, batch_size=args.batch, sampler=sampler)

    best, stale = 1e9, 0
    for ep in range(1, args.epochs+1):
        if args.skip_train: break
        enc.train(); head.train()
        tr_loss=0.0
        for batch in tqdm(loader, desc=f"Ep{ep}", leave=False):
            opt.zero_grad(set_to_none=True)
            loss=0.0
            for i in batch:
                s=ds[i]
                g=s.col_graph.to(enc.op_emb.weight.device)
                feat=torch.cat([enc(g), norm(s.row_feat).to(g.x_num.device)])
                p=head(feat.unsqueeze(0))
                y=torch.tensor([float(s.label==1)],device=p.device)
                loss+=crit(p,y)
            loss.backward(); opt.step()
            tr_loss+=loss.item()
        tr_loss/=len(loader)

        enc.eval(); head.eval()
        vl=0.0
        with torch.no_grad():
            for i in val_idx:
                s=ds[i]
                g=s.col_graph.to(enc.op_emb.weight.device)
                feat=torch.cat([enc(g), norm(s.row_feat).to(g.x_num.device)])
                p=head(feat.unsqueeze(0))
                y=torch.tensor([float(s.label==1)],device=p.device)
                vl+=crit(p,y).item()
        vl/=len(val_idx)
        log(f"Ep{ep} train={tr_loss:.6f} val={vl:.6f}")

        if not args.skip_train:
            if vl<best: best, stale=vl,0
            else: stale+=1
            if stale>=12:
                log("early stop"); break

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc": enc.state_dict(),
        "head": head.state_dict(),
        "mu": mu, "sig": sig
    }, args.ckpt)
    log(f"saved → {args.ckpt}")

    enc.eval(); head.eval()
    agg={k:0.0 for k in ("row","col","cost","hyb","opt","ai")}
    cnt={k:0   for k in agg}
    with torch.no_grad():
        for s in tqdm(ds,desc="Metrics"):
            if not math.isnan(s.row_time): agg["row"]+=s.row_time; cnt["row"]+=1
            if not math.isnan(s.col_time): agg["col"]+=s.col_time; cnt["col"]+=1
            rule_rt = s.col_time if s.orig_cost>5e4 else s.row_time
            if not math.isnan(rule_rt): agg["cost"]+=rule_rt; cnt["cost"]+=1
            hyb_rt  = s.col_time if s.hybrid_flag else s.row_time
            if not math.isnan(hyb_rt): agg["hyb"]+=hyb_rt; cnt["hyb"]+=1
            best_rt = min(s.row_time, s.col_time)
            if not math.isnan(best_rt): agg["opt"]+=best_rt; cnt["opt"]+=1
            g=s.col_graph.to(enc.op_emb.weight.device)
            feat=torch.cat([enc(g), norm(s.row_feat).to(g.x_num.device)])
            choose = head(feat.unsqueeze(0)).item()>=0.5
            ai_rt   = s.col_time if choose else s.row_time
            if not math.isnan(ai_rt): agg["ai"]+=ai_rt; cnt["ai"]+=1

    for k in agg:
        agg[k]=agg[k]/cnt[k] if cnt[k] else float("nan")
    print(f"Row={agg['row']:.6f}"
          f"  Col={agg['col']:.6f}"
          f"  CostRule={agg['cost']:.6f}"
          f"  Hybrid={agg['hyb']:.6f}"
          f"  Optimal={agg['opt']:.6f}"
          f"  AI={agg['ai']:.6f}")

    # TorchScript export of only the head (takes a feature tensor)
    class HeadWrapper(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x)
    hw = HeadWrapper(head.cpu()).eval()
    out = Path(args.ckpt).with_suffix(".head.pt")
    torch.jit.script(hw).save(out)
    log(f"TorchScript head saved → {out}")

if __name__=="__main__":
    main()
