#!/usr/bin/env python3
# decision_tree_row_col_32.py
# ------------------------------------------------------------
import argparse, json, math, os, random, sys
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
from tqdm.auto import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

# --- 日志 ----------------------------------------------------
def log_info(msg):  print(f"[INFO]  {msg}")
def log_warn(msg):  print(f"[WARN]  {msg}", file=sys.stderr)

# --- 1) 递归解析行存计划，生成 32-维特征 ------------------------
def _safe_f(v):            # number or str → float
    try:
        if isinstance(v, (int, float)): return float(v)
        if isinstance(v, str):          return float(v)
    except:                             pass
    return 0.0

def _bytes_from(txt: str) -> float:
    if not txt: return 0.0
    txt = txt.strip(); factor = 1.0
    if txt[-1] in "KkMmGg":
        factor = dict(K=1e3, k=1e3, M=1e6, m=1e6, G=1e9, g=1e9)[txt[-1]]
        txt = txt[:-1]
    try:  return float(txt) * factor
    except: return 0.0

class _Agg:
    __slots__ = ("re","rp","f","rc","ec","pc","dr",
                 "sel_sum","sel_min","sel_max",
                 "ratio_sum","ratio_max",
                 "cnt","cRange","cRef","cEq","cIdx","cFull","idxUse",
                 "maxDepth","fanoutMax",
                 "grp","ord","tmp",
                 "hasLimit","hasDistinct","hasUnion","hasConst",
                 "numUnion")
    def __init__(self):
        for k in self.__slots__: setattr(self, k, 0)
        self.sel_min = 1e30

def _get_bool(v, default=False):
    if v is None: return default
    if isinstance(v, bool): return v
    if isinstance(v, str):
        v = v.lower(); return v in ("yes","true","1")
    return default

def _walk(node, a: _Agg, depth: int):
    if isinstance(node, dict):
        if "table" in node:
            tbl = node["table"]; ci = tbl.get("cost_info",{})
            re = _safe_f(tbl.get("rows_examined_per_scan"))
            rp = _safe_f(tbl.get("rows_produced_per_join"))
            fl = _safe_f(tbl.get("filtered"))
            rc = _safe_f(ci.get("read_cost"))
            ec = _safe_f(ci.get("eval_cost"))
            pc = _safe_f(ci.get("prefix_cost"))
            dr = ci.get("data_read_per_join")
            dr = _bytes_from(dr) if isinstance(dr,str) else _safe_f(dr)

            a.re += re; a.rp += rp; a.f += fl
            a.rc += rc; a.ec += ec; a.pc += pc; a.dr += dr
            if re>0:
                sel = rp/re
                a.sel_sum += sel
                a.sel_min  = min(a.sel_min, sel)
                a.sel_max  = max(a.sel_max, sel)
                a.fanoutMax = max(a.fanoutMax, sel)
            ratio = rc/ec if ec>0 else rc
            a.ratio_sum += ratio
            a.ratio_max  = max(a.ratio_max, ratio)
            a.cnt += 1

            at = tbl.get("access_type","ALL")
            if at == "const": a.hasConst = 1
            if   at == "range":   a.cRange += 1
            elif at == "ref":     a.cRef   += 1
            elif at == "eq_ref":  a.cEq    += 1
            elif at == "index":   a.cIdx   += 1
            else:                 a.cFull  += 1
            if _get_bool(tbl.get("using_index")): a.idxUse += 1

        if "limit"    in node: a.hasLimit    = 1
        if "distinct" in node: a.hasDistinct = 1
        if "union_result" in node:
            a.hasUnion = 1; a.numUnion += 1

        if "grouping_operation" in node: a.grp = 1
        if "ordering_operation" in node or _get_bool(node.get("using_filesort")): a.ord = 1
        if _get_bool(node.get("using_temporary_table")): a.tmp = 1

        for k,v in node.items():
            if k != "table": _walk(v, a, depth+1)

    elif isinstance(node, list):
        for v in node: _walk(v, a, depth+1)

    a.maxDepth = max(a.maxDepth, depth)

def extract_features32(path: Path) -> Tuple[np.ndarray, float]:
    feat = np.zeros(32, dtype=np.float32)
    try:
        j = json.loads(path.read_text())
        qb = j.get("query_block", {})
    except Exception as e:
        log_warn(f"bad JSON {path}: {e}")
        return feat, 0.0

    agg = _Agg(); _walk(qb, agg, 1)

    if agg.cnt == 0: return feat, 0.0
    inv = 1.0/agg.cnt; k = 0
    log1p = lambda x: math.log1p(max(0.0,x))

    feat[k:k+7] = [log1p(agg.re*inv), log1p(agg.rp*inv), log1p(agg.f*inv),
                   log1p(agg.rc*inv), log1p(agg.ec*inv), log1p(agg.pc*inv),
                   log1p(agg.dr*inv)];                               k += 7
    feat[k:k+5] = [agg.cRange*inv, agg.cRef*inv, agg.cEq*inv,
                   agg.cIdx*inv,   agg.cFull*inv];                   k += 5
    feat[k]   = agg.idxUse*inv;                                      k += 1
    sel_mean  = agg.sel_sum*inv
    feat[k:k+3] = [sel_mean, agg.sel_min if agg.sel_min<1e29 else 0.0, agg.sel_max]; k += 3
    feat[k:k+2] = [agg.maxDepth, agg.fanoutMax];                     k += 2
    feat[k:k+3] = [agg.grp, agg.ord, agg.tmp];                       k += 3
    feat[k:k+2] = [agg.ratio_sum*inv, agg.ratio_max];                k += 2

    qcost = _safe_f(qb.get("cost_info",{}).get("query_cost"))
    feat[k] = log1p(qcost);                                          k += 1
    root_rows = _safe_f(qb.get("rows_produced_per_join"))
    feat[k] = log1p(root_rows);                                      k += 1

    feat[k:k+3] = [agg.hasConst, agg.hasLimit, agg.hasDistinct];     k += 3
    feat[k:k+2] = [agg.hasUnion, min(agg.numUnion,10)];              k += 2
    feat[k] = sel_mean * agg.maxDepth;                               k += 1
    full_frac = agg.cFull*inv if agg.cFull else 1e-3
    feat[k] = (agg.idxUse*inv) / full_frac                           # 31

    return feat, qcost

# --- 2) 数据集封装（仅改动行计划特征提取） ---------------------
class Sample:
    __slots__ = ("row_feat","label","row_time","col_time",
                 "orig_cost","hybrid_flag")
    def __init__(self, row_feat, label, rt, ct, qc, hyb):
        self.row_feat = row_feat
        self.label    = label
        self.row_time = rt
        self.col_time = ct
        self.orig_cost= qc
        self.hybrid_flag = hyb

class RowColDataset:
    def __init__(self, base: Path, dirs: List[str]):
        self.samples: List[Sample] = []
        for ds in dirs:
            work = base/ds
            row_dir = work/"row_plans"
            csv     = work/"query_costs.csv"
            if not csv.exists():
                log_warn(f"missing {csv}"); continue
            lines = csv.read_text().splitlines()
            for ln in tqdm(lines[1:], desc=f"parse {ds}"):
                qid, lab, rt, ct, hyb = ln.split(",")[:5]
                jp = row_dir/f"{qid}.json"
                if not jp.exists(): continue
                feat, qc = extract_features32(jp)
                self.samples.append(Sample(
                    feat, int(lab) if lab else -1,
                    float(rt) if rt else math.nan,
                    float(ct) if ct else math.nan,
                    qc, float(hyb)))
        log_info(f"Loaded {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- 3) 主程序 -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/wuy/query_costs")
    ap.add_argument("--data_dirs", action="append")
    ap.add_argument("--checkpoint", default="checkpoints/best_rowcol_tree32.joblib")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--max_depth", type=int)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    # “强制列存”阈值（原始 rows_examined_per_scan）
    ap.add_argument("--rows_thresh", type=int, default=100_000)
    args = ap.parse_args()

    data_dirs = args.data_dirs or ["carcinogenesis"]
    ds = RowColDataset(Path(args.base), data_dirs)
    if len(ds) == 0: log_warn("empty dataset"); sys.exit(1)

    idxs = list(range(len(ds))); random.shuffle(idxs)
    split = int(0.8*len(idxs))
    train_idx, val_idx = idxs[:split], idxs[split:]
    log_info(f"Train={len(train_idx)}  Val={len(val_idx)}")

    X_train = np.stack([ds[i].row_feat for i in train_idx])
    y_train = np.array([1 if ds[i].label==1 else 0 for i in train_idx])
    X_val   = np.stack([ds[i].row_feat for i in val_idx])
    y_val   = np.array([1 if ds[i].label==1 else 0 for i in val_idx])

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        tree = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=42)
        tree.fit(X_train, y_train)
        val_loss = log_loss(y_val, tree.predict_proba(X_val)[:,1])
        joblib.dump(tree, args.checkpoint)
        log_info(f"Saved model → {args.checkpoint}  (val log-loss={val_loss:.6f})")
    else:
        tree = joblib.load(args.checkpoint)
        log_info(f"Loaded model {args.checkpoint}")

    # 运行时评估 -------------------------------------------------
    log_thresh = math.log1p(args.rows_thresh)
    agg = {k:0.0 for k in
           ("row","col","cost","hyb","opt","ai")}
    cnt = {k:0   for k in
           ("row","col","cost","hyb","opt","ai")}

    for s in tqdm(ds.samples, desc="metrics"):
        # baseline times
        if not math.isnan(s.row_time): agg["row"]+=s.row_time; cnt["row"]+=1
        if not math.isnan(s.col_time): agg["col"]+=s.col_time; cnt["col"]+=1
        rule_rt = s.col_time if s.orig_cost>5e4 else s.row_time
        if not math.isnan(rule_rt): agg["cost"]+=rule_rt; cnt["cost"]+=1
        hyb_rt  = s.col_time if s.hybrid_flag else s.row_time
        if not math.isnan(hyb_rt):  agg["hyb"] +=hyb_rt;  cnt["hyb"] +=1
        opt_rt  = min(s.row_time, s.col_time)
        if not math.isnan(opt_rt):  agg["opt"] +=opt_rt;  cnt["opt"] +=1

        # --- our decision ---
        if s.row_feat[0] >= log_thresh:      # 强制列存
            choose_col = True
        else:
            choose_col = tree.predict_proba(s.row_feat.reshape(1,-1))[0,1] >= .5
        ai_rt = s.col_time if choose_col else s.row_time
        if not math.isnan(ai_rt): agg["ai"] += ai_rt; cnt["ai"] += 1

    for k in agg:
        if cnt[k]: agg[k] /= cnt[k]
    print(f"Row={agg['row']:.6f}  Col={agg['col']:.6f}  "
          f"CostRule={agg['cost']:.6f}  Hybrid={agg['hyb']:.6f}  "
          f"Optimal={agg['opt']:.6f}  AI={agg['ai']:.6f}")

if __name__ == "__main__":
    main()
