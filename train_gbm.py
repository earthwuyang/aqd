#!/usr/bin/env python3
"""
train_rf63.py  –  faithful Python port of train_dtree_enhanced.cpp
===============================================================
• 63-dim feature vector (byte-for-byte identical to the C++ order)
• covering-index stats via SHOW CREATE TABLE (PyMySQL)
• sample-weighted Random-Forest (scikit-learn, bootstrapped)
• JSON plan parser identical to C++ helpers
"""

import argparse, json, math, os, re, sys, time
from pathlib import Path
from typing  import Dict, Set, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import (accuracy_score, balanced_accuracy_score,
                               f1_score, confusion_matrix)
from joblib import dump, load

# --------------------------------------------------------------------- const
NUM_FEATS   = 63
GAP_EMPH    = 2.0
MODEL_FILE  = "rf_model.pkl"
rng_seed    = 42
np.random.seed(rng_seed)

# --------------------------------------------------------------------- util helpers
def log(msg, lvl="INFO"):
    sys.stderr.write(f"[{lvl}] {msg}\n")

def progress(tag:str, cur:int, tot:int, width:int=40):
    frac = cur / tot if tot else 1.0
    filled = int(frac*width)
    bar = "="*filled + " "*(width-filled)
    print(f"\r{tag} [{bar}] {int(frac*100):3d}% ({cur}/{tot})", end="",
          file=sys.stdout, flush=True)
    if cur==tot: print("", file=sys.stdout)

def log1p_clip(x:float)->float:
    return math.log1p(max(0.0,x))

def safe_f(v):
    if isinstance(v,(int,float)):  return float(v)
    if isinstance(v,str):
        try:return float(v)
        except: return 0.0
    return 0.0

_SIZE_RE = re.compile(r"^\s*([0-9.]+)\s*([KMGkmg]?)")
def str_size_to_num(s:str)->float:
    m = _SIZE_RE.match(s or "")
    if not m: return 0.0
    val, suf = float(m.group(1)), m.group(2).upper()
    return val * {"":1,"K":1e3,"M":1e6,"G":1e9}.get(suf,1)

def getBool(o:dict, key:str)->bool:
    v = o.get(key)
    if isinstance(v,bool): return v
    if isinstance(v,str):  return v.lower() in ("yes","true","1")
    return False

# --------------------------------------------------------------------- MySQL index map
try:
    import pymysql
except ModuleNotFoundError:
    pymysql = None
    log("PyMySQL not found – covering-index features will be 0", "WARN")

indexCols : Dict[str, Set[str]] = {}     # idx_name  →  {colA,colB,…}

def load_index_defs(host:str, port:int, user:str, pwd:str,
                    databases:List[str]) -> None:
    """
    Fills the global indexCols with idx-name → set(columns) across all DBs/tables.
    If MySQL or PyMySQL is unavailable we leave the map empty.
    """
    global indexCols
    indexCols.clear()
    if pymysql is None:
        return

    cre = re.compile(r"KEY\s+`([^`]+)`\s*\(\s*([^)]+)\)", re.IGNORECASE)

    for db in databases:
        try:
            conn = pymysql.connect(host=host, port=port, user=user,
                                   password=pwd, database=db,
                                   charset="utf8mb4",
                                   cursorclass=pymysql.cursors.Cursor)
        except Exception as e:
            log(f"MySQL connect '{db}' failed: {e}", "WARN")
            continue

        with conn:
            cur = conn.cursor()
            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
            for tbl in tables:
                cur.execute(f"SHOW CREATE TABLE `{tbl}`")
                row = cur.fetchone()
                if not row or len(row)<2: continue
                ddl = row[1]
                for m in cre.finditer(ddl):
                    idx = m.group(1)
                    cols = {c.strip().strip('` ') for c in m.group(2).split(',')}
                    indexCols[idx] = cols  # (same behaviour as C++ – last wins)

# --------------------------------------------------------------------- feature extraction
def walk_plan(node, agg:dict, depth:int=1):
    if isinstance(node, dict):
        if "table" in node and isinstance(node["table"], dict):
            t  = node["table"]
            ci = t.get("cost_info", {})
            re_ = safe_f(t.get("rows_examined_per_scan",0))
            rp  = safe_f(t.get("rows_produced_per_join",0))
            fl  = safe_f(t.get("filtered",0))
            rc  = safe_f(ci.get("read_cost",0))
            ec  = safe_f(ci.get("eval_cost",0))
            pc  = safe_f(ci.get("prefix_cost",0))
            dr  = safe_f(ci.get("data_read_per_join",0))
            if isinstance(ci.get("data_read_per_join"), str):
                dr = str_size_to_num(ci["data_read_per_join"])

            agg["re"] += re_;     agg["rp"] += rp;    agg["f"]  += fl
            agg["rc"] += rc;      agg["ec"] += ec;    agg["pc"] += pc
            agg["dr"] += dr;      agg["drSum"] += dr
            agg["drMax"] = max(agg["drMax"], dr)
            agg["cnt"]  += 1

            if "possible_keys" in t and isinstance(t["possible_keys"], list):
                agg["sumPK"] += len(t["possible_keys"])

            agg["maxPrefix"] = max(agg["maxPrefix"], pc)
            agg["minRead"]   = min(agg["minRead"], rc)

            if re_>0:
                sel = rp/re_
                agg["selSum"] += sel
                agg["selMin"]  = min(agg["selMin"], sel)
                agg["selMax"]  = max(agg["selMax"], sel)
                agg["fanoutMax"]=max(agg["fanoutMax"], sel)
            ratio = rc/ec if ec>0 else rc
            agg["ratioSum"] += ratio
            agg["ratioMax"]  = max(agg["ratioMax"], ratio)

            at = t.get("access_type","ALL")
            if at=="range":   agg["cRange"]+=1
            elif at=="ref":   agg["cRef"] +=1
            elif at=="eq_ref":agg["cEq"]  +=1
            elif at=="index": agg["cIdx"] +=1
            else:             agg["cFull"]+=1
            if getBool(t,"using_index"): agg["idxUse"]+=1

            # -------- covering index? --------
            if ("used_columns" in t and isinstance(t["used_columns"], list)
                and isinstance(t.get("key"), str)):
                idx_name = t["key"]
                cols = indexCols.get(idx_name)
                if cols:
                    if all((isinstance(u,str) and u in cols)
                           for u in t["used_columns"]):
                        agg["coverCount"] += 1

        # plan-level flags
        if node.get("grouping_operation"):   agg["grp"]=True
        if node.get("ordering_operation") or getBool(node,"using_filesort"):
            agg["ord"]=True
        if getBool(node,"using_temporary_table"): agg["tmp"]=True

        for k,v in node.items():
            if k!="table":
                walk_plan(v, agg, depth+1)

    elif isinstance(node, list):
        for v in node:
            walk_plan(v, agg, depth)
    agg["maxDepth"]=max(agg["maxDepth"], depth)

def plan2feat_py(plan:dict)->np.ndarray|None:
    if "query_block" not in plan: return None
    qb = plan["query_block"]
    # union handling: keep only first branch (same as C++ fast path)
    if "union_result" in qb:
        specs = qb["union_result"].get("query_specifications",[])
        if not specs: return None
        qb = specs[0]["query_block"]

    agg = dict(re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0,
               drSum=0,drMax=0,
               selSum=0, selMin=1e30, selMax=0,
               ratioSum=0, ratioMax=0,
               cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,
               sumPK=0, maxPrefix=0,minRead=1e30,
               fanoutMax=0,
               grp=False, ord=False, tmp=False,
               coverCount=0, maxDepth=0)

    walk_plan(qb, agg, 1)
    if not agg["cnt"]:
        return None
    inv = 1.0/agg["cnt"]
    qCost   = safe_f(qb.get("cost_info",{}).get("query_cost",0))
    rootRow = safe_f(qb.get("rows_produced_per_join",0))

    f=[]
    a=agg  # shorthand

    # 0-6 basic
    f += [log1p_clip(a["re"]*inv), log1p_clip(a["rp"]*inv),
          log1p_clip(a["f"]*inv),  log1p_clip(a["rc"]*inv),
          log1p_clip(a["ec"]*inv), log1p_clip(a["pc"]*inv),
          log1p_clip(a["dr"]*inv)]
    # 7-12 access fractions
    f += [a["cRange"]*inv, a["cRef"]*inv, a["cEq"]*inv,
          a["cIdx"]*inv,  a["cFull"]*inv, a["idxUse"]*inv]
    # 13-17
    f += [a["selSum"]*inv, a["selMin"], a["selMax"],
          a["maxDepth"], a["fanoutMax"]]
    # 18-20 flags
    f += [float(a["grp"]), float(a["ord"]), float(a["tmp"])]
    # 21-22 ratios
    f += [a["ratioSum"]*inv, a["ratioMax"]]
    # 23-24 cost / rows
    f += [log1p_clip(qCost), log1p_clip(rootRow)]
    # 25-27 cost ratios
    f += [log1p_clip((a["pc"]*inv)/max(1e-6,a["rc"]*inv)),
          log1p_clip((a["rc"]*inv)/max(1e-6,a["re"]*inv)),
          log1p_clip((a["ec"]*inv)/max(1e-6,a["re"]*inv))]
    # 28-29 single/multi block
    f += [float(a["cnt"]==1), float(a["cnt"]>1)]
    # 30-31 deeper
    f += [log1p_clip(a["maxDepth"]*(a["idxUse"]*inv)),
          log1p_clip((a["idxUse"]*inv)/
                     max(a["cFull"]*inv,1e-3))]
    # 32-39 misc structure
    f += [float(a["cnt"]),
          a["cnt"] and float(a["sumPK"])/a["cnt"] or 0.0,
          log1p_clip(a["maxPrefix"]),
          log1p_clip(a["minRead"] if a["minRead"]<1e30 else 0.0),
          float(a["cnt"]>1 and (a["cnt"]-1)/a["cnt"] or 0.0),
          rootRow>0 and float(a["re"])/rootRow or 0.0,
          float(a["selMax"]-a["selMin"]),
          (float(a["idxUse"])/
           max(1, a["cRange"]+a["cRef"]+a["cEq"]+a["cIdx"]))]
    # 40-41 raw cost + flag
    f += [float(qCost), float(qCost>5e4)]
    # 42-43 cover fraction (need indexCols)
    if indexCols:
        f += [a["cnt"] and float(a["coverCount"])/a["cnt"] or 0.0,
              float(a["coverCount"]==a["cnt"])]
    else:
        f += [0.0,0.0]
    # 44 seq proxy
    f += [log1p_clip(a["re"]*inv)-log1p_clip(a["selSum"]*inv)]
    # 45-46 tbl count raw+log
    f += [float(a["cnt"]), log1p_clip(a["cnt"])]
    # 47-48 possible keys
    f += [float(a["sumPK"]),
          a["cnt"] and float(a["sumPK"])/a["cnt"] or 0.0]
    # 49-50 cover counts
    f += [float(a["coverCount"]),
          a["cnt"] and float(a["coverCount"])/a["cnt"] or 0.0]
    # 51-56 fine access ratios
    f += [a["idxUse"]*inv, a["cRange"]*inv, a["cRef"]*inv,
          a["cEq"]*inv,   a["cIdx"]*inv,  a["cFull"]*inv]
    # 57-59 prefix/read skew & sel range
    f += [log1p_clip(a["maxPrefix"]*inv),
          log1p_clip(a["minRead"] if a["minRead"]<1e30 else 0.0),
          float(a["selMax"]-a["selMin"])]
    # 60-62 extremes
    f += [a["ratioMax"], a["fanoutMax"],
          (a["selMin"]>0 and
           float(a["selMax"]/a["selMin"]) or 0.0)]

    assert len(f)==NUM_FEATS
    return np.asarray(f, np.float32)

# --------------------------------------------------------------------- data loader
def load_dataset(base_dir:Path, datasets:List[str]) \
        -> Tuple[np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns  (X, y, row_t, col_t, qCost)  for further weighting / eval
    """
    feats, labels, rowT, colT, qCost = [], [], [], [], []
    for d in datasets:
        row_plan_dir = base_dir/d/"row_plans"
        csv_path     = base_dir/d/"query_costs.csv"
        if not (row_plan_dir.is_dir() and csv_path.exists()):
            log(f"missing data dir '{d}' – skipped","WARN")
            continue

        # map qid → (label,row_t,col_t,qcost flag columns 5/6 optional)
        meta={}
        with open(csv_path) as fh:
            next(fh)  # header
            for line in fh:
                parts=line.rstrip("\n").split(",",6)
                if len(parts)<4: continue
                qid,lab,rt,ct = parts[:4]
                meta[qid]= (int(lab=="1"),
                            float(rt or 0.),
                            float(ct or 0.))

        for plan_file in tqdm(list(row_plan_dir.glob("*.json")),
                              desc=f"scan {d}", unit="plan"):
            qid = plan_file.stem
            if qid not in meta: continue
            plan = json.load(open(plan_file))
            vec  = plan2feat_py(plan)
            if vec is None: continue
            label, rt, ct = meta[qid]
            feats.append(vec); labels.append(label)
            rowT.append(rt);  colT.append(ct)
            qCost.append(vec[40])        # raw query_cost already in feature
    return (np.vstack(feats).astype(np.float32),
            np.array(labels, np.int8),
            np.array(rowT),
            np.array(colT),
            np.array(qCost))



# --------------------------------------------------------------------- main
def main() -> None:
    import argparse, time, warnings
    from pathlib import Path

    import numpy as np
    import lightgbm as lgb
    from joblib import dump, load
    from sklearn.metrics import (accuracy_score,
                                 balanced_accuracy_score,
                                 f1_score, confusion_matrix)

    # -------------------------- CLI ---------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", required=True,
                    help="comma-separated training workloads")
    ap.add_argument("--base", default="/home/wuy/query_costs")

    # LightGBM hyper-params
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=0.06)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--colsample", type=float, default=0.8)

    ap.add_argument("--model", default="lgb_gap_model.pkl")
    ap.add_argument("--skip_train", action="store_true")

    # MySQL flags (unchanged)
    ap.add_argument("--mysql_host", default="127.0.0.1")
    ap.add_argument("--mysql_port", type=int, default=44444)
    ap.add_argument("--mysql_user", default="root")
    ap.add_argument("--mysql_pass", default="")
    args = ap.parse_args()

    def log(msg, lvl="INFO"):
        print(f"[{lvl}] {msg}")

    data_dirs = [d.strip() for d in args.data_dirs.split(",") if d.strip()]
    base_dir  = Path(args.base)

    # ---------------- index-metadata (unchanged) --------------------
    load_index_defs(args.mysql_host, args.mysql_port,
                    args.mysql_user, args.mysql_pass,
                    data_dirs)

    # ---------------- load raw dataset -----------------------------
    X, y_cls, row_t, col_t, qcost = load_dataset(base_dir, data_dirs)
    log(f"Loaded {len(X):,} samples")

    # ---------------- guard against zero / negative runtimes -------
    eps = 1e-6          # 1 µs  – runtime floor
    row_clip = np.clip(row_t, eps, None)
    col_clip = np.clip(col_t, eps, None)

    y_reg = np.log(row_clip) - np.log(col_clip)          # >0 ⇒ column faster

    mask_finite = np.isfinite(y_reg)
    if mask_finite.sum() < len(y_reg):
        log(f"Dropping {len(y_reg) - mask_finite.sum()} rows "
            f"with non-finite targets", "WARN")

    X       = X[mask_finite]
    y_cls   = y_cls[mask_finite]
    row_t   = row_t[mask_finite]
    col_t   = col_t[mask_finite]
    qcost   = qcost[mask_finite]
    y_reg   = y_reg[mask_finite]

    # ---------------- class–balance + cost/gap weighting -----------
    P, N = y_cls.sum(), len(y_cls) - y_cls.sum()
    w1 = len(y_cls) / (2*P) if P else 1.0
    w0 = len(y_cls) / (2*N) if N else 1.0

    gap   = np.abs(row_t - col_t)
    w_gap = 1 + np.log1p(gap)                 # smoother than raw ratio
    w_cost= 1 + np.minimum(qcost, 1e6) / 1e5
    sample_w = np.where(y_cls == 1, w1, w0) * w_gap * w_cost

    # ---------------- LightGBM model -------------------------------
    model_path = Path(args.model)
    if args.skip_train and model_path.exists():
        log(f"Loading model ← {model_path}")
        reg = load(model_path)
    else:
        log("Training LightGBM regressor …")
        t0 = time.time()
        reg = lgb.LGBMRegressor(
            n_estimators       = args.trees,
            max_depth          = args.max_depth,
            learning_rate      = args.learning_rate,
            subsample          = args.subsample,
            colsample_bytree   = args.colsample,
            objective          = "regression",
            n_jobs             = -1,
            random_state       = 42
        )

        # LightGBM warns about small datasets → suppress noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X, y_reg, sample_weight=sample_w)

        log(f"fit() done in {time.time()-t0:.1f}s")
        dump(reg, model_path)
        log(f"Model saved → {model_path}")

    # ---------------- evaluation -----------------------------------
    gap_hat    = reg.predict(X)
    choose_col = gap_hat > 0.0
    preds_cls  = choose_col.astype(int)

    acc   = accuracy_score(y_cls, preds_cls)
    bacc  = balanced_accuracy_score(y_cls, preds_cls)
    f1    = f1_score(y_cls, preds_cls)
    cm    = confusion_matrix(y_cls, preds_cls)

    log(f"accuracy={acc:.4f}  balanced={bacc:.4f}  F1={f1:.4f}")
    log("confusion matrix (rows=true)", "INFO")
    log(cm, "INFO")

    # ---------- runtime stats (same as original summary) -----------
    rt_rf   = np.mean(np.where(choose_col, col_t, row_t))
    rt_row  = row_t.mean()
    rt_col  = col_t.mean()
    rt_rule = np.mean(np.where(qcost > 5e4, col_t, row_t))
    rt_opt  = np.mean(np.minimum(row_t, col_t))

    log(f"Avg row   runtime : {rt_row:.3f}s")
    log(f"Avg col   runtime : {rt_col:.3f}s")
    log(f"Cost rule runtime : {rt_rule:.3f}s")
    log(f"LGBM gap runtime  : {rt_rf:.3f}s")
    log(f"Optimal   runtime : {rt_opt:.3f}s")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
