#!/usr/bin/env python3
"""
feature_importance_rf_holdout.py  –  train on several workloads, find the
truly *predictive* features on a held-out workload, drop the weak / harmful
ones and see if the model gets better.

Example
-------
python feature_importance_rf_holdout.py \
        --data_root /home/wuy/query_costs \
        --train_dirs airline,hepatitis,hybench_sf1 \
        --test_dir  credit \
        --trees 300 \
        --metric balanced_accuracy \
        --keep_fraction 0.97
"""
import argparse, json, math, pathlib, time
import numpy as np
from tqdm import tqdm

from sklearn.ensemble      import RandomForestClassifier
from sklearn.inspection    import permutation_importance
from sklearn.metrics       import (accuracy_score,
                                   balanced_accuracy_score,
                                   f1_score,
                                   confusion_matrix,
                                   make_scorer)

# --------------------------------------------------------------------------- #
#                               constants & helpers                           #
# --------------------------------------------------------------------------- #
NUM_FEATS = 15                              # 1-to-1 with patched C++
def log1p_clip(x): return math.log1p(max(0.0, x))

def safe_f(v):
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(v)
        except: return 0.0
    return 0.0

def str_size_to_num(s: str) -> float:
    s = s.strip()
    if not s: return 0.0
    mul = 1
    if s[-1] in "Gg": mul, s = 1e9, s[:-1]
    elif s[-1] in "Mm": mul, s = 1e6, s[:-1]
    elif s[-1] in "Kk": mul, s = 1e3, s[:-1]
    try: return float(s) * mul
    except: return 0.0



class Agg:
    """Light aggregation, only what we really need."""
    __slots__ = ("re","rp","rc","ec","pc","drSum",
                 "cRange","cRef","cEq","cIdx","cFull","idxUse",
                 "grp","tmp","fanoutMax","maxDepth")
    def __init__(self):
        self.re = self.rp = self.rc = self.ec = self.pc = self.drSum = 0.0
        self.cRange = self.cRef = self.cEq = self.cIdx = self.cFull = self.idxUse = 0
        self.grp = self.tmp = False
        self.fanoutMax = 0.0
        self.maxDepth  = 0


# ----------  D F S   w a l k  ------------------------------------------
def walk15(node, a: Agg, depth=1):
    if isinstance(node, dict):
        if "table" in node:
            t  = node["table"]
            ci = t.get("cost_info", {})
            re = safe_f(t.get("rows_examined_per_scan", 0))
            rp = safe_f(t.get("rows_produced_per_join", 0))
            rc = safe_f(ci.get("read_cost", 0))
            ec = safe_f(ci.get("eval_cost", 0))
            pc = safe_f(ci.get("prefix_cost", 0))
            dr = (str_size_to_num(ci["data_read_per_join"])
                  if isinstance(ci.get("data_read_per_join"), str)
                  else safe_f(ci.get("data_read_per_join", 0)))

            a.re += re; a.rp += rp; a.rc += rc; a.ec += ec; a.pc += pc; a.drSum += dr

            at = t.get("access_type", "ALL")
            if   at == "range":  a.cRange += 1
            elif at == "ref":    a.cRef   += 1
            elif at == "eq_ref": a.cEq    += 1
            elif at == "index":  a.cIdx   += 1
            else:                a.cFull  += 1
            if t.get("using_index"): a.idxUse += 1

            if re > 0:
                a.fanoutMax = max(a.fanoutMax, rp / re)

        if node.get("grouping_operation"):    a.grp = True
        if node.get("using_temporary_table"): a.tmp = True
        for k, v in node.items():
            if k != "table":
                walk15(v, a, depth + 1)
    elif isinstance(node, list):
        for v in node:
            walk15(v, a, depth)
    a.maxDepth = max(a.maxDepth, depth)


# ----------  p l a n   –>  15-dim vector  -------------------------------
FEAT_DIM = 15
def plan2feat15(plan: dict):
    if "query_block" not in plan:
        return None
    qb = plan["query_block"]
    if "union_result" in qb:
        specs = qb["union_result"].get("query_specifications", [])
        if not specs:
            return None
        qb = specs[0]["query_block"]

    a = Agg()
    walk15(qb, a)

    root_row = safe_f(qb.get("rows_produced_per_join", 0))
    qcost    = safe_f(qb.get("cost_info", {}).get("query_cost", 0))
    inv_tab  = 1.0 / max(1, a.cRange+a.cRef+a.cEq+a.cIdx+a.cFull)   # tables

    # -------- build vector ----------
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    k = 0
    v[k] = log1p_clip(a.re);                      k += 1
    v[k] = log1p_clip(a.rp);                      k += 1
    v[k] = (a.rp / a.re) if a.re else 0.0;        k += 1
    v[k] = qcost / max(1.0, root_row);            k += 1
    v[k] = a.drSum / 1_048_576.0;                 k += 1            # MB
    v[k] = (v[k] / max(1.0, root_row));           k += 1            # MB per row
    v[k] = a.cRange * inv_tab;                    k += 1
    v[k] = a.cRef   * inv_tab;                    k += 1
    v[k] = a.cFull  * inv_tab;                    k += 1
    v[k] = a.idxUse * inv_tab;                    k += 1
    v[k] = float(a.grp and a.tmp);                k += 1
    v[k] = a.fanoutMax;                           k += 1
    v[k] = a.maxDepth;                            k += 1
    v[k] = log1p_clip(qcost);                     k += 1
    v[k] = float(qcost > 5e4);                    k += 1
    assert k == FEAT_DIM
    return v



# --------------------------------------------------------------------------- #
#                           dataset loader                                    #
# --------------------------------------------------------------------------- #
def load_dataset(root: str, dirs):
    X, y = [], []
    for d in dirs:
        plan_dir = pathlib.Path(root) / d / "row_plans"
        csv_path = pathlib.Path(root) / d / "query_costs.csv"
        if not plan_dir.is_dir() or not csv_path.exists():
            continue

        # qid → 1/0
        qlab = {}
        with open(csv_path) as fh:
            next(fh)  # skip header
            for ln in fh:
                qid, lab, *_ = ln.rstrip("\n").split(",", 2)
                qlab[qid] = int(lab == "1")

        for pf in tqdm(plan_dir.glob("*.json"), desc=f"scan {d}"):
            vec = plan2feat15(json.load(open(pf)))
            if vec is None: continue
            if pf.stem not in qlab: continue
            X.append(vec); y.append(qlab[pf.stem])
    return np.asarray(X), np.asarray(y)

# --------------------------------------------------------------------------- #
#                                 main                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",  default="/home/wuy/query_costs")
    ap.add_argument("--train_dirs", default="airline,hepatitis,hybench_sf1")
    ap.add_argument("--test_dir",   default="credit")
    ap.add_argument("--trees",      type=int, default=300)
    ap.add_argument("--metric",     choices=["accuracy",
                                             "balanced_accuracy",
                                             "f1"],
                    default="accuracy")
    ap.add_argument("--keep_fraction", type=float, default=0.95,
                    help="keep features until this fraction of total positive "
                         "importance is covered")
    ap.add_argument("--min_keep",      type=int,   default=20,
                    help="always keep at least this many features")
    ap.add_argument("--out",           default="feat_importance.csv")
    args = ap.parse_args()

    # --------------- data -----------------
    train_dirs = [d.strip() for d in args.train_dirs.split(",") if d.strip()]
    X_tr, y_tr = load_dataset(args.data_root, train_dirs)
    X_te, y_te = load_dataset(args.data_root, [args.test_dir])
    print(f"[INFO] train {X_tr.shape}   test {X_te.shape}")

    # --------------- scoring --------------
    if   args.metric == "accuracy":
        scorer = make_scorer(accuracy_score)
    elif args.metric == "f1":
        scorer = make_scorer(f1_score)
    else:
        scorer = make_scorer(balanced_accuracy_score)

    # --------------- fit base RF ----------
    rf = RandomForestClassifier(n_estimators=args.trees,
                                max_depth=None,
                                min_samples_leaf=5,
                                class_weight="balanced",
                                n_jobs=-1,
                                random_state=123)

    t0 = time.time()
    rf.fit(X_tr, y_tr)
    print(f"[INFO] RF fitted in {time.time()-t0:.1f}s")

    base_pred = rf.predict(X_te)
    print("[INFO] baseline:",
          f"acc={accuracy_score(y_te, base_pred):.4f}",
          f"bal_acc={balanced_accuracy_score(y_te, base_pred):.4f}",
          f"F1={f1_score(y_te, base_pred):.4f}")
    print(confusion_matrix(y_te, base_pred))

    # --------------- permutation importance --------------
    print("[INFO] permutation importance …")
    perm = permutation_importance(rf, X_te, y_te,
                                  n_repeats=10,
                                  scoring=scorer,
                                  n_jobs=-1,
                                  random_state=123)

    importances = perm.importances_mean          # positive ⇒ helpful
    base_score  = scorer._score_func(y_te, base_pred)
    drop_metric = base_score - importances       # >0 means drop hurts

    # save CSV for inspection
    idx_sorted = np.argsort(drop_metric)[::-1]
    with open(args.out, "w") as fh:
        fh.write("feat_id,perm_importance\n")
        for fid in idx_sorted:
            fh.write(f"{fid},{drop_metric[fid]:.6f}\n")
    print(f"[INFO] CSV written → {args.out}")

    # --------------- choose features ---------------------
    pos_idx   = [i for i,v in enumerate(drop_metric) if v > 0]
    pos_imps  = drop_metric[pos_idx]
    if len(pos_idx) == 0:                      # degenerate
        keep = list(range(NUM_FEATS))
    else:
        order    = np.argsort(pos_imps)[::-1]
        pos_idx  = np.asarray(pos_idx)[order]
        pos_imps = pos_imps[order]
        total    = pos_imps.sum()
        cum      = np.cumsum(pos_imps) / total
        # keep until cumulative ≥ keep_fraction
        cut = np.searchsorted(cum, args.keep_fraction, side="right") + 1
        cut = max(cut, args.min_keep)
        keep = list(pos_idx[:cut])

    print(f"[INFO] keeping {len(keep)}/{NUM_FEATS} "
          f"(≥{args.keep_fraction*100:.1f}% of positive importance, "
          f"min_keep={args.min_keep})")

    # --------------- re-fit on trimmed feature set -------
    rf_trim = RandomForestClassifier(n_estimators=args.trees,
                                     max_depth=None,
                                     min_samples_leaf=5,
                                     class_weight="balanced",
                                     n_jobs=-1,
                                     random_state=456)
    rf_trim.fit(X_tr[:, keep], y_tr)
    trim_pred = rf_trim.predict(X_te[:, keep])
    print("[INFO] after trim:",
          f"acc={accuracy_score(y_te, trim_pred):.4f}",
          f"bal_acc={balanced_accuracy_score(y_te, trim_pred):.4f}",
          f"F1={f1_score(y_te, trim_pred):.4f}")
    print(confusion_matrix(y_te, trim_pred))


if __name__ == "__main__":
    main()
