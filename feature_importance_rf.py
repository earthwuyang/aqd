#!/usr/bin/env python3
"""
feature_importance_rf.py  —  analyse rf_even_more.cpp models
-------------------------------------------------------------------
* Works with the JSON that rf_even_more.cpp writes (forest only)
* Handles 72-dim (or any dim) vectors automatically
* Produces two CSV files:
      importance_splitcount.csv
      importance_permutation.csv
-------------------------------------------------------------------
CLI flags
  --model=rf_model_even_more.json
  --data_root=/home/wuy/query_costs
  --dirs=DIR1,DIR2,...
  --metric=f1        (acc | f1 | precision | recall)
  --permute          run permutation importance as well
-------------------------------------------------------------------
Author:  <you>
"""
import argparse, json, math, os, sys, random, gzip, pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ----------------------------------------------------------------------
# 1. tiny helper: our extremely small RF implementation in Python
#    (reads the exact same JSON layout as the C++ version)
# ----------------------------------------------------------------------
class Forest:
    def __init__(self, model_json):
        self.trees = model_json["forest"]

    def _predict_tree(self, tree, x):
        nid = 0
        while tree[nid]["feat"] != -1:
            feat = tree[nid]["feat"]
            thr  = tree[nid]["thr"]
            nid  = tree[nid]["left"] if x[feat] < thr else tree[nid]["right"]
        return tree[nid]["prob"]

    def predict_proba(self, X):
        # vectorised but still fast enough (32 trees × ~20 000 rows ≈ 0.1 s)
        out = np.zeros(len(X), dtype=np.float32)
        for t in self.trees:
            out += np.fromiter((self._predict_tree(t, row) for row in X), dtype=np.float32)
        out /= len(self.trees)
        return out

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    # ---------- split-count importance ----------
    def split_counts(self, n_feats):
        cnt = np.zeros(n_feats, dtype=np.int32)
        for t in self.trees:
            for node in t:
                if node["feat"] != -1:
                    cnt[node["feat"]] += 1
        return cnt

# ----------------------------------------------------------------------
# 2. load dataset exactly like the C++ training code --------------------
# ----------------------------------------------------------------------
def log1p_clip(v):               # same helper
    return math.log1p(max(0.0, v))

# --------------------------------------------------------------------
# helper in feature_importance_rf.py
# --------------------------------------------------------------------
import json, math, os, re, functools, collections
from pathlib import Path

NUM_FEATS = 72               # keep in sync

# --------------------------------------------------------------------
# 1) helpers exactly like in C++
# --------------------------------------------------------------------
def safe_f(v):                        # tolerant numeric
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 0.0
    return 0.0


def log1p_clip(v):
    return math.log1p(max(0.0, v))


_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([KMGkmg]?)")
def str_size_to_num(s):
    m = _SIZE_RE.match(s)
    if not m:
        return 0.0
    x, suf = float(m.group(1)), m.group(2).lower()
    if suf == 'k':
        x *= 1e3
    elif suf == 'm':
        x *= 1e6
    elif suf == 'g':
        x *= 1e9
    return x


def get_bool(obj, key):
    v = obj.get(key)
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "1")
    return False


# --------------------------------------------------------------------
# 2) index-coverage map (reuse the global one the C++ produces)
#     👉  indexCols = { "idx_name": {"colA","colB",…}, … }
# --------------------------------------------------------------------
indexCols = {}          # the main script will populate this


# --------------------------------------------------------------------
# 3) DFS walk – identical logic  -------------------------------------
# --------------------------------------------------------------------
Agg = collections.namedtuple("Agg",
        "re rp f rc ec pc dr drSum drMax "
        "selSum selMin selMax ratioSum ratioMax "
        "cnt cRange cRef cEq cIdx cFull idxUse sumPossibleKeys "
        "maxPrefix minRead fanoutMax "
        "grp ord tmp grpTmp coverCount "
        "usedColsSum usedColsMax usedColsRatioSum usedColsRatioMax "
        "maxDepth")

# ------------------------------------------------------------------
# replace the old empty_agg() with this version
# ------------------------------------------------------------------
def empty_agg() -> Agg:
    """
    Create an Agg filled with neutral elements.

    • numeric counters / sums  → 0
    • minima                   → 1e30   (same as C++ code)
    • booleans                 → False  (0 works too)
    """
    zeros = [0.0] * len(Agg._fields)           # start with all-zero
    fld = Agg._fields.index                    # tiny helper

    # set the special minima exactly like the C++ struct
    zeros[fld('selMin')]  = 1e30
    zeros[fld('minRead')] = 1e30

    return Agg._make(zeros)


def merge(a,b):
    # element-wise “add or max/min”
    return Agg(
        a.re+b.re, a.rp+b.rp, a.f+b.f, a.rc+b.rc, a.ec+b.ec, a.pc+b.pc,
        a.dr+b.dr, a.drSum+b.drSum, max(a.drMax, b.drMax),
        a.selSum+b.selSum, min(a.selMin,b.selMin), max(a.selMax,b.selMax),
        a.ratioSum+b.ratioSum, max(a.ratioMax,b.ratioMax),
        a.cnt+b.cnt,
        a.cRange+b.cRange, a.cRef+b.cRef, a.cEq+b.cEq, a.cIdx+b.cIdx,
        a.cFull+b.cFull, a.idxUse+b.idxUse,
        a.sumPossibleKeys+b.sumPossibleKeys,
        max(a.maxPrefix,b.maxPrefix), min(a.minRead,b.minRead),
        max(a.fanoutMax,b.fanoutMax),
        a.grp or b.grp, a.ord or b.ord, a.tmp or b.tmp,
        a.grpTmp or b.grpTmp or ((a.grp or b.grp) and (a.tmp or b.tmp)),
        a.coverCount+b.coverCount,
        0,0,0,0,
        max(a.maxDepth,b.maxDepth)
    )

def walk(node, depth=1):
    agg = empty_agg()
    # ----- TABLE ----------------------------------------------------
    if isinstance(node, dict) and "table" in node:
        t = node["table"]
        ci = t.get("cost_info", {})
        re_ = safe_f(t.get("rows_examined_per_scan"))
        rp  = safe_f(t.get("rows_produced_per_join"))
        fl  = safe_f(t.get("filtered"))
        rc  = safe_f(ci.get("read_cost"))
        ec  = safe_f(ci.get("eval_cost"))
        pc  = safe_f(ci.get("prefix_cost"))
        dr  = 0.0
        if "data_read_per_join" in ci:
            v = ci["data_read_per_join"]
            dr = str_size_to_num(v) if isinstance(v,str) else safe_f(v)

        sel = rp/re_ if re_>0 else 0.0
        ratio = rc/ec if ec>0 else rc

        agg = agg._replace(
            re = re_, rp = rp, f = fl,
            rc = rc, ec = ec, pc = pc,
            dr = dr, drSum = dr, drMax = dr,
            selSum = sel, selMin = sel, selMax = sel,
            ratioSum = ratio, ratioMax = ratio,
            cnt = 1,
            maxPrefix = pc,
            minRead = rc,
            fanoutMax = sel
        )

        at = t.get("access_type","ALL")
        if   at == "range":  agg = agg._replace(cRange=1)
        elif at == "ref":    agg = agg._replace(cRef=1)
        elif at == "eq_ref": agg = agg._replace(cEq=1)
        elif at == "index":  agg = agg._replace(cIdx=1)
        else:                agg = agg._replace(cFull=1)

        if get_bool(t,"using_index"):
            agg = agg._replace(idxUse=1)

        if isinstance(t.get("possible_keys"), list):
            agg = agg._replace(sumPossibleKeys=len(t["possible_keys"]))

        # covering index?
        if isinstance(t.get("used_columns"), list) and isinstance(t.get("key"), str):
            idxname = t["key"]
            cols = indexCols.get(idxname)
            if cols:
                if all(col in cols for col in t["used_columns"] if isinstance(col,str)):
                    agg = agg._replace(coverCount=1)

    # ----- flags on this node ---------------------------------------
    if isinstance(node, dict):
        if "grouping_operation" in node:
            agg = agg._replace(grp=True)
        if "ordering_operation" in node or get_bool(node,"using_filesort"):
            agg = agg._replace(ord=True)
        if get_bool(node,"using_temporary_table"):
            agg = agg._replace(tmp=True)

    # ----- recurse ---------------------------------------------------
    if isinstance(node, dict):
        for k,v in node.items():
            if k == "table":        # already processed
                continue
            agg_child = walk(v, depth+1)
            agg = merge(agg, agg_child)

    elif isinstance(node, list):
        for el in node:
            agg = merge(agg, walk(el, depth))

    # update maxDepth
    if depth > agg.maxDepth:
        agg = agg._replace(maxDepth=depth)
    return agg


# --------------------------------------------------------------------
# 4) full plan → 72-dim vector (faithful copy of C++)
# --------------------------------------------------------------------
def plan2feat_py(plan):
    if "query_block" not in plan:
        return None
    rootQB = plan["query_block"]

    blocks = []
    qCost = rootRow = 0.0
    if "union_result" in rootQB:
        specs = rootQB["union_result"]["query_specifications"]
        if not isinstance(specs, list) or not specs:
            return None
        for sp in specs:
            qb = sp["query_block"]
            blocks.append(qb)
            qCost   += safe_f(qb.get("cost_info",{}).get("query_cost"))
            rootRow += safe_f(qb.get("rows_produced_per_join"))
    else:
        blocks.append(rootQB)
        qCost   = safe_f(rootQB.get("cost_info",{}).get("query_cost"))
        rootRow = safe_f(rootQB.get("rows_produced_per_join"))

    agg = empty_agg()
    for qb in blocks:
        agg = merge(agg, walk(qb, 1))
    if agg.cnt == 0:
        return None

    inv = 1.0/agg.cnt
    f = []

    # 0-62  (identical order to C++)
    f += [log1p_clip(agg.re*inv), log1p_clip(agg.rp*inv), log1p_clip(agg.f*inv),
          log1p_clip(agg.rc*inv), log1p_clip(agg.ec*inv), log1p_clip(agg.pc*inv),
          log1p_clip(agg.dr*inv),
          agg.cRange*inv, agg.cRef*inv, agg.cEq*inv, agg.cIdx*inv, agg.cFull*inv, agg.idxUse*inv,
          agg.selSum*inv, agg.selMin, agg.selMax, agg.maxDepth, agg.fanoutMax,
          float(agg.grp), float(agg.ord), float(agg.tmp),
          agg.ratioSum*inv, agg.ratioMax,
          log1p_clip(qCost), log1p_clip(rootRow),
          log1p_clip((agg.pc*inv)/(agg.rc*inv+1e-6)),
          log1p_clip((agg.rc*inv)/(agg.re*inv+1e-6)),
          log1p_clip((agg.ec*inv)/(agg.re*inv+1e-6)),
          float(agg.cnt==1), float(agg.cnt>1),
          log1p_clip(agg.maxDepth*(agg.idxUse*inv)),
          log1p_clip((agg.idxUse*inv) / max(1e-3, agg.cFull*inv)),
          float(agg.cnt),
          (agg.cnt and agg.sumPossibleKeys/agg.cnt) or 0.0,
          log1p_clip(agg.maxPrefix),
          log1p_clip(agg.minRead if agg.minRead<1e30 else 0.0),
          (agg.cnt>1 and (agg.cnt-1)/agg.cnt) or 0.0,
          (rootRow>0 and agg.re/rootRow) or 0.0,
          agg.selMax-agg.selMin,
          (agg.idxUse and (agg.idxUse) / max(1e-6, agg.cRange+agg.cRef+agg.cEq+agg.cIdx)) or 0.0,
          qCost,
          float(qCost>5e4),
          (agg.cnt and agg.coverCount/agg.cnt) or 0.0,
          float(agg.coverCount==agg.cnt),
          log1p_clip(agg.re*inv)-log1p_clip(agg.selSum*inv),
          float(agg.cnt),
          log1p_clip(agg.cnt),
          float(agg.sumPossibleKeys),
          (agg.cnt and agg.sumPossibleKeys/agg.cnt) or 0.0,
          float(agg.coverCount),
          (agg.cnt and agg.coverCount/agg.cnt) or 0.0,
          agg.idxUse*inv, agg.cRange*inv, agg.cRef*inv,
          agg.cEq*inv, agg.cIdx*inv, agg.cFull*inv,
          log1p_clip(agg.maxPrefix*inv),
          log1p_clip(agg.minRead if agg.minRead<1e30 else 0.0),
          agg.selMax-agg.selMin,
          agg.ratioMax,
          agg.fanoutMax,
          (agg.selMin>0 and agg.selMax/agg.selMin) or 0.0]

    # 63-71  (new extras)
    f += [ float(qCost>5e4 and rootRow<1e3),
           float(qCost<2e4 and (agg.dr>6.4e7 or agg.re>5e6)),
           log1p_clip(qCost/(agg.re*inv+1e-6)),
           log1p_clip(qCost/((agg.dr*inv)/1048576+1e-6)),
           (agg.selMin>0 and agg.selMax/agg.selMin) or 0.0 ]

    mbAll = agg.drSum/1048576.0
    f += [ log1p_clip(mbAll),
           float(agg.grp and agg.tmp),
           log1p_clip(agg.drMax/1048576.0),
           float(agg.tmp and agg.drMax>64*1024*1024) ]

    assert len(f) == NUM_FEATS
    return np.asarray(f, dtype=np.float32)

# --------------------------------------------------------------------
# 5) public API for the outer script ---------------------------------
# --------------------------------------------------------------------
def load_plan2feat(path: str) -> np.ndarray | None:
    with open(path) as fh:
        plan = json.load(fh)

    # skip “Impossible WHERE”, “No matching rows”, etc.
    if (plan.get('query_block', {})
            .get('message', '')
            .lower().startswith('impossible')):
        return None                 # ←  just signal “skip”

    vec = plan2feat_py(plan)        # may still raise for other reasons
    return vec


def load_vectors_on_the_fly(data_root, dirs):
    X, y = [], []
    for d in dirs:
        csv = Path(data_root)/d/"query_costs.csv"
        with open(csv) as fp:
            header = fp.readline()
            for line in fp:
                cols = line.rstrip("\n").split(",", 8)
                qid   = cols[0]
                label = int(cols[1])
                plan  = Path(data_root)/d/"row_plans"/f"{qid}.json"
                vec = load_plan2feat(plan)
                if vec is None:
                    continue
                X.append(vec)
                y.append(label)
    X = np.vstack(X)
    y = np.asarray(y, dtype=np.int8)
    return X, y


# ----------------------------------------------------------------------
# 3. permutation importance --------------------------------------------
# ----------------------------------------------------------------------
def permutation_importance(model, X, y, metric="f1", n_rounds=4, seed=0):
    rng = np.random.default_rng(seed)
    if metric == "acc":
        scorer = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    elif metric == "precision":
        scorer = lambda y_true, y_pred: precision_score(y_true, y_pred)
    elif metric == "recall":
        scorer = lambda y_true, y_pred: recall_score(y_true, y_pred)
    else:
        scorer = lambda y_true, y_pred: f1_score(y_true, y_pred)

    base = scorer(y, model.predict(X))
    drops = np.zeros(X.shape[1], dtype=np.float64)

    for f in tqdm(range(X.shape[1]), desc="permute", leave=False):
        # repeat a few times for stability
        score_drop = 0.0
        for _ in range(n_rounds):
            X_pert = X.copy()
            rng.shuffle(X_pert[:, f])          # in-place permutation
            pert = scorer(y, model.predict(X_pert))
            score_drop += base - pert
        drops[f] = score_drop / n_rounds
    return drops, base

# ----------------------------------------------------------------------
# 4. tie everything together -------------------------------------------
# ----------------------------------------------------------------------
def main():
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument("--model",   default="rf_model_even_more.json")
    argp.add_argument("--data_root", default="/home/wuy/query_costs")
    argp.add_argument("--dirs",      required=True, help="comma-separated list of DIR names")
    argp.add_argument("--metric",    default="f1",  choices=["acc","f1","precision","recall"])
    argp.add_argument("--permute",   action="store_true", help="also run permutation importance")
    argp.add_argument("--rounds",    type=int, default=4, help="··for permutation importance")
    args = argp.parse_args()

    # ---- 1) load model ------------------------------------------------
    with open(args.model) as fp:
        model = Forest(json.load(fp))

    # ---- 2) load cached vectors --------------------------------------
    dirs = [d.strip() for d in args.dirs.split(",")]
    X, y = load_vectors_on_the_fly(args.data_root, dirs)
    n_feats = X.shape[1]
    print(f"√ loaded {len(X):,} samples – {n_feats} features")

    # ---- 3) simple split-count importance ----------------------------
    split_cnt = model.split_counts(n_feats)
    df_split  = (pd.Series(split_cnt, name="split_count")
                   .sort_values(ascending=False).to_frame())
    df_split.index.name = "feat_id"
    df_split.to_csv("importance_splitcount.csv")
    print("→ importance_splitcount.csv written")

    # ---- 4) permutation importance (optional) ------------------------
    if args.permute:
        drops, base = permutation_importance(model, X, y,
                                             metric=args.metric,
                                             n_rounds=args.rounds)
        df_perm = (pd.Series(drops, name="Δ"+args.metric)
                     .sort_values(ascending=False).to_frame())
        df_perm.index.name = "feat_id"
        df_perm.to_csv("importance_permutation.csv")
        print(f"→ importance_permutation.csv written   (base {args.metric}={base:.4f})")

if __name__ == "__main__":
    main()
