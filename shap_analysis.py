#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_router.py  –  Python re-implementation of the C++ LightGBM pipeline
                   (dataset loader → LODO / CV → LightGBM → SHAP heat-map)

Usage
-----
python train_router.py \
    --base   /home/wuy/query_costs_trace \
    --dirs   tpch_sf1,tpch_sf10,tpcds_sf1,airline \
    --cv     lodo               # or 'mix5' for mixed 5-fold
    --trees  400 --max_depth 4  \
    --top_k  20                 # heat-map top-k features
"""

import argparse, json, os, sys, glob, pathlib, random
import pandas as pd
import numpy  as np
import lightgbm as lgb
import shap, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

# --------------------------------------------------------------------------- #
# 1. Utilities:                                                               #
# --------------------------------------------------------------------------- #
def read_one_dataset(dir_path: pathlib.Path):
    """Return a DataFrame with 24-d features + label + dir_tag."""
    feat_csv = dir_path / "features_140d.csv"
    meta_csv = dir_path / "query_costs.csv"
    if not feat_csv.exists() or not meta_csv.exists():
        print(f"[WARN] {dir_path} missing csv, skipped", file=sys.stderr)
        return pd.DataFrame()

    # features_24d.csv: query_id + 24 floats --------------------------------
    feat = pd.read_csv(feat_csv)
    if "query_id" not in feat.columns:
        raise ValueError(f"{feat_csv} missing query_id column")
    # query_costs.csv : query_id, use_imci, row_time, column_time, ...
    meta = pd.read_csv(meta_csv)
    keep = ["query_id", "row_time", "column_time"]
    meta = meta[keep]

    df = feat.merge(meta, on="query_id", how="inner")
    if df.empty:
        print(f"[WARN] {dir_path} join produced 0 rows", file=sys.stderr)
        return df
    df["label"] = (df["column_time"] <= df["row_time"]).astype(int)
    df["dir_tag"] = dir_path.name
    return df


def leave_one_out_dirs(dirs):
    """Generator of (train_dirs, val_dir) tuples for LODO."""
    for v in dirs:
        tr = [d for d in dirs if d != v]
        yield tr, [v]


def weighted_train_val_split(df, dir_weights, seed=42):
    """Return weighted train/val indices (mixed 5-fold)."""
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(df, df["label"]):
        yield tr_idx, va_idx, None   # weight handled elsewhere


def build_lgb_dataset(df: pd.DataFrame):
    y = df["label"].values
    X = df.drop(columns=["label", "dir_tag", "query_id",
                         "row_time", "column_time"], errors="ignore")
    return X, y


def per_dir_weight(series_dir_tag):
    cnts = series_dir_tag.value_counts().to_dict()
    total = series_dir_tag.size
    return series_dir_tag.map(lambda d: (total / cnts[d]) ** 0.2)


# --------------------------------------------------------------------------- #
# 2. Main driver                                                              #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default='/home/wuy/query_costs_trace',
                   help="root directory that contains all dataset sub-dirs")
    p.add_argument("--dirs", default=[
        # "tpcds_sf1_templates",
        "tpch_sf1",
        "tpch_sf10",
        "tpch_sf100",
        "tpcds_sf1",
        "tpcds_sf10",
        "tpcds_sf100",
        "hybench_sf1",
        "hybench_sf10",
        "airline",
        "credit",
        "carcinogenesis",
        "employee",
        "financial",
        "geneea",
        "hepatitis"
    ], help="comma-separated list of dataset sub-directories")
    p.add_argument("--cv", choices=["lodo", "mix5"], default="lodo")
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--top_k", type=int, default=20,
                   help="top-k features for SHAP heat-map")
    args = p.parse_args()

    base = pathlib.Path(args.base) #.expanduser()
    # dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    dirs = args.dirs
    if not dirs:
        p.error("--dirs is empty")

    # ------------------------------------------------------------------ #
    # 2.1 Load every dataset                                             #
    # ------------------------------------------------------------------ #
    dfs = []
    for d in dirs:
        df = read_one_dataset(base / d)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        print("[ERR] no valid datasets found", file=sys.stderr)
        sys.exit(1)
    df_all = pd.concat(dfs, ignore_index=True)
    #  Rename/alias the 32 hand-picked features so they exist by name    #
    # ------------------------------------------------------------------ #
    ALIAS = {
        "avg_rc"          : "f02",
        "qcost/bytes"     : "f96",
        "avg_pc"          : "f03",
        "min_read"        : "f23",
        "max_prefix"      : "f22",
        "avg_bytes"       : "f05",
        "join_imb"        : "f79",
        "keyPartsRatio"   : "f122",
        "keySelAvg"       : "f124",
        "keyLenAvg"       : "f123",
        "avg_rc/ec"       : "f29",
        "pc/pcDepth3"     : "f50",
        "rootRows"        : "f32",
        "avg_re"          : "f00",
        "max_rc/ec"       : "f30",
        "outerRows"       : "f31",
        "log_re_scaled"   : "f92",
        "pcDepth3"        : "f34",
        "avg_sel"         : "f16",
        "join_depth"      : "f18",
        "avg_rp"          : "f01",
        "eq_ref_ratio"    : "f09",
        "rc/re_ratio"     : "f27",
        "pc/rc_ratio"     : "f26",
        "log_qcost"       : "f94",
        "qcost/rootRows"  : "f95",
        "max_sel"         : "f15",
        "log_rootRows"    : "f91",
        "ec/re_ratio"     : "f28",
        "fanout_amp"      : "f33",
        "late_fanout"     : "f54",
        "ref_ratio"       : "f08"
    }

    # create real columns with the descriptive names
    for nice, raw in ALIAS.items():
        if nice not in df_all.columns and raw in df_all.columns:
            df_all[nice] = df_all[raw]      # cheap view assignment

    X_all, y_all = build_lgb_dataset(df_all)

    # ------------------------------------------------------------------ #
    # 2.2 CV scheme: create (train, val) splits                          #
    # ------------------------------------------------------------------ #
    splits = []
    if args.cv == "lodo":
        for tr_dirs, va_dirs in leave_one_out_dirs(dirs):
            tr_idx = df_all.index[df_all["dir_tag"].isin(tr_dirs)].to_numpy()
            va_idx = df_all.index[df_all["dir_tag"].isin(va_dirs)].to_numpy()
            splits.append((tr_idx, va_idx, va_dirs[0]))
    else:  # mix5
        for tr_idx, va_idx, _ in weighted_train_val_split(df_all, None):
            splits.append((tr_idx, va_idx, None))

    best_bal_acc = -1
    best_model   = None
    best_idx     = None

    os.makedirs("checkpoints", exist_ok=True)

    for fold, (tr_idx, va_idx, val_tag) in enumerate(splits, 1):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]

        # per-dataset weight (same exponent as C++)
        w_tr = per_dir_weight(df_all.iloc[tr_idx]["dir_tag"])

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.trees,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=42,
            importance_type="gain"
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        y_pred = clf.predict(X_va)
        bal_acc = balanced_accuracy_score(y_va, y_pred)
        tag = val_tag if val_tag else f"mix_fold{fold}"
        print(f"[Fold {fold}] val tag={tag}  BalAcc={bal_acc:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_model   = clf
            best_idx     = fold
            best_tag     = tag

        # Save each fold model
        clf.booster_.save_model(f"checkpoints/lgb_fold{fold}_{tag}.txt")

    print(f"\n[Best] fold {best_idx} ({best_tag})  BalAcc={best_bal_acc:.4f}")
    best_path = f"checkpoints/lgb_best.txt"
    best_model.booster_.save_model(best_path)
    print(f"[✓] Best model saved → {best_path}")


    # ------------------------------------------------------------------ #
    # 2·2-bis  — fixed 32-feature subset (from Table A-1), SHAP, metrics #
    # ------------------------------------------------------------------ #
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

    # ---------- fixed feature list (keep order!) -----------------------
    FIXED32 = [
        "avg_rc", "qcost/bytes", "avg_pc", "min_read", "max_prefix",
        "avg_bytes", "join_imb", "keyPartsRatio", "keySelAvg", "keyLenAvg",
        "avg_rc/ec", "pc/pcDepth3", "rootRows", "avg_re", "max_rc/ec",
        "outerRows", "log_re_scaled", "pcDepth3", "avg_sel", "join_depth",
        "avg_rp", "eq_ref_ratio", "rc/re_ratio", "pc/rc_ratio", "log_qcost",
        "qcost/rootRows", "max_sel", "log_rootRows", "ec/re_ratio",
        "fanout_amp", "late_fanout", "ref_ratio"
    ]

    missing = [f for f in FIXED32 if f not in X_all.columns]
    if missing:
        raise ValueError("These requested features are missing: " + ", ".join(missing))

    # ---------- SHAP + metrics for the full model ----------------------
    print("[SHAP] computing SHAP values for the 140-feature model …")
    sv_full = shap.TreeExplainer(best_model.booster_).shap_values(
                X_all, check_additivity=False)[0]

    # ---------- train a model on the fixed-32 subset -------------------
    X32 = X_all[FIXED32]
    clf32 = lgb.LGBMClassifier(
        objective="binary", n_estimators=args.trees, max_depth=args.max_depth,
        learning_rate=args.lr, subsample=0.7, colsample_bytree=0.8,
        random_state=123, importance_type="gain")
    clf32.fit(X32, y_all, sample_weight=per_dir_weight(df_all["dir_tag"]))
    print("[INFO] fixed-32-feature model trained")

    # ---------- helper for confusion-matrix metrics --------------------
    def show_metrics(model, X, y, tag):
        pred = model.predict(X)
        cm   = confusion_matrix(y, pred)
        acc  = accuracy_score(y, pred)
        prec = precision_score(y, pred)
        rec  = recall_score(y, pred)
        print(f"\n[{tag}]")
        print("Confusion-matrix (true rows / pred cols):\n", cm)
        print(f"accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}")
        return cm, acc, prec, rec

    cm140, acc140, prec140, rec140 = show_metrics(best_model, X_all, y_all, "140-feature")
    cm32 , acc32 , prec32 , rec32  = show_metrics(clf32    , X32   , y_all, "fixed-32")

    # ---------- SHAP heat-map for the fixed 32 -------------------------
    # ------------------------------------------------------------------
    #  evaluate 32-feature model  ➜  confusion-matrix / metrics
    # ------------------------------------------------------------------
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    X_32 = X_all[FIXED32]                 # the same rows, only 32 columns
    y_pred32 = clf32.predict(X_32)        # clf32  = your 32-feature model
    cm32 = confusion_matrix(y_all, y_pred32)
    acc32 = accuracy_score(y_all, y_pred32)
    prec32 = precision_score(y_all, y_pred32)
    rec32 = recall_score(y_all, y_pred32)

    print("\n[32-feat model] Confusion-matrix (true rows / pred cols):")
    print(cm32)
    print(f"accuracy={acc32:.4f}  precision={prec32:.4f}  recall={rec32:.4f}")
    
    expl32   = shap.TreeExplainer(clf32.booster_)
    sv32_all = expl32.shap_values(X_32, check_additivity=False)
    sv32     = sv32_all[1] if isinstance(sv32_all, list) else sv32_all  # <-- sv32 exists
    # ---------------------------------------------------------------
    # SHAP heat-map – first 16 features from Table \ref{tab:feat-top32}
    # ---------------------------------------------------------------
    TABLE16 = [
        "avg_rc", "qcost/bytes", "avg_pc", "min_read", "max_prefix",
        "avg_bytes", "join_imb", "keyPartsRatio", "keySelAvg", "keyLenAvg",
        "avg_rc/ec", "pc/pcDepth3", "rootRows", "avg_re", "max_rc/ec",
        "outerRows"
    ]

    # safety check: are they all present?
    missing = [f for f in TABLE16 if f not in X_32.columns]
    if missing:
        raise ValueError("Missing columns for heat-map: " + ", ".join(missing))

    # keep the 16 columns and keep them in the same order as TABLE16
    sv_tab16  = sv32[:, [FIXED32.index(f) for f in TABLE16]]
    heat_df16 = pd.DataFrame(sv_tab16, columns=TABLE16).T

    plt.figure(figsize=(10, 0.08*len(TABLE16) + 3))
    sns.heatmap(heat_df16, cmap="coolwarm", center=0,
                cbar_kws={"label": "SHAP value"})
    plt.yticks(rotation=0)
    plt.title("SHAP Heat-map – top 16 features (Table \\ref{tab:feat-top32})")
    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.savefig("shap_heatmap_table16.png", dpi=150)
    plt.close()
    print("[✓] shap_heatmap_table16.png written")


    # ---------- save both boosters -------------------------------------
    best_model.booster_.save_model("checkpoints/lgb_140_full.txt")
    clf32    .booster_.save_model("checkpoints/lgb_fixed32.txt")
    print("[✓] models saved:  lgb_140_full.txt  &  lgb_fixed32.txt")

    # ---------- LaTeX helper: metrics summary table --------------------
    latex_path = "metrics_full_vs_fixed32.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[htbp]\\centering\\scriptsize\n")
        f.write("\\caption{Classifier performance: all 140 features vs. "
                "hand-chosen 32 features}\\label{tab:metrics140vs32}\n")
        f.write("\\begin{tabular}{lcccc}\\toprule\n")
        f.write("Model & Accuracy & Precision & Recall & "
                "TN/FP/FN/TP \\\\\\midrule\n")
        f.write(f"140 feat & {acc140:.4f} & {prec140:.4f} & {rec140:.4f} & "
                f"{cm140[0,0]}/{cm140[0,1]}/{cm140[1,0]}/{cm140[1,1]} \\\\\n")
        f.write(f"Fixed 32 & {acc32:.4f} & {prec32:.4f} & {rec32:.4f} & "
                f"{cm32[0,0]}/{cm32[0,1]}/{cm32[1,0]}/{cm32[1,1]} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"[✓] LaTeX metrics table → {latex_path}")



if __name__ == "__main__":
    main()
