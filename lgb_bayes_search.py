#!/usr/bin/env python3
# ------------------------------------------------------------
# bayes_search  (all 16 dirs, multi-thread Optuna)
# ------------------------------------------------------------
import os, argparse, yaml, warnings
from pathlib import Path
from typing  import List, Tuple
import numpy as np
import pandas as pd
import optuna, lightgbm as lgb

# ---------- 全局常量 ----------
NUM_FEATS = 32
TOP32_IDX = [0, 1, 4, 6, 7, 9, 10, 12,
             15,16,17,18,19,20,23,24,
             27,28,29,30,31,32,33,34,
             40,43,45,48,52,57,60,79]

DATA_DIRS = [
    "tpch_sf1","tpch_sf10","tpch_sf100",
    "tpcds_sf1","tpcds_sf10","tpcds_sf100",
    "hybench_sf1","hybench_sf10",
    "airline","credit","carcinogenesis",
    "employee","financial","geneea","hepatitis"
]

class Sample:
    __slots__ = ("feat","row_t","col_t")
    def __init__(self, feat:np.ndarray, row_t:float, col_t:float):
        self.feat, self.row_t, self.col_t = feat, row_t, col_t

# ---------- 读取单个数据目录 ----------
def load_dir(root:Path, subdir:str) -> List[Sample]:
    feat_csv  = root/subdir/"features_140d.csv"
    cost_csv  = root/subdir/"query_costs.csv"   # row_time,column_time 在这里
    if not feat_csv.is_file():
        warnings.warn(f"{feat_csv} not found, skip")
        return []

    # 1) 140 维特征
    df_feat = pd.read_csv(feat_csv)
    if "query_id" not in df_feat.columns:
        warnings.warn(f"{feat_csv} 缺少 query_id，跳过")
        return []
    feats_140 = df_feat.iloc[:, 3:143].values.astype(np.float32)
    feat32    = feats_140[:, TOP32_IDX]
    qid_feat  = df_feat["query_id"]

    # 2) 执行时间
    if not cost_csv.is_file():
        warnings.warn(f"{cost_csv} not found, skip dir")
        return []
    df_cost = pd.read_csv(cost_csv, usecols=["query_id","row_time","column_time"])

    # 兼容 column_time / col_time
    if "column_time" not in df_cost.columns:
        if "col_time" in df_cost.columns:
            df_cost = df_cost.rename(columns={"col_time":"column_time"})
        else:
            warnings.warn(f"{cost_csv} 缺 column_time，跳过目录")
            return []

    merged = pd.merge(qid_feat.to_frame(), df_cost,
                      on="query_id", how="inner")
    if merged.empty:
        warnings.warn(f"{subdir}: no overlap rows after merge, skip")
        return []

    # 对齐 feature 行顺序
    merged = merged.set_index("query_id")
    aligned_feat32 = feat32[[qid in merged.index for qid in qid_feat]]

    samples = [Sample(f, r, c)
               for f, r, c in zip(aligned_feat32,
                                  merged["row_time"].values.astype(np.float32),
                                  merged["column_time"].values.astype(np.float32))]
    print(f"[INFO] {subdir:12}  loaded {len(samples):7,} samples")
    return samples

# ---------- 样本 → 矩阵 ----------
def build_dataset(samples:List[Sample], lam:float, scale:float):
    X  = np.asarray([s.feat  for s in samples], dtype=np.float32)
    yR = np.log1p([s.row_t for s in samples], dtype=np.float32)
    yC = np.log1p([s.col_t for s in samples], dtype=np.float32)
    diff = np.abs([s.row_t - s.col_t for s in samples])
    w   = 1.0 + lam*np.tanh(np.asarray(diff)/scale)
    return X,yR,yC,w

# ---------- Optuna 目标 ----------
def make_objective(train, valid, early_round):
    Xtr,yRtr,yCtr,wtr = train
    Xva,yRva,yCva,wva = valid

    def fit_one(X, y, w, params):
        dtrain = lgb.Dataset(X, y, weight=w)
        dval   = lgb.Dataset(Xva, yRva if y is yRtr else yCva,
                             weight=wva, reference=dtrain)
        return lgb.train(params, dtrain, 2000,
                         valid_sets=[dval],
                        #  early_stopping_rounds=early_round,
                        #  verbose_eval=False
                         )

    def objective(trial:optuna.Trial):
        params = {
            "objective":"regression_l2","metric":"l2","boosting":"gbdt",
            "verbosity":-1,"num_threads":0,
            "learning_rate": trial.suggest_float("lr",0.01,0.1,log=True),
            "num_leaves"   : trial.suggest_int("num_leaves",128,1024,log=True),
            "max_depth"    : trial.suggest_int("max_depth",-1,18),
            "feature_fraction":trial.suggest_float("feature_fraction",0.6,1.0),
            "bagging_fraction":trial.suggest_float("bagging_fraction",0.6,1.0),
            "bagging_freq":1,
            "min_data_in_leaf":trial.suggest_int("min_leaf",20,200),
            "lambda_l1": trial.suggest_float("l1", 1e-8, 5.0, log=True),
            "lambda_l2": trial.suggest_float("l2", 1e-8, 5.0, log=True),
            "seed":trial.number
        }
        gbm_row = fit_one(Xtr,yRtr,wtr,params)
        params["seed"] += 777
        gbm_col = fit_one(Xtr,yCtr,wtr,params)

        pR = gbm_row.predict(Xva, num_iteration=gbm_row.best_iteration)
        pC = gbm_col.predict(Xva, num_iteration=gbm_col.best_iteration)
        return np.mean(((pR-yRva)**2 + (pC-yCva)**2)/2)

    return objective

# ---------- 主函数 ----------
def main(args):
    root = Path(args.root).expanduser()
    all_samples:List[Sample] = []
    for sub in DATA_DIRS:
        all_samples += load_dir(root, sub)

    if not all_samples:
        print(">> No data, exit"); return

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_samples))
    split = int(len(all_samples)*0.8)
    train_samp = [all_samples[i] for i in idx[:split]]
    valid_samp = [all_samples[i] for i in idx[split:]]

    train = build_dataset(train_samp, args.cost_lambda, args.cost_scale)
    valid = build_dataset(valid_samp, args.cost_lambda, args.cost_scale)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(make_objective(train,valid,args.early_round),
                   n_trials=args.n_trials,
                   n_jobs=args.n_jobs)
    print("[BEST]  value =", study.best_value)
    with open(args.out,"w") as f: yaml.safe_dump(study.best_trial.params,f)
    print("Saved best params -->", args.out)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/wuy/query_costs_trace",
        help="根目录，例如  /home/wuy/query_costs_trace")
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--n_jobs",   type=int, default=30)
    ap.add_argument("--cost_lambda", type=float, default=2.0)
    ap.add_argument("--cost_scale",  type=float, default=1e6)
    ap.add_argument("--early_round", type=int, default=80)
    ap.add_argument("--out", default="best_params.yaml")
    args = ap.parse_args()
    main(args)
