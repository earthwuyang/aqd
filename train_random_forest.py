#!/usr/bin/env python3
"""
Random Forest Training Script using the same 85-feature schema as train_lightgbm.py.
This provides an alternative model that can be tuned to be conservative
(less prone to predicting DuckDB) via class_weight.

Note: This script is for offline evaluation and comparison. It does not
integrate with the C inference path that expects a LightGBM model file.
"""

import os
import sys
import glob
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier


FEATURE_NAMES = [
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type",
    "join_type_inner", "join_type_left", "join_type_right", "join_type_full", "join_type_cross",
    "predicate_simple_eq", "predicate_range", "predicate_like", "predicate_in", "predicate_exists",
    "has_parameters", "num_cte", "max_subquery_depth", "has_recursive_cte", "has_lateral_join",
    "selectivity_high", "selectivity_medium", "selectivity_low", "cardinality_large", "cardinality_medium",
    "index_usage_likely", "partition_pruning_likely", "parallel_safe", "has_volatile_funcs", "cost_estimate_high",
    "total_projected_bytes", "avg_projected_row_fraction", "max_projected_row_fraction", "projected_column_count",
    "projected_text_columns", "projected_numeric_columns", "projected_json_columns", "output_row_width",
    "limit_value", "has_order_by_limit", "avg_scan_fraction", "max_scan_fraction", "total_rowstore_bytes_est",
    "total_columnar_bytes_est", "has_covering_index", "covering_index_score", "order_by_index_match",
    "predicate_correlation_max", "predicate_correlation_avg", "group_ndv_est", "groups_per_input_row",
    "fk_to_pk_joins", "many_to_many_joins", "star_schema_score", "topk_indexed", "topk_log_limit",
    "text_predicate_indexable", "text_predicate_nonindexable", "duckdb_table_count", "duckdb_parquet_table_count",
    "duckdb_pushdown_score", "volatile_function_count", "parallel_unsafe_function_count",
    "estimated_rows_output", "estimated_result_bytes"
]


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def load_data(data_dir, filename="training_data.csv"):
    logger = setup_logger()
    patterns = [
        os.path.join(data_dir, "training_data_*.csv"),
        os.path.join(data_dir, "training_data*.csv"),
        os.path.join(data_dir, "training_data_combined.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if files:
        dfs = []
        for f in files:
            try:
                dfp = pd.read_csv(f)
                logger.info(f"Loaded {len(dfp)} from {os.path.basename(f)}")
                dfs.append(dfp)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        if not dfs:
            raise RuntimeError("No valid CSVs loaded")
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total: {len(df)} samples from {len(dfs)} files")
    else:
        fp = os.path.join(data_dir, filename)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"No training data in {data_dir}")
        df = pd.read_csv(fp)
        logger.info(f"Loaded {len(df)} samples from {fp}")

    # Filter invalid rows
    max_time_ms = 60000
    df = df[(df['pg_time_ms'] > 0) & (df['pg_time_ms'] < max_time_ms) &
            (df['duck_time_ms'] > 0) & (df['duck_time_ms'] < max_time_ms)]

    # Ensure columns
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0

    return df


def prepare(df):
    df = df.copy()
    eps = 1e-3
    df['target'] = np.log((df['pg_time_ms'].values + eps) / (df['duck_time_ms'].values + eps))
    df['class_label'] = (df['target'] > 0).astype(int)
    return df


def evaluate(y_true, y_pred, pg_ms, duck_ms):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    lgbm_latencies = np.where(y_pred == 1, duck_ms, pg_ms)
    avg_lat = float(np.mean(lgbm_latencies))
    return acc, prec, rec, cm, avg_lat


def main():
    parser = argparse.ArgumentParser(description='Train RandomForest on 85-feature schema')
    parser.add_argument('--data-dir', default='lightgbm_training_data_new')
    parser.add_argument('--data-file', default='training_data.csv')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--max-depth', type=int, default=12)
    parser.add_argument('--min-samples-leaf', type=int, default=30)
    parser.add_argument('--class-weight-mode', choices=['fixed', 'balanced', 'balanced_tilt'], default='balanced_tilt')
    parser.add_argument('--neg-weight', type=float, default=3.0, help='Used when class-weight-mode=fixed')
    parser.add_argument('--pos-weight', type=float, default=0.6, help='Used when class-weight-mode=fixed')
    parser.add_argument('--neg-tilt', type=float, default=1.5, help='Multiply negative weight on top of balanced weights (balanced_tilt)')
    parser.add_argument('--min-loggap', type=float, default=0.0, help='Train only on samples with |log(pg/duck)| >= min-loggap')
    args = parser.parse_args()

    logger = setup_logger()
    df = load_data(args.data_dir, args.data_file)
    df = prepare(df)

    # Optionally restrict to strong-signal samples for training to reduce noise
    if args.min_loggap > 0.0:
        strong = df['target'].abs().to_numpy() >= float(args.min_loggap)
        df_train = df[strong]
        if len(df_train) < len(df) * 0.1:
            # fall back if too few samples
            df_train = df
    else:
        df_train = df

    X = df_train[FEATURE_NAMES].to_numpy()
    y = df_train['class_label'].astype(int).to_numpy()
    pg = df_train['pg_time_ms'].to_numpy()
    duck = df_train['duck_time_ms'].to_numpy()

    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val, pg_train, pg_val, duck_train, duck_val = train_test_split(
        X, y, pg, duck, test_size=0.2, random_state=42, stratify=strat
    )

    # Configure class weights
    if args.class_weight_mode == 'fixed':
        class_weight = {0: float(args.neg_weight), 1: float(args.pos_weight)}
    elif args.class_weight_mode == 'balanced':
        class_weight = 'balanced'
    else:  # balanced_tilt
        # Compute balanced weights manually then apply negative tilt
        counts = np.bincount(y_train.astype(int), minlength=2)
        total = counts.sum()
        # Avoid div by zero
        w0 = total / (2.0 * max(counts[0], 1))
        w1 = total / (2.0 * max(counts[1], 1))
        w0 *= float(args.neg_tilt)
        class_weight = {0: float(w0), 1: float(w1)}
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Fixed decision rule: predict DuckDB if proba>0.5 (equivalent to margin>0)
    proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (proba > 0.5).astype(int)

    acc, prec, rec, cm, avg_lat = evaluate(y_val, y_pred, pg_val, duck_val)
    pos_rate = float(np.mean(y_pred)) if len(y_pred) else 0.0
    logger.info("RandomForest Validation:")
    logger.info(f"  Accuracy:  {acc:.3f}")
    logger.info(f"  Precision: {prec:.3f}")
    logger.info(f"  Recall:    {rec:.3f}")
    logger.info(f"  Predicted DuckDB fraction: {pos_rate:.4f}")
    logger.info("Confusion Matrix:")
    if cm.shape == (2, 2):
        logger.info(f"                 Predicted PG  Predicted DuckDB")
        logger.info(f"  Actual PG:     {cm[0,0]:8d}     {cm[0,1]:8d}")
        logger.info(f"  Actual DuckDB: {cm[1,0]:8d}     {cm[1,1]:8d}")
    else:
        logger.info(f"  Matrix: {cm}")
    logger.info(f"Avg latency (ms) under RF routing: {avg_lat:.2f}")

    # Show top features
    importances = clf.feature_importances_
    top = np.argsort(importances)[::-1][:15]
    logger.info("Top 15 features:")
    for idx in top:
        logger.info(f"  {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

if __name__ == '__main__':
    main()
