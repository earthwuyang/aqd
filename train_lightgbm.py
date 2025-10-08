    #!/usr/bin/env python3
"""
LightGBM Training Script - 85-Feature Schema (v2.2.0).
Implements self-paced, Taylor-weighted boosting with focal loss to better handle
class imbalance when routing queries between PostgreSQL and DuckDB.
"""

import os
import sys
import json
import glob
import inspect
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import log_loss, balanced_accuracy_score
import logging
from datetime import datetime
import argparse


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _clip_probabilities(p, eps=1e-6):
    return np.clip(p, eps, 1.0 - eps)


# Feature names must match the kernel (see PreOptFeaturesToArray)
BASE_FEATURE_NAMES = [
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

# DERIVED FEATURES DISABLED - C code doesn't compute them, causes segfault
# Re-enable after implementing derived features in preopt_feature_extractor.c
DERIVED_FEATURE_SPECS = [
    # ("idx_cover_gap", -1),
    # ("topk_idx_gap", -1),
    # ("covering_idx_effect", -1),
    # ("col_ratio", 1),
    # ("duckdb_suitability", 1),
    # ("pushdown_intensity", 1),
    # ("pg_parallel_edge", -1),
    # ("duck_parallel_edge", 1),
    # ("scan_to_result", 1),
    # ("project_frac_gap", 1),
    # ("fanout", 1),
    # ("star_col_gain", 1),
    # ("topk_no_idx", 1),
]

DERIVED_FEATURE_NAMES = [name for name, _ in DERIVED_FEATURE_SPECS]

# Use only base features to match C code (85 features)
FEATURE_NAMES = BASE_FEATURE_NAMES  # + DERIVED_FEATURE_NAMES

FEATURE_SCHEMA_VERSION = "v2.2.0"

class LightGBMTrainer:
    def __init__(
        self,
        data_dir="lightgbm_training_data",
        model_dir="lightgbm_models",
        *,
        adaptive_margin=False,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        # Raw-score threshold (log-odds). Keep fixed margin at 0.0.
        self.threshold = 0.0
        # Adaptive (floating) margin toggle (per-bucket thresholds on validation)
        self.adaptive_margin = bool(adaptive_margin)
        self.adaptive_thresholds = None  # dict[int->float] learned on validation
        # Features to form buckets for adaptive margins (must exist in FEATURE_NAMES)
        self.adaptive_features = [
            'index_usage_likely',
            'has_covering_index',
            'order_by_index_match',
            'selectivity_high',
        ]
        self.num_epochs = 3
        self.trees_per_epoch = 1000
        self.base_learning_rate = 0.03
        self.runtime_alpha = 0.6
        self.focal_alpha = 0.25  # retained for experimentation; not used directly in current weights
        self.focal_gamma = 2.0   # retained for experimentation; not used directly in current weights
        # Global class weighting to discourage positives (DuckDB=1)
        # Applied directly instead of inverse-frequency equalization.
        self.neg_class_weight_global = 1.6  # weight for label 0 (PG better)
        self.pos_class_weight_global = 0.7  # weight for label 1 (DuckDB better)
        self.weight_schedule = [
            {
                'gamma_pos': 1.0,
                'gamma_neg': 2.8,
                'tau': 0.35,
                'hard_neg_threshold': 0.50,
                'hard_neg_boost': 3.5,
                'fp_boost': 1.2,
                'clip': (0.20, 5.5),
            },
            {
                'gamma_pos': 1.0,
                'gamma_neg': 3.0,
                'tau': 0.35,
                'hard_neg_threshold': 0.52,
                'hard_neg_boost': 4.0,
                'fp_boost': 1.3,
                'clip': (0.15, 5.0),
            },
            {
                'gamma_pos': 1.1,
                'gamma_neg': 3.2,
                'tau': 0.35,
                'hard_neg_threshold': 0.55,
                'hard_neg_boost': 5.0,
                'fp_boost': 1.4,
                'clip': (0.10, 4.8),
            },
            {
                'gamma_pos': 1.1,
                'gamma_neg': 3.3,
                'tau': 0.35,
                'hard_neg_threshold': 0.58,
                'hard_neg_boost': 5.0,
                'fp_boost': 1.5,
                'clip': (0.08, 4.5),
            },
        ]
        self.gap_sharpener_k = 6.0
        self.gap_sharpener_m = 0.20
        # Monotone constraints to bias decisions conservatively.
        # Negative means increasing the feature should reduce the raw score (favor PG),
        # Positive means increasing the feature should increase the raw score (favor DuckDB).
        constraint_map = {
            'index_usage_likely': -1,
            'has_covering_index': -1,
            'order_by_index_match': -1,
            'selectivity_high': -1,
            'selectivity_low': +1,
            'total_columnar_bytes_est': +1,
            'duckdb_pushdown_score': +1,
            'cardinality_large': +1,
            'avg_scan_fraction': +1,
            'max_scan_fraction': +1,
        }
        self.monotone_constraints = [constraint_map.get(name, 0) for name in FEATURE_NAMES]

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Focal loss configuration
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        sig = inspect.signature(lgb.train)
        self._supports_custom_objective = 'fobj' in sig.parameters

    def _compute_bucket_ids(self, df):
        """Compute integer bucket ids from selected boolean features for adaptive margin."""
        bits = np.zeros(len(df), dtype=np.int64)
        for i, fname in enumerate(self.adaptive_features):
            if fname in df.columns:
                col = (df[fname].to_numpy() > 0).astype(np.int64)
            else:
                col = np.zeros(len(df), dtype=np.int64)
            bits |= (col << i)
        return bits

    @staticmethod
    def _tune_thresholds_by_bucket(raw_scores, bucket_ids, pg_ms, duck_ms, *, labels=None,
                                   objective='latency', fbeta=1.0, q=101, min_bucket=50):
        """Tune per-bucket raw-score thresholds.

        objective:
          - 'latency': minimize average latency in each bucket
          - 'precision': maximize precision in each bucket
          - 'f1': maximize F1 in each bucket
          - 'fbeta': maximize F-beta (uses provided fbeta)
        """
        thresholds = {}
        uniq = np.unique(bucket_ids)
        for b in uniq:
            idx = (bucket_ids == b)
            if np.count_nonzero(idx) < min_bucket:
                thresholds[int(b)] = 0.0
                continue
            scores_b = raw_scores[idx]
            pg_b = pg_ms[idx]
            duck_b = duck_ms[idx]
            y_b = None if labels is None else labels[idx]
            # Build quantile grid per-bucket
            try:
                qs = np.linspace(0.0, 1.0, q)
                grid = np.unique(np.quantile(scores_b, qs))
            except Exception:
                grid = np.array([0.0])
            best_t = 0.0
            # Initialize depending on objective
            if objective == 'latency':
                best_score = float('inf')
            else:
                best_score = -float('inf')

            for t in grid:
                pred_duck = scores_b > t
                if objective == 'latency':
                    lats = np.where(pred_duck, duck_b, pg_b)
                    score = -float(np.mean(lats))  # negative for argmax
                else:
                    if y_b is None or y_b.size == 0:
                        continue
                    # Compute precision/recall
                    tp = float(np.sum((pred_duck == 1) & (y_b == 1)))
                    fp = float(np.sum((pred_duck == 1) & (y_b == 0)))
                    fn = float(np.sum((pred_duck == 0) & (y_b == 1)))
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    if objective == 'precision':
                        score = prec
                    elif objective == 'f1' or objective == 'fbeta':
                        beta2 = fbeta * fbeta
                        denom = (beta2 * prec + rec)
                        fscore = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
                        score = fscore
                    else:
                        score = prec

                if score > best_score:
                    best_score = score
                    best_t = float(t)

            thresholds[int(b)] = best_t
        return thresholds

    @staticmethod
    def _apply_bucket_thresholds(raw_scores, bucket_ids, thresholds):
        """Apply per-bucket thresholds to get predicted_duckdb mask."""
        pred = np.zeros_like(raw_scores, dtype=bool)
        # Vectorized apply by iterating buckets
        for b, t in thresholds.items():
            mask = (bucket_ids == b)
            if np.any(mask):
                pred[mask] = raw_scores[mask] > t
        return pred

    def _focal_loss_objective(self, preds, train_data):
        """Custom focal loss objective returning gradient and Hessian."""
        labels = train_data.get_label()
        grad, hess = self._compute_focal_grad_hess(
            preds,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        return grad, hess

    @staticmethod
    def _compute_focal_grad_hess(preds, labels, *, alpha=0.25, gamma=2.0, eps=1e-9):
        preds = preds.astype(np.float64)
        labels = labels.astype(np.float64)
        p = sigmoid(preds)
        p = np.clip(p, eps, 1.0 - eps)

        grad = np.zeros_like(p)
        hess = np.zeros_like(p)

        pos_mask = labels > 0.5
        neg_mask = ~pos_mask

        if np.any(pos_mask):
            p_pos = p[pos_mask]
            one_minus = np.clip(1.0 - p_pos, eps, 1.0)
            ce = -np.log(p_pos)
            w = alpha * (one_minus ** gamma)
            dw_dz = -alpha * gamma * (one_minus ** (gamma - 1)) * p_pos * one_minus
            d2w_dz2 = (
                alpha * gamma * (gamma - 1) * (one_minus ** (gamma - 2)) * (p_pos * one_minus) ** 2
                - alpha * gamma * (one_minus ** (gamma - 1)) * p_pos * one_minus * (1.0 - 2.0 * p_pos)
            )
            grad[pos_mask] = w * (p_pos - 1.0) + ce * dw_dz
            hess[pos_mask] = 2.0 * (p_pos - 1.0) * dw_dz + w * p_pos * one_minus + ce * d2w_dz2

        if np.any(neg_mask):
            p_neg = p[neg_mask]
            q = np.clip(p_neg, eps, 1.0 - eps)
            one_minus_q = np.clip(1.0 - p_neg, eps, 1.0)
            ce = -np.log(one_minus_q)
            w = (1.0 - alpha) * (q ** gamma)
            dw_dz = (1.0 - alpha) * gamma * (q ** (gamma - 1)) * q * one_minus_q
            d2w_dz2 = (
                (1.0 - alpha)
                * gamma
                * (
                    (gamma - 1.0) * (q ** (gamma - 2)) * (q * one_minus_q) ** 2
                    + (q ** (gamma - 1)) * q * one_minus_q * (1.0 - 2.0 * q)
                )
            )
            grad[neg_mask] = w * p_neg + ce * dw_dz
            hess[neg_mask] = 2.0 * p_neg * dw_dz + w * p_neg * one_minus_q + ce * d2w_dz2

        hess = np.clip(hess, 1e-6, None)
        return grad.astype(np.float64), hess.astype(np.float64)

    def load_data(self, filename="training_data.csv"):
        """Load training data from CSV or multiple CSV files"""
        # Always try to load ALL training data files first
        patterns = [
            os.path.join(self.data_dir, "training_data_*.csv"),
            os.path.join(self.data_dir, "training_data*.csv"),
            os.path.join(self.data_dir, "training_data_combined.csv")
        ]

        files = []
        for pattern in patterns:
            pattern_files = glob.glob(pattern)
            files.extend(pattern_files)

        # Remove duplicates and sort
        files = sorted(list(set(files)))

        if files:
            # Load and concatenate all CSV files
            dfs = []
            for file in files:
                try:
                    df_part = pd.read_csv(file)
                    dfs.append(df_part)
                    self.logger.info(f"Loaded {len(df_part)} samples from {os.path.basename(file)}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {os.path.basename(file)}: {e}")

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Total: {len(df)} samples from {len(dfs)} files")
            else:
                self.logger.error("No valid training data files could be loaded")
                sys.exit(1)
        else:
            # Fallback to specific file if no pattern matches found
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                self.logger.info(f"Loaded {len(df)} samples from {filepath}")
            else:
                self.logger.error(f"No training data found in {self.data_dir}")
                self.logger.info(f"Looking for: training_data*.csv files or {filename}")
                sys.exit(1)

        # Validate feature columns exist
        missing_features = [f for f in BASE_FEATURE_NAMES if f not in df.columns]
        if missing_features:
            self.logger.warning(f"Missing features in data: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0

        # Filter out queries where either engine failed or took too long
        max_time_ms = 60000  # 60 seconds
        df = df[(df['pg_time_ms'] > 0) & (df['pg_time_ms'] < max_time_ms) &
                (df['duck_time_ms'] > 0) & (df['duck_time_ms'] < max_time_ms)]

        self.logger.info(f"After filtering: {len(df)} valid samples")
        self.logger.info(f"Feature vector size: {len(FEATURE_NAMES)} features")
        return df

    @staticmethod
    def _infer_dataset(row):
        """Best-effort dataset inference from query identifiers."""
        query_id = str(row.get('query_id', ''))
        query_type = str(row.get('query_type', '')).lower()

        if query_type in {'ap', 'tp'}:
            marker = f"_{query_type}_"
            if marker in query_id:
                return query_id.split(marker)[0]

        # Fallback: take everything before the last underscore+number suffix
        parts = query_id.split('_')
        if len(parts) > 2:
            return '_'.join(parts[:-2])
        if len(parts) > 1:
            return parts[0]
        return 'unknown'

    def prepare_features(self, df):
        """Augment dataframe with targets, labels, and weighting metadata."""
        df = df.copy()

        eps = 1e-3  # Small epsilon to avoid division by zero
        df['target'] = np.log((df['pg_time_ms'].values + eps) / (df['duck_time_ms'].values + eps))
        df['class_label'] = (df['target'] > 0).astype(int)
        df['relative_gap'] = np.abs(df['target'])
        df['regret'] = np.abs(df['pg_time_ms'] - df['duck_time_ms'])
        df['min_runtime'] = np.minimum(df['pg_time_ms'], df['duck_time_ms']) + eps
        df['dataset'] = df.apply(self._infer_dataset, axis=1)

        # Log summary statistics for visibility
        self.logger.info(
            "Prepared features: mean target=%.3f, std target=%.3f, positive ratio=%.2f",
            df['target'].mean(),
            df['target'].std(),
            df['class_label'].mean(),
        )

        # Derived features disabled - C code doesn't compute them
        # df = self._augment_features(df)
        return df

    def _augment_features(self, df):
        """Compute derived cross-features that help separate PG vs DuckDB."""
        required = {
            'has_order_by_limit', 'order_by_index_match', 'topk_indexed', 'limit_value',
            'covering_index_score', 'predicate_simple_eq', 'predicate_in',
            'total_columnar_bytes_est', 'total_rowstore_bytes_est',
            'has_aggregates', 'has_group_by', 'analytical_pattern', 'star_schema_score',
            'duckdb_pushdown_score', 'selectivity_low', 'predicate_range', 'predicate_like',
            'projected_column_count', 'output_row_width', 'parallel_safe', 'cardinality_large',
            'avg_scan_fraction', 'max_scan_fraction', 'parallel_unsafe_function_count',
            'volatile_function_count', 'total_projected_bytes', 'estimated_result_bytes',
            'max_projected_row_fraction', 'avg_projected_row_fraction', 'many_to_many_joins',
            'num_joins', 'index_usage_likely'
        }
        missing = required - set(df.columns)
        if missing:
            self.logger.warning("Derived feature prerequisites missing: %s", sorted(missing))
            for col in missing:
                df[col] = 0.0

        eps = 1e-6

        has_order_by_limit = df['has_order_by_limit'].to_numpy(dtype=float, copy=False)
        order_by_index_match = df['order_by_index_match'].to_numpy(dtype=float, copy=False)
        topk_indexed = df['topk_indexed'].to_numpy(dtype=float, copy=False)
        limit_value = np.maximum(df['limit_value'].to_numpy(dtype=float, copy=False), 0.0)

        idx_cover_gap = has_order_by_limit * (1.0 - order_by_index_match)
        df['idx_cover_gap'] = idx_cover_gap

        topk_idx_gap = has_order_by_limit * (1.0 - topk_indexed) * np.log1p(limit_value)
        df['topk_idx_gap'] = np.clip(topk_idx_gap, 0.0, 10.0)

        covering_index_score = df['covering_index_score'].to_numpy(dtype=float, copy=False)
        predicate_simple_eq = df['predicate_simple_eq'].to_numpy(dtype=float, copy=False)
        predicate_in = df['predicate_in'].to_numpy(dtype=float, copy=False)
        has_order = df['has_order_by_limit'].to_numpy(dtype=float, copy=False)
        df['covering_idx_effect'] = covering_index_score * (
            predicate_simple_eq + predicate_in + has_order
        )

        total_columnar = np.maximum(df['total_columnar_bytes_est'].to_numpy(dtype=float, copy=False), 0.0)
        total_rowstore = np.maximum(df['total_rowstore_bytes_est'].to_numpy(dtype=float, copy=False), 0.0)
        col_ratio = total_columnar / (total_columnar + total_rowstore + eps)
        df['col_ratio'] = col_ratio

        has_aggregates = df['has_aggregates'].to_numpy(dtype=float, copy=False)
        has_group_by = df['has_group_by'].to_numpy(dtype=float, copy=False)
        analytical_pattern = df['analytical_pattern'].to_numpy(dtype=float, copy=False)
        star_schema = df['star_schema_score'].to_numpy(dtype=float, copy=False)
        df['duckdb_suitability'] = col_ratio * (
            has_aggregates + has_group_by + analytical_pattern + star_schema
        )

        duckdb_pushdown = df['duckdb_pushdown_score'].to_numpy(dtype=float, copy=False)
        selectivity_low = df['selectivity_low'].to_numpy(dtype=float, copy=False)
        predicate_range = df['predicate_range'].to_numpy(dtype=float, copy=False)
        predicate_like = df['predicate_like'].to_numpy(dtype=float, copy=False)
        projected_cols = df['projected_column_count'].to_numpy(dtype=float, copy=False)
        output_row_width = np.maximum(df['output_row_width'].to_numpy(dtype=float, copy=False), 1.0)
        pushdown_base = selectivity_low + predicate_range + predicate_like + (projected_cols / output_row_width)
        df['pushdown_intensity'] = np.clip(duckdb_pushdown * pushdown_base, 0.0, 15.0)

        parallel_safe = df['parallel_safe'].to_numpy(dtype=float, copy=False)
        cardinality_large = df['cardinality_large'].to_numpy(dtype=float, copy=False)
        avg_scan_fraction = df['avg_scan_fraction'].to_numpy(dtype=float, copy=False)
        max_scan_fraction = df['max_scan_fraction'].to_numpy(dtype=float, copy=False)
        df['pg_parallel_edge'] = parallel_safe * (cardinality_large + avg_scan_fraction + max_scan_fraction)

        parallel_unsafe_fn = df['parallel_unsafe_function_count'].to_numpy(dtype=float, copy=False)
        volatile_fn = df['volatile_function_count'].to_numpy(dtype=float, copy=False)
        df['duck_parallel_edge'] = parallel_unsafe_fn + volatile_fn - parallel_safe

        total_projected_bytes = np.maximum(df['total_projected_bytes'].to_numpy(dtype=float, copy=False), 0.0)
        estimated_result_bytes = np.maximum(df['estimated_result_bytes'].to_numpy(dtype=float, copy=False), 0.0)
        df['scan_to_result'] = np.log1p(total_projected_bytes) - np.log1p(estimated_result_bytes + eps)

        max_proj_frac = df['max_projected_row_fraction'].to_numpy(dtype=float, copy=False)
        avg_proj_frac = df['avg_projected_row_fraction'].to_numpy(dtype=float, copy=False)
        df['project_frac_gap'] = max_proj_frac - avg_proj_frac

        many_to_many = df['many_to_many_joins'].to_numpy(dtype=float, copy=False)
        num_joins = df['num_joins'].to_numpy(dtype=float, copy=False)
        df['fanout'] = many_to_many * num_joins

        df['star_col_gain'] = star_schema * col_ratio * has_aggregates

        index_usage_likely = df['index_usage_likely'].to_numpy(dtype=float, copy=False)
        df['topk_no_idx'] = has_order * (1.0 - order_by_index_match) * np.log1p(limit_value) * (1.0 - index_usage_likely)

        return df

    def _compute_sample_weights(self, df_subset, predictions, dataset_counts, epoch, *, is_validation=False):
        """Compute self-paced weights with asymmetric focusing and hard-negative emphasis."""
        if df_subset.empty:
            return np.array([])

        total = len(df_subset)
        labels = df_subset['class_label'].to_numpy()

        # Fixed class weighting to discourage positive predictions
        class_factor = np.where(labels == 1, self.pos_class_weight_global, self.neg_class_weight_global)

        raw_gap = df_subset['relative_gap'].to_numpy()
        gap_factor = 1.0 + 1.0 / (1.0 + np.exp(-self.gap_sharpener_k * (raw_gap - self.gap_sharpener_m)))

        dataset_factor = df_subset['dataset'].map(
            lambda ds: 1.0 / np.sqrt(max(dataset_counts.get(ds, total), 1))
        ).to_numpy()

        regret_base = np.maximum(df_subset['regret'].to_numpy(), 1e-3)
        regret_factor = np.power(regret_base, 0.25)

        runtime_factor = np.power(df_subset['min_runtime'].to_numpy(), self.runtime_alpha)

        if predictions is None or len(predictions) == 0:
            raw_pred = np.zeros(total)
            prob = np.full(total, 0.5)
        else:
            raw_pred = np.asarray(predictions)
            prob = _clip_probabilities(sigmoid(raw_pred))

        schedule = self.weight_schedule[min(epoch, len(self.weight_schedule) - 1)]
        focus = np.ones_like(prob)

        pos_mask = labels == 1
        if np.any(pos_mask):
            focus[pos_mask] *= np.power(
                np.maximum(1.0 - prob[pos_mask], 1e-3),
                schedule['gamma_pos'],
            )

        neg_mask = ~pos_mask
        if np.any(neg_mask):
            over_conf = np.maximum(prob[neg_mask] - schedule['tau'], 0.0)
            neg_focus = np.where(
                over_conf > 0.0,
                1.0 + np.power(over_conf, schedule['gamma_neg']),
                1.0,
            )
            focus[neg_mask] *= neg_focus

        weights = (
            class_factor
            * gap_factor
            * dataset_factor
            * regret_factor
            * runtime_factor
            * focus
        )

        if not is_validation:
            hard_neg_threshold = schedule.get('hard_neg_threshold')
            hard_neg_boost = schedule.get('hard_neg_boost', 1.0)
            if hard_neg_threshold is not None and hard_neg_boost > 1.0:
                hard_neg_mask = (labels == 0) & (prob >= hard_neg_threshold)
                if np.any(hard_neg_mask):
                    weights[hard_neg_mask] *= hard_neg_boost

            # Mild explicit false-positive emphasis
            fp_boost = float(schedule.get('fp_boost', 1.0))
            if fp_boost > 1.0:
                fp_mask = (labels == 0) & (raw_pred > 0)
                if np.any(fp_mask):
                    weights[fp_mask] *= fp_boost

        # Asymmetric small-gap handling: strongly downweight weak positives
        # and mildly downweight weak negatives.
        small_gap = np.log(1.08)
        pos_small = (labels == 1) & (raw_gap < small_gap)
        neg_small = (labels == 0) & (raw_gap < small_gap)
        if np.any(pos_small):
            weights[pos_small] *= 0.1
        if np.any(neg_small):
            weights[neg_small] *= 0.5

        clip_low_base, clip_high_base = schedule['clip']
        clip_low = clip_low_base / np.sqrt(epoch + 1)
        clip_high = clip_high_base
        weights = np.clip(weights, clip_low, clip_high)

        weights /= np.mean(weights)
        return weights


    def evaluate_routing(self, model, X_val, pg_times_val, duck_times_val, threshold=0.0):
        """Evaluate routing decisions with classification metrics"""
        # Get raw predictions (log-odds)
        raw_predictions = model.predict(X_val, num_iteration=model.best_iteration, raw_score=True)
        probabilities = sigmoid(raw_predictions)

        # Fixed margin decision rule: route to DuckDB if raw_score > 0
        predicted_duckdb = raw_predictions > 0

        # Ground truth: which engine is actually faster
        actual_duckdb = duck_times_val < pg_times_val

        # Calculate classification metrics
        accuracy = accuracy_score(actual_duckdb, predicted_duckdb)
        precision = precision_score(actual_duckdb, predicted_duckdb, zero_division=0)
        recall = recall_score(actual_duckdb, predicted_duckdb, zero_division=0)
        conf_matrix = confusion_matrix(actual_duckdb, predicted_duckdb)

        # Calculate latencies for different routing strategies
        # 1. LightGBM routing
        lgbm_latencies = np.where(predicted_duckdb, duck_times_val, pg_times_val)
        lgbm_avg_latency = np.mean(lgbm_latencies)
        lgbm_total_latency = np.sum(lgbm_latencies)

        # 2. Always PostgreSQL
        pg_avg_latency = np.mean(pg_times_val)
        pg_total_latency = np.sum(pg_times_val)

        # 3. Always DuckDB
        duck_avg_latency = np.mean(duck_times_val)
        duck_total_latency = np.sum(duck_times_val)

        # 4. Oracle (perfect routing)
        oracle_latencies = np.minimum(pg_times_val, duck_times_val)
        oracle_avg_latency = np.mean(oracle_latencies)
        oracle_total_latency = np.sum(oracle_latencies)

        # Print results
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ROUTING EVALUATION ({len(FEATURE_NAMES)} features, threshold = 0.000000)")
        self.logger.info(f"{'='*60}")

        self.logger.info(f"\nBinary Classification Metrics:")
        self.logger.info(f"  Accuracy:  {accuracy:.3f}")
        self.logger.info(f"  Precision: {precision:.3f} (correctly predicted DuckDB / all predicted DuckDB)")
        self.logger.info(f"  Recall:    {recall:.3f} (correctly predicted DuckDB / all actual DuckDB better)")

        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"                 Predicted PG  Predicted DuckDB")

        # Handle cases where confusion matrix might be smaller (all predictions same class)
        if conf_matrix.shape == (2, 2):
            self.logger.info(f"  Actual PG:     {conf_matrix[0,0]:8d}     {conf_matrix[0,1]:8d}")
            self.logger.info(f"  Actual DuckDB: {conf_matrix[1,0]:8d}     {conf_matrix[1,1]:8d}")
        elif conf_matrix.shape == (2, 1):
            # All predictions are the same class
            if predicted_duckdb.sum() == 0:  # All predicted PostgreSQL
                self.logger.info(f"  Actual PG:     {conf_matrix[0,0]:8d}            0")
                self.logger.info(f"  Actual DuckDB: {conf_matrix[1,0]:8d}            0")
            else:  # All predicted DuckDB
                self.logger.info(f"  Actual PG:            0     {conf_matrix[0,0]:8d}")
                self.logger.info(f"  Actual DuckDB:        0     {conf_matrix[1,0]:8d}")
        elif conf_matrix.shape == (1, 2):
            # All actual labels are the same class
            self.logger.info(f"  Actual PG:     {conf_matrix[0,0]:8d}     {conf_matrix[0,1]:8d}")
            self.logger.info(f"  Actual DuckDB:        0            0")
        else:
            # Single class case
            self.logger.info(f"  Single class case - confusion matrix shape: {conf_matrix.shape}")
            self.logger.info(f"  Matrix: {conf_matrix}")

        self.logger.info(f"\nAverage Query Latency (ms):")
        self.logger.info(f"  Always PostgreSQL: {pg_avg_latency:10.2f}")
        self.logger.info(f"  Always DuckDB:     {duck_avg_latency:10.2f}")
        self.logger.info(f"  LightGBM Routing:  {lgbm_avg_latency:10.2f}")
        self.logger.info(f"  Oracle (Perfect):  {oracle_avg_latency:10.2f}")

        self.logger.info(f"\nTotal Execution Time (ms):")
        self.logger.info(f"  Always PostgreSQL: {pg_total_latency:12.0f}")
        self.logger.info(f"  Always DuckDB:     {duck_total_latency:12.0f}")
        self.logger.info(f"  LightGBM Routing:  {lgbm_total_latency:12.0f}")
        self.logger.info(f"  Oracle (Perfect):  {oracle_total_latency:12.0f}")

        # Performance improvements
        best_single = min(pg_avg_latency, duck_avg_latency)
        lgbm_vs_best = (best_single - lgbm_avg_latency) / best_single * 100
        lgbm_vs_oracle = (lgbm_avg_latency - oracle_avg_latency) / oracle_avg_latency * 100

        self.logger.info(f"\nPerformance Analysis:")
        self.logger.info(f"  LightGBM vs Best Single Engine: {lgbm_vs_best:+.1f}%")
        self.logger.info(f"  LightGBM vs Oracle (gap):       {lgbm_vs_oracle:+.1f}%")
        self.logger.info(f"  Routing Decisions: {np.sum(predicted_duckdb)}/{len(predicted_duckdb)} chose DuckDB")

        return accuracy, precision, recall, conf_matrix

    def train(self, prepared_df):
        """Train LightGBM classifier with self-paced, Taylor-weighted boosting."""
        dataset_counts = prepared_df['dataset'].value_counts()

        # Stratified split to maintain class balance
        indices = prepared_df.index.to_numpy()
        stratify_series = prepared_df['class_label']
        stratify_arg = stratify_series.to_numpy() if stratify_series.nunique() > 1 else None
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=stratify_arg,
            random_state=42,
        )

        train_df = prepared_df.loc[train_idx].reset_index(drop=True)
        val_df = prepared_df.loc[val_idx].reset_index(drop=True)

        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df['class_label'].values.astype(float)
        X_val = val_df[FEATURE_NAMES].values
        y_val = val_df['class_label'].values.astype(float)

        train_preds = np.zeros(len(train_df))
        val_preds = np.zeros(len(val_df))
        booster = None

        self.logger.info(
            "Training with self-paced Taylor-weighted boosting: %d epochs × %d trees (early_stop=150)",
            self.num_epochs,
            self.trees_per_epoch,
        )

        for epoch in range(self.num_epochs):
            lr = self.base_learning_rate * (0.8 ** epoch)
            params = {
                'objective': 'binary',
                'metric': ['binary_logloss'],
                'boosting_type': 'gbdt',
                'num_leaves': 64,
                'max_depth': 8,
                'learning_rate': lr,
                'feature_fraction': 0.60,
                'bagging_fraction': 0.60,
                'bagging_freq': 1,
                'min_child_samples': 800,
                'lambda_l2': 10.0,
                'min_gain_to_split': 0.20,
                'path_smooth': 20.0,
                'monotone_constraints': self.monotone_constraints,
                'verbosity': -1,
                'num_threads': -1,
                'seed': 42,
            }

            use_custom_obj = self.use_focal_loss and self._supports_custom_objective
            if use_custom_obj:
                params['objective'] = 'none'

            # Train on the full dataset each epoch (avoid oversampling positives in epoch 0)
            X_epoch, y_epoch, df_epoch = X_train, y_train, train_df
            pred_epoch = train_preds

            train_weights = self._compute_sample_weights(
                df_epoch,
                pred_epoch,
                dataset_counts,
                epoch,
                is_validation=False,
            )
            val_weights = self._compute_sample_weights(
                val_df,
                val_preds,
                dataset_counts,
                epoch,
                is_validation=True,
            )

            train_data = lgb.Dataset(
                X_epoch,
                label=y_epoch,
                weight=train_weights,
                feature_name=FEATURE_NAMES,
                free_raw_data=False,
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                weight=val_weights,
                feature_name=FEATURE_NAMES,
                reference=train_data,
                free_raw_data=False,
            )

            train_kwargs = {
                'params': params,
                'train_set': train_data,
                'num_boost_round': self.trees_per_epoch,
                'init_model': booster,
                'valid_sets': [val_data],
                'valid_names': ['val'],
                'keep_training_booster': True,
                'callbacks': [
                    lgb.early_stopping(150, first_metric_only=True),
                    lgb.log_evaluation(100),
                ],
            }

            if use_custom_obj:
                train_kwargs['fobj'] = self._focal_loss_objective

            try:
                booster = lgb.train(**train_kwargs)
            except TypeError as exc:
                if use_custom_obj:
                    self.logger.warning(
                        "LightGBM version does not support custom objective (fobj). Falling back to binary objective. Error: %s",
                        exc,
                    )
                    params['objective'] = 'binary'
                    train_kwargs.pop('fobj', None)
                    train_kwargs['params'] = params
                    booster = lgb.train(**train_kwargs)
                    self.use_focal_loss = False
                else:
                    raise

            train_preds = booster.predict(X_train, raw_score=True)
            val_preds = booster.predict(X_val, raw_score=True)

        self.model = booster

        # No margin calibration; keep fixed margin decision rule at 0
        # Recompute final predictions for both train and validation
        train_preds = self.model.predict(X_train, raw_score=True)
        val_preds = self.model.predict(X_val, raw_score=True)
        prob_val = _clip_probabilities(sigmoid(val_preds))
        binary_val = val_preds > 0

        try:
            val_logloss = log_loss(y_val, prob_val, labels=[0, 1])
        except ValueError:
            val_logloss = float('nan')

        val_bal_acc = balanced_accuracy_score(y_val, binary_val)
        val_precision = precision_score(y_val, binary_val, zero_division=0)
        val_recall = recall_score(y_val, binary_val, zero_division=0)

        # Classification metrics (margin > 0) for train and validation
        binary_train = train_preds > 0
        train_acc = accuracy_score(y_train, binary_train)
        train_prec = precision_score(y_train, binary_train, zero_division=0)
        train_rec = recall_score(y_train, binary_train, zero_division=0)

        val_acc = accuracy_score(y_val, binary_val)

        self.logger.info("Validation metrics after self-paced training:")
        self.logger.info(f"  LogLoss: {val_logloss:.4f}")
        self.logger.info(f"  Balanced Accuracy: {val_bal_acc:.4f}")
        self.logger.info(f"  Precision (DuckDB): {val_precision:.4f}")
        self.logger.info(f"  Recall (DuckDB): {val_recall:.4f}")

        self.logger.info("\nClassification metrics (margin > 0):")
        self.logger.info(f"  Train - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        self.logger.info(f"  Valid - Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # Final evaluation with fixed threshold 0
        self.logger.info("Using raw score threshold: 0.0 (fixed margin)")
        self.evaluate_routing(
            self.model,
            X_val,
            val_df['pg_time_ms'].to_numpy(),
            val_df['duck_time_ms'].to_numpy(),
            threshold=0.0,
        )

        # Optional: Adaptive (floating) margin via per-bucket thresholds
        if self.adaptive_margin:
            self.logger.info("\n===== ADAPTIVE (FLOATING) MARGIN EVALUATION =====")
            raw_val = val_preds
            pg_ms_val = val_df['pg_time_ms'].to_numpy()
            duck_ms_val = val_df['duck_time_ms'].to_numpy()
            bucket_ids = self._compute_bucket_ids(val_df)
            thresholds = self._tune_thresholds_by_bucket(
                raw_val,
                bucket_ids,
                pg_ms_val,
                duck_ms_val,
                labels=y_val,
                objective=getattr(self, 'adaptive_objective', 'latency'),
                fbeta=getattr(self, 'adaptive_fbeta', 1.0),
                q=101,
                min_bucket=50,
            )
            pred_duck = self._apply_bucket_thresholds(raw_val, bucket_ids, thresholds)

            # Classification metrics under adaptive thresholds
            acc_ad = accuracy_score(y_val, pred_duck)
            prec_ad = precision_score(y_val, pred_duck, zero_division=0)
            rec_ad = recall_score(y_val, pred_duck, zero_division=0)
            cm_ad = confusion_matrix(y_val, pred_duck)

            # Latencies
            lats = np.where(pred_duck, duck_ms_val, pg_ms_val)
            avg_lat = float(np.mean(lats))
            total_lat = float(np.sum(lats))
            avg_pg = float(np.mean(pg_ms_val))
            avg_duck = float(np.mean(duck_ms_val))
            avg_oracle = float(np.mean(np.minimum(pg_ms_val, duck_ms_val)))

            self.logger.info("Adaptive Classification Metrics:")
            self.logger.info(f"  Accuracy:  {acc_ad:.3f}")
            self.logger.info(f"  Precision: {prec_ad:.3f}")
            self.logger.info(f"  Recall:    {rec_ad:.3f}")
            self.logger.info("Confusion Matrix:")
            if cm_ad.shape == (2, 2):
                self.logger.info(f"                 Predicted PG  Predicted DuckDB")
                self.logger.info(f"  Actual PG:     {cm_ad[0,0]:8d}     {cm_ad[0,1]:8d}")
                self.logger.info(f"  Actual DuckDB: {cm_ad[1,0]:8d}     {cm_ad[1,1]:8d}")
            else:
                self.logger.info(f"  Matrix: {cm_ad}")

            self.logger.info("\nAverage Query Latency (ms):")
            self.logger.info(f"  Always PostgreSQL: {avg_pg:10.2f}")
            self.logger.info(f"  Always DuckDB:     {avg_duck:10.2f}")
            self.logger.info(f"  Adaptive Routing:  {avg_lat:10.2f}")
            self.logger.info(f"  Oracle (Perfect):  {avg_oracle:10.2f}")

            best_single = min(avg_pg, avg_duck)
            lgbm_vs_best = (best_single - avg_lat) / best_single * 100.0
            lgbm_vs_oracle = (avg_lat - avg_oracle) / avg_oracle * 100.0
            self.logger.info("\nPerformance Analysis (Adaptive):")
            self.logger.info(f"  Adaptive vs Best Single Engine: {lgbm_vs_best:+.1f}%")
            self.logger.info(f"  Adaptive vs Oracle (gap):       {lgbm_vs_oracle:+.1f}%")
            self.logger.info(f"  Routing Decisions: {int(np.sum(pred_duck))}/{len(pred_duck)} chose DuckDB")

            # Persist thresholds for later inspection/use
            self.adaptive_thresholds = thresholds

        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        self.logger.info(f"\nTop 15 most important features (out of {len(FEATURE_NAMES)}):")
        for _, row in feature_imp.head(15).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.1f}")

    def save_model(self, name_suffix=""):
        """Save model and configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lightgbm_model"

        # Save model in native LightGBM format (for C API)
        model_path = os.path.join(self.model_dir, f"{model_name}.txt")

        # Save the model directly in standard LightGBM format
        self.model.save_model(model_path, num_iteration=self.model.best_iteration)

        self.logger.info(f"Model saved to: {model_path}")

        # Save threshold separately for easy access (fixed raw-score threshold)
        threshold_path = os.path.join(self.model_dir, f"{model_name}_threshold.txt")
        with open(threshold_path, 'w') as f:
            f.write(f"{self.threshold:.6f}\n")

        self.logger.info(f"Threshold saved to: {threshold_path}")

        # Save configuration and metrics
        config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        config = {
            'version': FEATURE_SCHEMA_VERSION,
            'features': FEATURE_NAMES,
            'num_features': len(FEATURE_NAMES),
            'threshold': 0.0,
            'model_type': 'binary_classification',
            'target': 'class_label (DuckDB better if log(pg_time/duck_time) > 0)',
            'decision_rule': 'route to DuckDB if raw_score > 0',
            'num_trees': self.model.num_trees(),
            'training_date': timestamp,
            'expansion_phase': 'Phase 3 - derived cross features',
            'monotone_constraints': self.monotone_constraints,
            'derived_features': DERIVED_FEATURE_NAMES,
            'adaptive_margin': bool(self.adaptive_margin),
            'adaptive_features': list(self.adaptive_features),
            'use_focal_loss': bool(self.use_focal_loss),
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to: {config_path}")

        # Save adaptive thresholds if available
        if self.adaptive_margin and isinstance(self.adaptive_thresholds, dict):
            thresholds_path = os.path.join(self.model_dir, f"{model_name}_adaptive_thresholds.json")
            with open(thresholds_path, 'w') as f:
                json.dump({str(k): float(v) for k, v in self.adaptive_thresholds.items()}, f, indent=2)
            self.logger.info(f"Adaptive thresholds saved to: {thresholds_path}")

        return model_path

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model with expanded features')
    parser.add_argument('--data-dir', default='lightgbm_training_data_new', help='Training data directory')
    parser.add_argument('--model-dir', default='lightgbm_models', help='Model output directory')
    parser.add_argument('--data-file', default='training_data.csv', help='Training data filename')
    parser.add_argument('--suffix', default='', help='Model name suffix')
    parser.add_argument('--adaptive-margin', action='store_true', help='Enable adaptive (floating) margin using per-bucket thresholds learned on validation')
    parser.add_argument('--adaptive-objective', choices=['latency','precision','f1','fbeta'], default='latency', help='Objective for tuning per-bucket thresholds')
    parser.add_argument('--adaptive-fbeta', type=float, default=1.0, help='Beta for F-beta when adaptive-objective=fbeta')
    parser.add_argument('--no-focal-loss', action='store_true', help='Disable focal loss objective (use standard binary logloss)')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Alpha parameter for focal loss (balance factor)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for focal loss (focus factor)')
    args = parser.parse_args()

    # Initialize trainer
    trainer = LightGBMTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        adaptive_margin=args.adaptive_margin,
        use_focal_loss=not args.no_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    trainer.adaptive_objective = args.adaptive_objective
    trainer.adaptive_fbeta = args.adaptive_fbeta

    # Load and prepare data
    df_raw = trainer.load_data(args.data_file)
    prepared_df = trainer.prepare_features(df_raw)

    # Train model
    trainer.train(prepared_df)

    # Save model
    model_path = trainer.save_model(args.suffix)

    print(f"\nTraining complete! Model saved to: {model_path}")
    print(f"Feature vector size: {len(FEATURE_NAMES)} features (v2.2.0 schema)")
    print(f"To use this model, set in PostgreSQL:")
    print(f"  SET lightgbm.model_path = '{os.path.abspath(model_path)}';")
    print(f"  SET lightgbm.routing_threshold = {trainer.threshold:.6f};")
    print(f"  SET lightgbm.enabled = true;")
    print(f"Decision rule: Route to DuckDB if raw_score > 0 (no calibration)")

if __name__ == "__main__":
    main()
