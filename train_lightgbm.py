#!/usr/bin/env python3
"""
LightGBM Training Script - Expanded 50 Features
Trains a regression model on log(pg_time/duck_time) with expanded feature set
Uses 50 features (25 original + 25 new) for improved routing accuracy
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import logging
from datetime import datetime
import argparse

# Expanded feature names - must match kernel exactly (v2.0.0 schema)
FEATURE_NAMES = [
    # Original 25 features
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type",

    # Phase 1 expansion - 25 new features
    "join_type_inner", "join_type_left", "join_type_right", "join_type_full", "join_type_cross",
    "predicate_simple_eq", "predicate_range", "predicate_like", "predicate_in", "predicate_exists",
    "has_parameters", "num_cte", "max_subquery_depth", "has_recursive_cte", "has_lateral_join",
    "selectivity_high", "selectivity_medium", "selectivity_low", "cardinality_large", "cardinality_medium",
    "index_usage_likely", "partition_pruning_likely", "parallel_safe", "has_volatile_funcs", "cost_estimate_high"
]

FEATURE_SCHEMA_VERSION = "v2.0.0"

class LightGBMTrainer:
    def __init__(self, data_dir="lightgbm_training_data", model_dir="lightgbm_models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.threshold = 0.0  # Fixed threshold
        self.num_epochs = 3
        self.trees_per_epoch = 400
        self.base_learning_rate = 0.045
        self.runtime_alpha = 0.5

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)

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
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
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

        return df

    def _compute_sample_weights(self, df_subset, predictions, dataset_counts, epoch):
        """Compute self-paced Taylor-weighted boosting weights."""
        if df_subset.empty:
            return np.array([])

        total = len(df_subset)
        class_counts = df_subset['class_label'].value_counts()
        class_factor = df_subset['class_label'].map(
            lambda lbl: total / (2.0 * max(class_counts.get(lbl, 1), 1))
        ).to_numpy()

        gap_factor = df_subset['relative_gap'].to_numpy() + 1e-3
        dataset_factor = df_subset['dataset'].map(
            lambda ds: 1.0 / np.sqrt(max(dataset_counts.get(ds, total), 1))
        ).to_numpy()
        regret_factor = df_subset['regret'].to_numpy() + 1e-3
        runtime_factor = np.power(df_subset['min_runtime'].to_numpy(), self.runtime_alpha)

        if predictions is None or len(predictions) == 0:
            prob = np.full(total, 0.5)
        else:
            prob = 1.0 / (1.0 + np.exp(-predictions))
        focal_factor = np.square(1.0 - 2.0 * np.abs(prob - 0.5)) + 1e-3

        weights = (
            class_factor
            * gap_factor
            * dataset_factor
            * regret_factor
            * runtime_factor
            * focal_factor
        )

        # Gradient compression for near-ties (log gap < 5%)
        small_gap = np.log(1.05)
        near_ties = df_subset['relative_gap'].to_numpy() < small_gap
        weights[near_ties] *= 0.3

        # Soft clipping with gradually widening bounds
        clip_low = 0.05 / np.sqrt(epoch + 1)
        clip_high = 50.0 * (1.0 + 0.5 * epoch)
        weights = np.clip(weights, clip_low, clip_high)

        # Normalize to keep numerical scale stable
        weights /= np.mean(weights)
        return weights

    @staticmethod
    def _calibrate_threshold(predictions, pg_times, duck_times):
        """Find routing threshold that minimizes total latency on validation set."""
        if len(predictions) == 0:
            return 0.0

        candidate_thresholds = np.linspace(-0.5, 0.5, 101)
        best_threshold = 0.0
        best_latency = float('inf')

        for thr in candidate_thresholds:
            routed_duck = predictions > thr
            total_latency = np.sum(np.where(routed_duck, duck_times, pg_times))
            if total_latency < best_latency:
                best_latency = total_latency
                best_threshold = thr

        return best_threshold

    def evaluate_routing(self, model, X_val, pg_times_val, duck_times_val, threshold=0):
        """Evaluate routing decisions with classification metrics"""
        # Get predictions
        predictions = model.predict(X_val, num_iteration=model.best_iteration)

        # Route based on threshold (positive = DuckDB, negative = PostgreSQL)
        predicted_duckdb = predictions > threshold

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
        self.logger.info(f"ROUTING EVALUATION (50 features, threshold = {threshold})")
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
        """Train LightGBM regression model with self-paced boosting."""
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
        y_train = train_df['target'].values
        X_val = val_df[FEATURE_NAMES].values
        y_val = val_df['target'].values

        train_preds = np.zeros(len(train_df))
        val_preds = np.zeros(len(val_df))
        booster = None

        self.logger.info(
            "Training with self-paced Taylor-weighted boosting: %d epochs × %d trees",
            self.num_epochs,
            self.trees_per_epoch,
        )

        for epoch in range(self.num_epochs):
            lr = self.base_learning_rate * (0.75 ** epoch)
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'goss',
                'num_leaves': 256,
                'max_depth': 18,
                'learning_rate': lr,
                'feature_fraction': 0.8,
                'bagging_fraction': 1.0,
                'bagging_freq': 0,
                'min_child_samples': 20,
                'lambda_l2': 1.0,
                'verbosity': -1,
                'num_threads': -1,
                'seed': 42,
            }

            train_weights = self._compute_sample_weights(train_df, train_preds, dataset_counts, epoch)
            val_weights = self._compute_sample_weights(val_df, val_preds, dataset_counts, epoch)

            train_data = lgb.Dataset(
                X_train,
                label=y_train,
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

            booster = lgb.train(
                params,
                train_data,
                num_boost_round=self.trees_per_epoch,
                init_model=booster,
                valid_sets=[val_data],
                valid_names=['val'],
                keep_training_booster=True,
                callbacks=[
                    lgb.early_stopping(100, first_metric_only=True),
                    lgb.log_evaluation(100),
                ],
            )

            train_preds = booster.predict(X_train)
            val_preds = booster.predict(X_val)

        self.model = booster

        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)

        self.logger.info("Validation metrics after self-paced training:")
        self.logger.info(f"  RMSE: {val_rmse:.4f}")
        self.logger.info(f"  MAE: {val_mae:.4f}")
        self.logger.info(f"  R²: {val_r2:.4f}")

        self.threshold = 0.0
        self.logger.info("Using fixed routing threshold: %.4f", self.threshold)

        self.evaluate_routing(
            self.model,
            X_val,
            val_df['pg_time_ms'].to_numpy(),
            val_df['duck_time_ms'].to_numpy(),
            threshold=self.threshold,
        )

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

        # Also save threshold separately for easy access
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
            'threshold': float(self.threshold),
            'model_type': 'regression',
            'target': 'log(pg_time/duck_time)',
            'num_trees': self.model.num_trees(),
            'training_date': timestamp,
            'expansion_phase': 'Phase 1 - 50 features'
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration saved to: {config_path}")

        return model_path

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model with expanded features')
    parser.add_argument('--data-dir', default='lightgbm_training_data', help='Training data directory')
    parser.add_argument('--model-dir', default='lightgbm_models', help='Model output directory')
    parser.add_argument('--data-file', default='training_data.csv', help='Training data filename')
    parser.add_argument('--suffix', default='', help='Model name suffix')
    args = parser.parse_args()

    # Initialize trainer
    trainer = LightGBMTrainer(data_dir=args.data_dir, model_dir=args.model_dir)

    # Load and prepare data
    df_raw = trainer.load_data(args.data_file)
    prepared_df = trainer.prepare_features(df_raw)

    # Train model
    trainer.train(prepared_df)

    # Save model
    model_path = trainer.save_model(args.suffix)

    print(f"\nTraining complete! Model saved to: {model_path}")
    print(f"Feature vector size: {len(FEATURE_NAMES)} features (v2.0.0 schema)")
    print(f"To use this model, set in PostgreSQL:")
    print(f"  SET lightgbm.model_path = '{os.path.abspath(model_path)}';")
    print(f"  SET lightgbm.routing_threshold = {trainer.threshold:.6f};")
    print(f"  SET lightgbm.enabled = true;")

if __name__ == "__main__":
    main()
