#!/usr/bin/env python3
"""
LightGBM Training Script v2 - Regression with Threshold Calibration
Trains a regression model on log(pg_time/duck_time) and calibrates threshold for minimum makespan
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

# Feature names - must match kernel exactly
FEATURE_NAMES = [
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type"
]

FEATURE_SCHEMA_VERSION = "v1.0.0"

class LightGBMTrainer:
    def __init__(self, data_dir="lightgbm_training_data", model_dir="lightgbm_models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.threshold = 0.0  # Log scale threshold
        
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
        # Check if specific file exists
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(df)} samples from {filepath}")
        else:
            # Try to load all matching CSV files
            pattern = os.path.join(self.data_dir, "training_data*.csv")
            files = glob.glob(pattern)
            
            if not files:
                self.logger.error(f"No training data found in {self.data_dir}")
                self.logger.info(f"Looking for: {filename} or training_data*.csv")
                sys.exit(1)
            
            # Load and concatenate all CSV files
            dfs = []
            for file in files:
                df_part = pd.read_csv(file)
                dfs.append(df_part)
                self.logger.info(f"Loaded {len(df_part)} samples from {os.path.basename(file)}")
            
            df = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"Total: {len(df)} samples from {len(files)} files")
        
        # Filter out queries where either engine failed or took too long
        max_time_ms = 60000  # 60 seconds
        df = df[(df['pg_time_ms'] > 0) & (df['pg_time_ms'] < max_time_ms) & 
                (df['duck_time_ms'] > 0) & (df['duck_time_ms'] < max_time_ms)]
        
        self.logger.info(f"After filtering: {len(df)} valid samples")
        return df
    
    def prepare_features(self, df):
        """Prepare features and regression target"""
        # Extract features
        X = df[FEATURE_NAMES].values
        
        # Create regression target: log(pg_time / duck_time)
        # Positive values mean DuckDB is faster
        # Negative values mean PostgreSQL is faster
        eps = 0.001  # Small epsilon to avoid log(0)
        y_regression = np.log((df['pg_time_ms'].values + eps) / (df['duck_time_ms'].values + eps))
        
        # Also store raw times for threshold calibration
        pg_times = df['pg_time_ms'].values
        duck_times = df['duck_time_ms'].values
        
        # Add sample weights based on query cost (expensive queries matter more)
        weights = np.maximum(pg_times, duck_times)
        weights = weights / weights.mean()  # Normalize
        
        self.logger.info(f"Feature shape: {X.shape}")
        self.logger.info(f"Target stats: mean={y_regression.mean():.3f}, std={y_regression.std():.3f}")
        self.logger.info(f"Positive targets (DuckDB faster): {(y_regression > 0).sum()}/{len(y_regression)}")
        
        return X, y_regression, pg_times, duck_times, weights
    
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
        self.logger.info(f"ROUTING EVALUATION (threshold = {threshold})")
        self.logger.info(f"{'='*60}")
        
        self.logger.info(f"\nBinary Classification Metrics:")
        self.logger.info(f"  Accuracy:  {accuracy:.3f}")
        self.logger.info(f"  Precision: {precision:.3f} (correctly predicted DuckDB / all predicted DuckDB)")
        self.logger.info(f"  Recall:    {recall:.3f} (correctly predicted DuckDB / all actual DuckDB better)")
        
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"                 Predicted PG  Predicted DuckDB")
        self.logger.info(f"  Actual PG:     {conf_matrix[0,0]:8d}     {conf_matrix[0,1]:8d}")
        self.logger.info(f"  Actual DuckDB: {conf_matrix[1,0]:8d}     {conf_matrix[1,1]:8d}")
        
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
    
    def train(self, X, y, pg_times, duck_times, weights):
        """Train LightGBM regression model with cross-validation"""
        # Split data
        X_train, X_val, y_train, y_val, pg_train, pg_val, duck_train, duck_val, w_train, w_val = \
            train_test_split(X, y, pg_times, duck_times, weights, test_size=0.2, random_state=42)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=FEATURE_NAMES)
        val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, feature_name=FEATURE_NAMES, reference=train_data)
        
        # Parameters for regression
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbosity': 1,
            'num_threads': -1,
            'seed': 42
        }
        
        # Train with early stopping
        self.logger.info("Training LightGBM regression model...")
        evals_result = {}
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(20),
                lgb.log_evaluation(20),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        self.logger.info(f"Validation metrics:")
        self.logger.info(f"  RMSE: {val_rmse:.4f}")
        self.logger.info(f"  MAE: {val_mae:.4f}")
        self.logger.info(f"  R²: {val_r2:.4f}")
        
        # Use fixed threshold = 0 (no calibration)
        self.threshold = 0.0
        
        # Evaluate routing with classification metrics
        self.evaluate_routing(self.model, X_val, pg_val, duck_val, threshold=self.threshold)
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.logger.info("\nTop 10 most important features:")
        for idx, row in feature_imp.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.1f}")
    
    def save_model(self, name_suffix=""):
        """Save model and configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lightgbm_model_{timestamp}{name_suffix}"
        
        # Save model in native LightGBM format (for C API)
        model_path = os.path.join(self.model_dir, f"{model_name}.txt")
        
        # Write header with feature names and version
        with open(model_path, 'w') as f:
            f.write(f"# lgbm_feature_names: {','.join(FEATURE_NAMES)}\n")
            f.write(f"# lgbm_version: {FEATURE_SCHEMA_VERSION}\n")
            f.write(f"# lgbm_threshold: 0.000000\n")  # Fixed threshold
            f.write(f"# lgbm_model_type: regression\n")
            f.write(f"# lgbm_target: log(pg_time/duck_time)\n")
            
            # Save the actual model
            model_str = self.model.model_to_string()
            f.write(model_str)
        
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
            'training_date': timestamp
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_path}")
        
        return model_path

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model for query routing')
    parser.add_argument('--data-dir', default='lightgbm_training_data', help='Training data directory')
    parser.add_argument('--model-dir', default='lightgbm_models', help='Model output directory')
    parser.add_argument('--data-file', default='training_data.csv', help='Training data filename')
    parser.add_argument('--suffix', default='', help='Model name suffix')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LightGBMTrainer(data_dir=args.data_dir, model_dir=args.model_dir)
    
    # Load and prepare data
    df = trainer.load_data(args.data_file)
    X, y, pg_times, duck_times, weights = trainer.prepare_features(df)
    
    # Train model
    trainer.train(X, y, pg_times, duck_times, weights)
    
    # Save model
    model_path = trainer.save_model(args.suffix)
    
    print(f"\nTraining complete! Model saved to: {model_path}")
    print(f"To use this model, set in PostgreSQL:")
    print(f"  SET lightgbm.model_path = '{os.path.abspath(model_path)}';")
    print(f"  SET lightgbm.routing_threshold = {trainer.threshold:.6f};")
    print(f"  SET lightgbm.enabled = true;")

if __name__ == "__main__":
    main()