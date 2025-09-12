#!/usr/bin/env python3
"""
Train LightGBM model for PostgreSQL/DuckDB query routing.
This script reads CSV feature data and trains a binary classifier
to predict which engine should execute each query.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import os
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import json

# Configure logging
logger = logging.getLogger(__name__)

class LightGBMTrainer:
    def __init__(self, data_dir: str, model_dir: str = "lightgbm_models"):
        """
        Initialize LightGBM trainer.
        
        Args:
            data_dir: Directory containing CSV feature files
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.feature_columns = [
            'num_tables',
            'num_joins', 
            'query_depth',
            'complexity_score',
            'has_aggregates',
            'has_group_by',
            'has_order_by',
            'has_limit',
            'has_distinct',
            'has_window_functions',
            'has_outer_joins',
            'estimated_join_complexity',
            'has_subqueries',
            'has_correlated_subqueries',
            'has_large_tables',
            'all_tables_small',
            'has_complex_expressions',
            'has_user_functions',
            'has_text_operations',
            'has_numeric_heavy_ops',
            'num_aggregate_funcs',
            'analytical_pattern',
            'transactional_pattern',
            'etl_pattern',
            'command_type'
        ]
        
        self.model = None
        self.scaler = None
        
    def load_training_data(self, datasets: list = None) -> pd.DataFrame:
        """Load and combine CSV training data from multiple datasets."""
        all_data = []
        
        # Find all CSV files if datasets not specified
        if datasets is None:
            datasets = []
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_features.csv'):
                    dataset_name = filename.replace('_features.csv', '')
                    datasets.append(dataset_name)
        
        logger.info(f"Loading data from datasets: {datasets}")
        
        for dataset in datasets:
            csv_file = os.path.join(self.data_dir, f"{dataset}_features.csv")
            
            if not os.path.exists(csv_file):
                logger.warning(f"CSV file not found: {csv_file}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded {len(df)} samples from {dataset}")
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if not all_data:
            raise ValueError("No training data found!")
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total combined samples: {len(combined_df)}")
        
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess data for training."""
        logger.info("Preprocessing training data...")
        
        # Remove rows with invalid execution times
        df = df[(df['postgres_time_ms'] > 0) & (df['duckdb_time_ms'] > 0)]
        logger.info(f"Samples after filtering invalid times: {len(df)}")
        
        # Ensure all feature columns exist
        missing_features = []
        for col in self.feature_columns:
            if col not in df.columns:
                missing_features.append(col)
                df[col] = 0  # Default value
        
        if missing_features:
            logger.warning(f"Missing feature columns (using defaults): {missing_features}")
        
        # Extract features
        X = df[self.feature_columns].copy()
        
        # Convert boolean columns to numeric
        bool_columns = ['has_aggregates', 'has_group_by', 'has_order_by', 'has_limit', 
                       'has_distinct', 'has_window_functions', 'has_outer_joins',
                       'has_subqueries', 'has_correlated_subqueries', 'has_large_tables',
                       'all_tables_small', 'has_complex_expressions', 'has_user_functions',
                       'has_text_operations', 'has_numeric_heavy_ops', 'analytical_pattern',
                       'transactional_pattern', 'etl_pattern']
        
        for col in bool_columns:
            if col in X.columns:
                X[col] = X[col].astype(int)
        
        # Create binary target: 1 = DuckDB preferred, 0 = PostgreSQL preferred
        y = (df['optimal_engine'] == 'duckdb').astype(int)
        
        # Log class distribution
        class_counts = y.value_counts()
        logger.info(f"Target distribution: PostgreSQL={class_counts.get(0, 0)}, DuckDB={class_counts.get(1, 0)}")
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """Train LightGBM model and return metrics."""
        logger.info("Training LightGBM model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Create datasets
        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
        
        # Train model with early stopping
        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[valid_dataset],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'num_features': len(self.feature_columns)
        }
        
        logger.info("Training completed!")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save_model(self, model_name: str = "lightgbm_routing_model"):
        """Save the trained model and feature information."""
        if self.model is None:
            raise ValueError("No model to save! Train a model first.")
        
        # Save LightGBM model
        model_path = os.path.join(self.model_dir, f"{model_name}.txt")
        self.model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save feature names and model info
        model_info = {
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),
            'model_type': 'LightGBM',
            'objective': 'binary',
            'target_description': 'Engine selection: 0=PostgreSQL, 1=DuckDB'
        }
        
        info_path = os.path.join(self.model_dir, f"{model_name}_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create simplified model file for C integration
        self._save_simplified_model(model_name)
        
        logger.info(f"Model info saved to: {info_path}")
    
    def _save_simplified_model(self, model_name: str):
        """Save a simplified model file for C integration."""
        # For the C integration, we'll save a simplified linear approximation
        # In practice, you'd want to use the full LightGBM C API
        
        # Get feature importances as weights (simplified)
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        })
        
        # Normalize importances to use as weights
        importance_df['weight'] = importance_df['importance'] / importance_df['importance'].sum()
        
        # Save in simple format for C code
        c_model_path = os.path.join(self.model_dir, f"{model_name}_simplified.txt")
        with open(c_model_path, 'w') as f:
            # Write header: num_features, num_trees (simplified to 1), weights, bias
            f.write(f"{len(self.feature_columns)}\n")
            f.write("1\n")  # Simplified to single "tree"
            
            # Write feature weights
            for weight in importance_df['weight']:
                f.write(f"{weight:.6f}\n")
            
            # Write bias (simplified)
            f.write("0.0\n")
        
        logger.info(f"Simplified C model saved to: {c_model_path}")
    
    def evaluate_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
        """Perform cross-validation evaluation."""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Create LightGBM classifier for sklearn compatibility
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(lgb_clf, X, y, cv=cv_folds, scoring='accuracy')
        
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def analyze_feature_importance(self):
        """Analyze and log feature importance."""
        if self.model is None:
            raise ValueError("No model available! Train a model first.")
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        logger.info(importance_df.head(10).to_string(index=False))
        
        # Save feature importance
        importance_path = os.path.join(self.model_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to: {importance_path}")
        
        return importance_df


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model for query routing')
    parser.add_argument('--data-dir', default='lightgbm_training_data',
                       help='Directory containing CSV feature files')
    parser.add_argument('--model-dir', default='lightgbm_models',
                       help='Directory to save trained models')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to use (default: all found)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--model-name', default='lightgbm_routing_model',
                       help='Name for saved model files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize trainer
        trainer = LightGBMTrainer(args.data_dir, args.model_dir)
        
        # Load training data
        df = trainer.load_training_data(args.datasets)
        
        # Preprocess data
        X, y = trainer.preprocess_data(df)
        
        # Train model
        metrics = trainer.train_model(X, y, args.test_size)
        
        # Cross-validation
        cv_scores = trainer.evaluate_cross_validation(X, y, args.cv_folds)
        
        # Feature importance analysis
        importance_df = trainer.analyze_feature_importance()
        
        # Save model
        trainer.save_model(args.model_name)
        
        # Save training summary
        summary = {
            'training_metrics': metrics,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'top_features': importance_df.head(10).to_dict('records')
        }
        
        summary_path = os.path.join(args.model_dir, f"{args.model_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nTraining completed successfully!")
        logger.info(f"Model files saved in: {args.model_dir}")
        logger.info(f"Training summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())