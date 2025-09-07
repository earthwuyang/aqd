#!/usr/bin/env python3
"""
Train Final LightGBM Model on ZSCE-Generated Training Data
Uses real dual-engine execution data for query routing decisions
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import joblib
import logging
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/final_lightgbm_training.log'),
        logging.StreamHandler()
    ]
)

class FinalLightGBMTrainer:
    """Train final LightGBM model on ZSCE dual-engine execution data"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.training_stats = {}
        
    def load_training_data(self) -> pd.DataFrame:
        """Load all available ZSCE training data"""
        logging.info("📂 Loading ZSCE training data...")
        
        # Find all training data files
        data_files = glob.glob('data/zsce_training_data_*.json')
        
        if not data_files:
            logging.error("No ZSCE training data files found!")
            return pd.DataFrame()
        
        logging.info(f"Found {len(data_files)} training data files")
        
        # Load and combine all data
        all_data = []
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                all_data.extend(data)
                logging.info(f"Loaded {len(data)} records from {file_path}")
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_data:
            logging.error("No training data loaded!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logging.info(f"📊 Total training records: {len(df):,}")
        
        return df
    
    def prepare_features_and_labels(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels for training"""
        logging.info("🔧 Preparing features and labels...")
        
        # Filter to successful dual executions only
        dual_success = df[(df['postgresql_success'] == True) & (df['duckdb_success'] == True)].copy()
        
        if len(dual_success) == 0:
            logging.error("No successful dual executions found!")
            return None, None, []
        
        logging.info(f"Using {len(dual_success):,} successful dual executions")
        
        # Feature columns (query characteristics)
        feature_columns = [
            'query_length', 'word_count', 'has_select', 'has_from', 'has_where',
            'has_join', 'has_group_by', 'has_order_by', 'has_having', 'has_limit',
            'has_count', 'has_sum', 'has_avg', 'has_min', 'has_max',
            'num_joins', 'num_tables', 'num_conditions', 'num_aggregates',
            'num_equals', 'num_comparisons', 'num_likes', 'num_ins',
            'num_strings', 'num_numbers', 'has_subquery', 'has_union',
            'has_distinct', 'has_case'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in dual_success.columns]
        missing_features = set(feature_columns) - set(available_features)
        
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
        
        logging.info(f"Using {len(available_features)} features: {available_features}")
        
        # Extract features
        X = dual_success[available_features].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Extract labels (optimal engine)
        y = dual_success['optimal_engine'].copy()
        
        # Add query type as feature
        if 'query_type' in dual_success.columns:
            query_type_encoded = pd.get_dummies(dual_success['query_type'], prefix='query_type')
            X = pd.concat([X, query_type_encoded], axis=1)
        
        # Add dataset as feature
        if 'dataset' in dual_success.columns:
            dataset_encoded = pd.get_dummies(dual_success['dataset'], prefix='dataset')
            X = pd.concat([X, dataset_encoded], axis=1)
        
        self.feature_names = list(X.columns)
        logging.info(f"Final feature set: {len(self.feature_names)} features")
        
        return X, y, dual_success
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train LightGBM model with hyperparameter optimization"""
        logging.info("🚀 Training LightGBM model...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        logging.info(f"Classes: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logging.info(f"Training set: {len(X_train):,} samples")
        logging.info(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter optimization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100],
            'min_data_in_leaf': [10, 20, 50],
            'feature_fraction': [0.8, 0.9, 1.0],
        }
        
        # Base model
        base_model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # Grid search
        logging.info("Performing hyperparameter optimization...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train final model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': best_params,
            'best_cv_score': grid_search.best_score_,
            'feature_importance': feature_importance.to_dict('records'),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.label_encoder.classes_,
                                                         output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return results
    
    def save_model(self):
        """Save trained model and preprocessing objects"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'models/final_lightgbm_model_{timestamp}.joblib'
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, model_path)
        
        # Save preprocessors
        scaler_path = f'models/final_scaler_{timestamp}.joblib'
        encoder_path = f'models/final_label_encoder_{timestamp}.joblib'
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save metadata
        metadata = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'encoder_path': encoder_path,
            'feature_names': self.feature_names,
            'classes': self.label_encoder.classes_.tolist(),
            'training_timestamp': timestamp,
            'training_stats': self.training_stats
        }
        
        metadata_path = f'models/final_model_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logging.info(f"💾 Model saved to {model_path}")
        logging.info(f"📊 Metadata saved to {metadata_path}")
        
        return model_path, metadata_path
    
    def print_training_summary(self, results: dict, data_stats: dict):
        """Print comprehensive training summary"""
        logging.info("\n" + "="*60)
        logging.info("🎯 FINAL LIGHTGBM MODEL TRAINING SUMMARY")
        logging.info("="*60)
        
        # Data statistics
        logging.info("📊 TRAINING DATA:")
        logging.info(f"   Total records: {data_stats.get('total_records', 'N/A'):,}")
        logging.info(f"   Successful dual executions: {data_stats.get('dual_executions', 'N/A'):,}")
        logging.info(f"   Features used: {len(self.feature_names)}")
        
        # Model performance
        logging.info("\n🎯 MODEL PERFORMANCE:")
        logging.info(f"   Test accuracy: {results['test_accuracy']:.4f}")
        logging.info(f"   Cross-validation: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        logging.info(f"   Best CV score: {results['best_cv_score']:.4f}")
        
        # Class distribution
        logging.info("\n📈 CLASSIFICATION REPORT:")
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                logging.info(f"   {class_name}: precision={metrics['precision']:.3f}, "
                           f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
        
        # Top features
        logging.info("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for i, feat in enumerate(results['feature_importance'][:10]):
            logging.info(f"   {i+1:2d}. {feat['feature']}: {feat['importance']:.4f}")
        
        # Best parameters
        logging.info(f"\n⚙️  BEST PARAMETERS:")
        for param, value in results['best_params'].items():
            logging.info(f"   {param}: {value}")
    
    def run_training(self):
        """Main training pipeline"""
        logging.info("🚀 Starting Final LightGBM Model Training")
        logging.info("=" * 60)
        
        # Load data
        df = self.load_training_data()
        if df.empty:
            logging.error("No training data available!")
            return False
        
        # Prepare features and labels
        X, y, dual_success = self.prepare_features_and_labels(df)
        if X is None:
            logging.error("Failed to prepare training data!")
            return False
        
        # Data statistics
        data_stats = {
            'total_records': len(df),
            'dual_executions': len(dual_success),
            'postgresql_preferred': len(dual_success[dual_success['optimal_engine'] == 'postgresql']),
            'duckdb_preferred': len(dual_success[dual_success['optimal_engine'] == 'duckdb']),
        }
        
        self.training_stats = data_stats
        
        # Train model
        results = self.train_model(X, y)
        
        # Save model
        model_path, metadata_path = self.save_model()
        
        # Print summary
        self.print_training_summary(results, data_stats)
        
        logging.info(f"\n🎉 Training completed successfully!")
        logging.info(f"📂 Model files saved in models/ directory")
        
        return True

def main():
    """Main function"""
    trainer = FinalLightGBMTrainer()
    success = trainer.run_training()
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())