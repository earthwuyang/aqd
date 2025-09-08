#!/usr/bin/env python3
"""
Real ML Model Training on Actual Database Execution Data
Trains LightGBM model on real SQLite/DuckDB query execution data
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealAQDModelTrainer:
    """Train ML models on real database execution data"""
    
    def __init__(self):
        self.base_dir = Path('/home/wuy/DB/pg_duckdb_postgres')
        self.data_dir = self.base_dir / 'real_data'
        self.models_dir = self.base_dir / 'models'
        self.results_dir = self.base_dir / 'results'
        
        # Create directories
        for dir_path in [self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_real_data(self):
        """Load real execution data"""
        data_file = self.data_dir / 'real_execution_data.csv'
        
        if not data_file.exists():
            logger.error(f"Real data file not found: {data_file}")
            return None
            
        df = pd.read_csv(data_file)
        logger.info(f"Loaded real execution data: {len(df)} queries")
        logger.info(f"Features available: {len([col for col in df.columns if col.startswith('feature_')])}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target variables"""
        logger.info("Preparing features and targets...")
        
        # Extract feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Combine features from both engines (use SQLite features as primary)
        sqlite_features = [col for col in feature_cols if 'sqlite' in col]
        duck_features = [col for col in feature_cols if 'duck' in col]
        
        logger.info(f"SQLite features: {len(sqlite_features)}")
        logger.info(f"DuckDB features: {len(duck_features)}")
        
        # Use SQLite features for consistency (representing PostgreSQL-like)
        X = df[sqlite_features].copy()
        
        # Target: predict which engine is faster (binary classification)
        y_classification = (df['best_engine'] == 'duckdb').astype(int)  # 1 if DuckDB faster, 0 if SQLite faster
        
        # Target: predict execution time difference (regression)
        y_regression = df['log_time_gap'].fillna(0)
        
        # Clean feature names (remove 'feature_sqlite_' prefix)
        X.columns = [col.replace('feature_sqlite_', '') for col in X.columns]
        
        logger.info(f"Final feature matrix: {X.shape}")
        logger.info(f"Classification target distribution:")
        logger.info(f"  SQLite faster (0): {(y_classification == 0).sum()} ({(y_classification == 0).mean():.1%})")
        logger.info(f"  DuckDB faster (1): {(y_classification == 1).sum()} ({(y_classification == 1).mean():.1%})")
        
        return X, y_classification, y_regression
    
    def train_classification_model(self, X, y):
        """Train binary classification model (which engine to choose)"""
        logger.info("Training classification model (engine selection)...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train LightGBM classifier
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_pred_train = (model.predict(X_train) > 0.5).astype(int)
        y_pred_test = (model.predict(X_test) > 0.5).astype(int)
        y_pred_proba = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"Classification Results:")
        logger.info(f"  Training Accuracy: {train_accuracy:.3f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.3f}")
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            logger.info(f"{i+1:2d}. {row['feature']}: {row['importance']:.2f}")
        
        return model, {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance,
            'predictions': y_pred_test,
            'probabilities': y_pred_proba,
            'test_labels': y_test
        }
    
    def train_regression_model(self, X, y):
        """Train regression model (predict execution time difference)"""
        logger.info("Training regression model (execution time gap)...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train LightGBM regressor
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"Regression Results:")
        logger.info(f"  Training RMSE: {train_rmse:.3f}")
        logger.info(f"  Test RMSE: {test_rmse:.3f}")
        logger.info(f"  Training MAE: {train_mae:.3f}")
        logger.info(f"  Test MAE: {test_mae:.3f}")
        logger.info(f"  Training R²: {train_r2:.3f}")
        logger.info(f"  Test R²: {test_r2:.3f}")
        
        return model, {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test,
            'test_labels': y_test
        }
    
    def analyze_query_patterns(self, df):
        """Analyze query execution patterns"""
        logger.info("Analyzing query execution patterns...")
        
        # OLTP vs OLAP performance
        oltp_data = df[df['query_type'] == 'oltp']
        olap_data = df[df['query_type'] == 'olap']
        
        patterns = {
            'total_queries': len(df),
            'oltp_queries': len(oltp_data),
            'olap_queries': len(olap_data),
            'overall_sqlite_faster': (df['best_engine'] == 'sqlite').mean(),
            'overall_duckdb_faster': (df['best_engine'] == 'duckdb').mean(),
            'oltp_sqlite_faster': (oltp_data['best_engine'] == 'sqlite').mean() if len(oltp_data) > 0 else 0,
            'oltp_duckdb_faster': (oltp_data['best_engine'] == 'duckdb').mean() if len(oltp_data) > 0 else 0,
            'olap_sqlite_faster': (olap_data['best_engine'] == 'sqlite').mean() if len(olap_data) > 0 else 0,
            'olap_duckdb_faster': (olap_data['best_engine'] == 'duckdb').mean() if len(olap_data) > 0 else 0,
            'avg_sqlite_time': df['sqlite_execution_time'].mean(),
            'avg_duckdb_time': df['duck_execution_time'].mean(),
            'median_sqlite_time': df['sqlite_execution_time'].median(),
            'median_duckdb_time': df['duck_execution_time'].median()
        }
        
        logger.info(f"Query Execution Patterns:")
        logger.info(f"  Total queries: {patterns['total_queries']}")
        logger.info(f"  OLTP queries: {patterns['oltp_queries']} ({patterns['oltp_queries']/patterns['total_queries']:.1%})")
        logger.info(f"  OLAP queries: {patterns['olap_queries']} ({patterns['olap_queries']/patterns['total_queries']:.1%})")
        logger.info(f"  SQLite faster overall: {patterns['overall_sqlite_faster']:.1%}")
        logger.info(f"  DuckDB faster overall: {patterns['overall_duckdb_faster']:.1%}")
        logger.info(f"  OLTP - SQLite faster: {patterns['oltp_sqlite_faster']:.1%}")
        logger.info(f"  OLAP - DuckDB faster: {patterns['olap_duckdb_faster']:.1%}")
        
        return patterns
    
    def save_models_and_results(self, classification_model, classification_results, 
                              regression_model, regression_results, patterns):
        """Save trained models and results"""
        logger.info("Saving models and results...")
        
        # Save classification model
        classification_model.save_model(str(self.models_dir / 'real_classification_model.txt'))
        
        # Save regression model
        regression_model.save_model(str(self.models_dir / 'real_regression_model.txt'))
        
        # Save results
        results = {
            'data_summary': patterns,
            'classification_performance': {
                'train_accuracy': float(classification_results['train_accuracy']),
                'test_accuracy': float(classification_results['test_accuracy']),
                'feature_importance': classification_results['feature_importance'].head(20).to_dict('records')
            },
            'regression_performance': {
                'train_rmse': float(regression_results['train_rmse']),
                'test_rmse': float(regression_results['test_rmse']),
                'train_mae': float(regression_results['train_mae']),
                'test_mae': float(regression_results['test_mae']),
                'train_r2': float(regression_results['train_r2']),
                'test_r2': float(regression_results['test_r2'])
            }
        }
        
        results_file = self.results_dir / 'real_ml_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return results

def main():
    """Main training pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    REAL ML MODEL TRAINING ON DATABASE EXECUTION DATA                ║
    ║                        Train LightGBM on Actual Query Performance                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    trainer = RealAQDModelTrainer()
    
    try:
        # Load real execution data
        logger.info("Loading real database execution data...")
        df = trainer.load_real_data()
        if df is None:
            return
        
        # Analyze query patterns
        patterns = trainer.analyze_query_patterns(df)
        
        # Prepare features
        X, y_classification, y_regression = trainer.prepare_features(df)
        
        # Train classification model (engine selection)
        classification_model, classification_results = trainer.train_classification_model(X, y_classification)
        
        # Train regression model (execution time prediction)
        regression_model, regression_results = trainer.train_regression_model(X, y_regression)
        
        # Save models and results
        results = trainer.save_models_and_results(
            classification_model, classification_results,
            regression_model, regression_results, 
            patterns
        )
        
        logger.info("\n" + "="*80)
        logger.info("🎉 REAL ML MODEL TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Classification Accuracy: {classification_results['test_accuracy']:.1%}")
        logger.info(f"📊 Regression R²: {regression_results['test_r2']:.3f}")
        logger.info(f"📁 Models saved in: {trainer.models_dir}")
        logger.info(f"📈 Results saved in: {trainer.results_dir}")
        
        print(f"\n🚀 KEY FINDINGS:")
        print(f"   • Engine selection accuracy: {classification_results['test_accuracy']:.1%}")
        print(f"   • DuckDB outperforms SQLite in {patterns['overall_duckdb_faster']:.1%} of queries")
        print(f"   • OLAP queries: DuckDB faster {patterns['olap_duckdb_faster']:.1%} of the time")
        print(f"   • OLTP queries: SQLite faster {patterns['oltp_sqlite_faster']:.1%} of the time")
        print(f"   • Average speedup when using correct engine: {1/min(patterns['avg_sqlite_time'], patterns['avg_duckdb_time'])*max(patterns['avg_sqlite_time'], patterns['avg_duckdb_time']):.1f}x")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()