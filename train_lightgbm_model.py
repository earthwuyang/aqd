#!/usr/bin/env python3
"""
AQD LightGBM Model Training
Trains LightGBM model to predict log(postgres_time / duckdb_time)
using only cached AQD kernel features (no label leakage).
"""

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import shap
from datetime import datetime

class AQDLightGBMTrainer:
    """
    Trains LightGBM model for AQD query routing predictions
    """
    
    def __init__(self, data_path=None, output_dir=None):
        base_dir = Path(__file__).resolve().parent
        # Default to repo-relative paths if not provided
        self.data_path = Path(data_path) if data_path else (base_dir / 'data' / 'execution_data')
        self.models_dir = Path(output_dir) if output_dir else (base_dir / 'models')
        self.model = None
        self.feature_names = None
        self.results = {}
    
    def load_training_data(self):
        """Load training data from collected execution data"""
        print("📊 Loading training data...")
        
        all_samples = []
        
        # Load from execution data JSON files
        execution_files = list(self.data_path.glob("*_execution_data.json"))
        if not execution_files:
            raise FileNotFoundError(f"No execution data files found in {self.data_path}")
        
        print(f"Found {len(execution_files)} execution data files")
        
        for file_path in execution_files:
            dataset_name = file_path.stem.replace('_execution_data', '')
            print(f"  Loading {dataset_name}...")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract successful dual executions
            for record in data:
                if (record.get('executed_postgres') and record.get('executed_duckdb') and
                    record.get('postgres_time') is not None and record.get('duckdb_time') is not None and
                    record.get('postgres_time') > 0 and record.get('duckdb_time') > 0):
                    
                    # Calculate derived metrics
                    time_difference = record['postgres_time'] - record['duckdb_time']
                    log_time_difference = np.log(record['postgres_time'] / record['duckdb_time'])
                    
                    sample = {
                        'dataset': dataset_name,
                        'query_type': record.get('query_type', 'Unknown'),
                        'postgres_time': record['postgres_time'],
                        'duckdb_time': record['duckdb_time'],
                        'time_difference': time_difference,
                        'log_time_difference': log_time_difference,
                        'query': record.get('query_text', '')[:200]  # Truncate for memory
                    }
                    
                    # Add AQD features if available
                    aqd_features = record.get('aqd_features', {})
                    if aqd_features:
                        for feat_name, feat_value in aqd_features.items():
                            if isinstance(feat_value, (int, float)) and not np.isnan(feat_value):
                                sample[f'aqd_{feat_name}'] = feat_value
                    
                    # Add basic query features as fallback
                    query_text = record.get('query_text', '').upper()
                    sample.update({
                        'query_length': len(query_text),
                        'has_join': int('JOIN' in query_text),
                        'has_group_by': int('GROUP BY' in query_text),
                        'has_order_by': int('ORDER BY' in query_text),
                        'has_where': int('WHERE' in query_text),
                        'has_aggregation': int(any(agg in query_text for agg in ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX'])),
                        'is_ap_query': int(record.get('query_type') == 'AP'),
                        'is_tp_query': int(record.get('query_type') == 'TP')
                    })
                    
                    all_samples.append(sample)
        
        if not all_samples:
            raise ValueError("No successful dual executions found! Need both PostgreSQL and DuckDB successes.")
        
        df = pd.DataFrame(all_samples)
        
        print(f"Loaded {len(df)} dual execution samples")
        print(f"Datasets: {sorted(df['dataset'].unique())}")
        print(f"Query types: {df['query_type'].value_counts().to_dict()}")
        print(f"Features found: {len([col for col in df.columns if col.startswith('aqd_')])}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the training data"""
        print("🔧 Preprocessing data...")
        
        # Convert boolean features to integers
        bool_columns = df.select_dtypes(include=['bool']).columns
        df[bool_columns] = df[bool_columns].astype(int)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Target variable (log-transformed time ratio as in AQD paper)
        # Use log_time_difference which is already computed
        df['target'] = df['log_time_difference']
        
        print(f"Features after preprocessing: {len(df.columns)}")
        print(f"Target statistics:")
        print(f"  Mean: {df['target'].mean():.3f}")
        print(f"  Std: {df['target'].std():.3f}")
        print(f"  Min: {df['target'].min():.3f}")
        print(f"  Max: {df['target'].max():.3f}")
        
        return df
    
    def prepare_features_target(self, df):
        """Prepare feature matrix and target vector"""
        # Use ONLY AQD kernel cached features to avoid leakage
        feature_cols = [
            col for col in df.columns
            if col.startswith('aqd_') and df[col].dtype in ['int64', 'float64']
        ]
        if not feature_cols:
            raise ValueError("No AQD features found in execution data. Ensure aqd feature logging is enabled and collected.")
        
        X = df[feature_cols]
        y = df['target']
        
        self.feature_names = feature_cols
        print(f"Selected {len(feature_cols)} features for training")
        print("Features:", feature_cols)
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train LightGBM model"""
        print("🤖 Training LightGBM model...")
        
        # Split data (adjust for very small datasets)
        if len(X) <= 5:
            # For very small datasets, use all data for training and testing
            X_train, X_test, y_train, y_test = X, X, y, y
            print("⚠️  Using all data for both training and testing due to small dataset size")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Model parameters optimized for AQD task
        params = {
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': random_state
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train, num_iteration=self.model.best_iteration)
        y_pred_test = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        self.results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_features': len(self.feature_names),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'best_iteration': self.model.best_iteration
        }
        
        print("\n📈 Training Results:")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Best iteration: {self.model.best_iteration}")
        
        return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def analyze_feature_importance(self, top_k=15):
        """Analyze and plot feature importance"""
        print("📊 Analyzing feature importance...")
        
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.feature_names
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_k} Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(top_k).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:25s} {row['importance']:8.1f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(top_k), y='feature', x='importance')
        plt.title('LightGBM Feature Importance (Top 15)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        self.models_dir.mkdir(exist_ok=True)
        plt.savefig(self.models_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {self.models_dir / 'feature_importance.png'}")
        
        return importance_df
    
    def perform_shap_analysis(self, X_sample, max_samples=100):
        """Perform SHAP analysis for model interpretability"""
        print("🔍 Performing SHAP analysis...")
        
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Sample data for SHAP (it can be slow on large datasets)
        if len(X_sample) > max_samples:
            X_shap = X_sample.sample(n=max_samples, random_state=42)
        else:
            X_shap = X_sample
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_shap)
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
        
        self.models_dir.mkdir(exist_ok=True)
        plt.savefig(self.models_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
        print(f"SHAP summary plot saved to {self.models_dir / 'shap_summary.png'}")
        plt.close()
        
        return shap_values
    
    def save_model(self):
        """Save the trained model and metadata"""
        if not self.model:
            raise ValueError("No model to save")
        
        self.models_dir.mkdir(exist_ok=True)
        
        # Save LightGBM model
        model_path = self.models_dir / 'lightgbm_model.txt'
        self.model.save_model(str(model_path))
        
        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_names,
            'results': self.results,
            'model_path': str(model_path),
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {self.models_dir}")
        print(f"   - LightGBM model: {model_path}")
        print(f"   - Metadata: {self.models_dir / 'model_metadata.json'}")
        
        return self.models_dir
    
    def train_complete_pipeline(self):
        """Complete training pipeline"""
        print("🚀 Starting AQD LightGBM Training Pipeline")
        print("=" * 60)
        
        try:
            # Load and preprocess data
            df = self.load_training_data()
            df = self.preprocess_data(df)
            
            if len(df) < 3:
                print("❌ Error: Too few training samples. Need at least 3 dual executions.")
                print(f"   Current samples: {len(df)}")
                print("   Try increasing query execution success rate or collecting more data.")
                return False
            elif len(df) < 10:
                print("⚠️  Warning: Very few training samples. Training a demonstration model only.")
                print(f"   Current samples: {len(df)} (recommended: 50+)")
                print("   This model is for demonstration - collect more data for production use.")
            elif len(df) < 50:
                print("⚠️  Warning: Very few training samples. Results may not be reliable.")
            
            # Prepare features and target
            X, y = self.prepare_features_target(df)
            
            # Train model
            X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = self.train_model(X, y)
            
            # Feature importance analysis
            importance_df = self.analyze_feature_importance()
            
            # SHAP analysis (optional, can be slow)
            try:
                shap_values = self.perform_shap_analysis(X_test)
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
            
            # Save model
            output_dir = self.save_model()
            
            # Summary
            print("\n" + "=" * 60)
            print("🎉 AQD LightGBM Training Completed!")
            print("=" * 60)
            print(f"Model performance (Test set):")
            print(f"  - RMSE: {self.results['test_rmse']:.4f}")
            print(f"  - MAE: {self.results['test_mae']:.4f}")
            print(f"  - R²: {self.results['test_r2']:.4f}")
            print(f"Training samples: {self.results['n_train']}")
            print(f"Test samples: {self.results['n_test']}")
            print(f"Features used: {self.results['n_features']}")
            print(f"\nModel saved to: {output_dir}")
            
            # Model quality assessment
            if self.results['test_r2'] > 0.6:
                print("\n✅ Model quality: Good (R² > 0.6)")
            elif self.results['test_r2'] > 0.3:
                print("\n⚠️  Model quality: Moderate (0.3 < R² < 0.6)")
            else:
                print("\n❌ Model quality: Poor (R² < 0.3)")
                print("   Consider collecting more diverse training data")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    import argparse
    parser = argparse.ArgumentParser(description='Train LightGBM for AQD using collected execution data')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory with *_execution_data.json (default: repo data/execution_data)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save models (default: repo models)')
    args = parser.parse_args()

    trainer = AQDLightGBMTrainer(data_path=args.data_dir, output_dir=args.output_dir)
    success = trainer.train_complete_pipeline()
    return success

if __name__ == "__main__":
    import sys
    # Install required packages if needed
    try:
        import lightgbm
        import shap
        import seaborn
        import matplotlib
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install lightgbm shap seaborn matplotlib scikit-learn")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1)
