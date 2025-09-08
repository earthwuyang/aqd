#!/usr/bin/env python3
"""
Python LightGBM Training Framework for AQD
Trains models to predict log-transformed execution time gap between PostgreSQL and DuckDB
"""

import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import joblib
import argparse
import os

class AQDLightGBMTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.shap_explainer = None
        self.feature_importance = []
        
    def load_data(self, input_file):
        """Load training data from JSON file"""
        print(f"Loading training data from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Extract features (all columns except target and metadata)
        metadata_cols = ['query_id', 'query_text', 'postgres_time', 'duckdb_time']
        target_col = 'log_time_gap'  # log(postgres_time / duckdb_time)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in metadata_cols + [target_col]]
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        
        # Handle missing values
        X = X.fillna(0)
        
        self.feature_names = feature_cols
        
        print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
        print(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature columns: {feature_cols}")
        
        return X, y
    
    def train_model(self, X, y, params=None):
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        # Default parameters optimized for AQD
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=self.feature_names)
        
        # Train model
        start_time = datetime.now()
        
        self.model = lgb.train(
            default_params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)  # Suppress training logs
            ]
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = list(zip(self.feature_names, importance))
        self.feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        results = {
            'training_time': training_time,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'num_features': len(self.feature_names),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'feature_importance': [
                {'feature': name, 'importance': float(imp)} 
                for name, imp in self.feature_importance[:20]  # Top 20
            ]
        }
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        return results
    
    def perform_shap_analysis(self, X, max_samples=1000):
        """Perform SHAP analysis for feature interpretation"""
        print("Performing SHAP analysis...")
        
        if self.model is None:
            raise ValueError("Model must be trained before SHAP analysis")
        
        # Sample data for SHAP (can be computationally expensive)
        if len(X) > max_samples:
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_idx]
        else:
            X_sample = X
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Calculate feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(0)
        shap_feature_importance = list(zip(self.feature_names, shap_importance))
        shap_feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"SHAP analysis completed for {len(X_sample)} samples")
        print("Top 10 SHAP features:")
        for i, (name, imp) in enumerate(shap_feature_importance[:10]):
            print(f"  {i+1}. {name}: {imp:.4f}")
        
        return {
            'shap_feature_importance': [
                {'feature': name, 'shap_importance': float(imp)} 
                for name, imp in shap_feature_importance[:20]
            ]
        }
    
    def save_model(self, output_file):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        print(f"Saving model to {output_file}")
        self.model.save_model(output_file)
        
        # Also save metadata
        metadata_file = output_file.replace('.txt', '_metadata.json')
        metadata = {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'feature_importance': [
                {'feature': name, 'importance': float(imp)} 
                for name, imp in self.feature_importance
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_file}")
    
    def predict(self, X):
        """Make predictions with trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)

def main():
    parser = argparse.ArgumentParser(description='AQD LightGBM Trainer')
    parser.add_argument('input_file', help='Input training data JSON file')
    parser.add_argument('output_model', help='Output model file (.txt)')
    parser.add_argument('--results', help='Output results JSON file', default='lightgbm_results.json')
    parser.add_argument('--shap', action='store_true', help='Perform SHAP analysis')
    parser.add_argument('--shap-samples', type=int, default=1000, help='Max samples for SHAP analysis')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        sys.exit(1)
    
    # Initialize trainer
    trainer = AQDLightGBMTrainer()
    
    try:
        # Load data
        X, y = trainer.load_data(args.input_file)
        
        if y is None:
            print("Error: No target column found in data")
            sys.exit(1)
        
        # Train model
        results = trainer.train_model(X, y)
        
        # SHAP analysis if requested
        if args.shap:
            shap_results = trainer.perform_shap_analysis(X, args.shap_samples)
            results.update(shap_results)
        
        # Save model
        trainer.save_model(args.output_model)
        
        # Save results
        with open(args.results, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.results}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()