#!/usr/bin/env python3
"""
AQD LightGBM Model Training
Trains LightGBM model to predict log(postgres_time / duckdb_time)
using AQD features extracted from PostgreSQL.
Works with the new unified JSON data format.
"""

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        """Load training data from unified JSON format"""
        print("📊 Loading training data from unified JSON format...")
        
        all_samples = []
        
        # First try to load the global unified file
        global_file = self.data_path / 'all_datasets_unified_training_data.json'
        if global_file.exists():
            print(f"Loading global unified file: {global_file}")
            with open(global_file, 'r') as f:
                data = json.load(f)
            print(f"  Found {len(data)} records in global file")
            all_samples.extend(self._process_unified_data(data, 'global'))
        else:
            # Load individual dataset unified files
            unified_files = list(self.data_path.glob("*_unified_training_data.json"))
            if not unified_files:
                # Fallback to old format if no unified files found
                return self._load_legacy_format()
            
            print(f"Found {len(unified_files)} unified data files")
            
            for file_path in unified_files:
                if 'all_datasets' in file_path.name:
                    continue  # Skip if we already loaded global
                    
                dataset_name = file_path.stem.replace('_unified_training_data', '')
                print(f"  Loading {dataset_name}...")
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                samples = self._process_unified_data(data, dataset_name)
                all_samples.extend(samples)
                print(f"    Processed {len(samples)} valid samples")
        
        if not all_samples:
            raise ValueError("No valid training samples found!")
        
        df = pd.DataFrame(all_samples)
        
        print(f"\n✅ Loaded {len(df)} training samples")
        print(f"Datasets: {df['dataset'].unique()}")
        print(f"Query types: {df['query_type'].value_counts().to_dict()}")
        
        # Show feature statistics
        feature_cols = [col for col in df.columns if col.startswith('aqd_feature_')]
        print(f"Number of AQD features: {len(feature_cols)}")
        
        return df
    
    def _process_unified_data(self, data, dataset_name):
        """Process records from unified JSON format"""
        samples = []
        
        for record in data:
            # Check for required fields
            if (record.get('log_time_difference') is None or
                record.get('postgres_time') is None or
                record.get('duckdb_time') is None):
                continue
            
            # Skip invalid time differences
            if not np.isfinite(record['log_time_difference']):
                continue
            
            sample = {
                'dataset': record.get('dataset', dataset_name),
                'query_type': record.get('query_type', 'Unknown'),
                'postgres_time': record['postgres_time'],
                'duckdb_time': record['duckdb_time'],
                'log_time_difference': record['log_time_difference'],
                'query': record.get('query_text', '')[:200]  # Truncate for memory
            }
            
            # Add all AQD features from the features dictionary
            features = record.get('features', {})
            if features:
                for feat_name, feat_value in features.items():
                    if isinstance(feat_value, (int, float)) and np.isfinite(feat_value):
                        # Normalize feature names
                        if feat_name.startswith('aqd_'):
                            sample[feat_name] = feat_value
                        else:
                            sample[f'aqd_{feat_name}'] = feat_value
            
            # Add derived query features as backup
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
            
            samples.append(sample)
        
        return samples
    
    def _load_legacy_format(self):
        """Fallback to load old execution_data.json format"""
        print("⚠️  No unified files found, trying legacy format...")
        
        all_samples = []
        execution_files = list(self.data_path.glob("*_execution_data.json"))
        
        if not execution_files:
            raise FileNotFoundError(f"No data files found in {self.data_path}")
        
        for file_path in execution_files:
            dataset_name = file_path.stem.replace('_execution_data', '')
            print(f"  Loading legacy format: {dataset_name}...")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for record in data:
                if (record.get('executed_postgres') and record.get('executed_duckdb') and
                    record.get('postgres_time') is not None and record.get('duckdb_time') is not None and
                    record.get('postgres_time') > 0 and record.get('duckdb_time') > 0):
                    
                    log_time_difference = np.log(record['postgres_time'] / record['duckdb_time'])
                    
                    sample = {
                        'dataset': dataset_name,
                        'query_type': record.get('query_type', 'Unknown'),
                        'postgres_time': record['postgres_time'],
                        'duckdb_time': record['duckdb_time'],
                        'log_time_difference': log_time_difference,
                        'query': record.get('query_text', '')[:200]
                    }
                    
                    # Add AQD features
                    aqd_features = record.get('aqd_features', {})
                    if aqd_features:
                        for feat_name, feat_value in aqd_features.items():
                            if isinstance(feat_value, (int, float)) and not np.isnan(feat_value):
                                sample[f'aqd_{feat_name}'] = feat_value
                    
                    all_samples.append(sample)
        
        return pd.DataFrame(all_samples)
    
    def prepare_features(self, df):
        """Prepare feature matrix and target"""
        print("\n🔧 Preparing features...")
        
        # Identify feature columns (all aqd_ columns plus derived features)
        feature_cols = []
        
        # First priority: AQD features from PostgreSQL
        aqd_feature_cols = [col for col in df.columns if col.startswith('aqd_feature_')]
        feature_cols.extend(sorted(aqd_feature_cols))
        
        # Add complexity score if available
        if 'aqd_complexity_score' in df.columns:
            feature_cols.append('aqd_complexity_score')
        
        # Add derived query features
        derived_features = ['query_length', 'has_join', 'has_group_by', 
                          'has_order_by', 'has_where', 'has_aggregation',
                          'is_ap_query', 'is_tp_query']
        for feat in derived_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        if not feature_cols:
            raise ValueError("No features found! Check data collection.")
        
        print(f"Using {len(feature_cols)} features")
        print(f"  - AQD features: {len(aqd_feature_cols)}")
        print(f"  - Derived features: {len([f for f in derived_features if f in feature_cols])}")
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        y = df['log_time_difference']
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train LightGBM model"""
        print("\n🚀 Training LightGBM model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(X.columns))
        valid_data = lgb.Dataset(X_test, label=y_test, feature_name=list(X.columns))
        
        # Set parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
        )
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        self.results = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        print(f"\n📈 Model Performance:")
        print(f"  Training RMSE: {self.results['train_rmse']:.4f}")
        print(f"  Test RMSE: {self.results['test_rmse']:.4f}")
        print(f"  Training MAE: {self.results['train_mae']:.4f}")
        print(f"  Test MAE: {self.results['test_mae']:.4f}")
        print(f"  Training R²: {self.results['train_r2']:.4f}")
        print(f"  Test R²: {self.results['test_r2']:.4f}")
        
        return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def analyze_routing_performance(self, df, X_test, y_test, y_pred_test):
        """Analyze routing performance with confusion matrix and comparisons"""
        print("\n" + "="*60)
        print("🎯 ROUTING PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Get test indices to match with original data
        test_indices = X_test.index
        test_data = df.loc[test_indices].copy()
        
        # Add predictions
        test_data['predicted_log_diff'] = y_pred_test
        test_data['actual_log_diff'] = y_test.values
        
        # Binary routing decisions (positive = use PostgreSQL, negative = use DuckDB)
        test_data['actual_routing'] = (test_data['actual_log_diff'] > 0).astype(int)
        test_data['predicted_routing'] = (test_data['predicted_log_diff'] > 0).astype(int)
        
        # Confusion Matrix
        cm = confusion_matrix(test_data['actual_routing'], test_data['predicted_routing'])
        
        print("\n📊 Confusion Matrix:")
        print("                 Predicted")
        print("                 DuckDB  PostgreSQL")
        print(f"Actual DuckDB      {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"Actual PostgreSQL  {cm[1,0]:5d}    {cm[1,1]:5d}")
        
        # Calculate metrics
        total = len(test_data)
        correct = cm[0,0] + cm[1,1]
        accuracy = correct / total * 100
        
        # True/False Positives/Negatives
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📈 Classification Metrics:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Calculate execution times for different routing strategies
        print("\n⏱️  End-to-End Execution Time Analysis:")
        
        # 1. Optimal routing (always choose faster)
        optimal_time = test_data[['postgres_time', 'duckdb_time']].min(axis=1).sum()
        
        # 2. LightGBM routing
        lgb_time = 0
        for idx, row in test_data.iterrows():
            if row['predicted_routing'] == 1:  # Route to PostgreSQL
                lgb_time += row['postgres_time']
            else:  # Route to DuckDB
                lgb_time += row['duckdb_time']
        
        # 3. Always PostgreSQL
        all_pg_time = test_data['postgres_time'].sum()
        
        # 4. Always DuckDB
        all_duck_time = test_data['duckdb_time'].sum()
        
        # 5. Cost-threshold routing (test multiple thresholds)
        print("\n  Strategy Comparison:")
        print(f"  {'Strategy':<25} {'Total Time (s)':<15} {'vs Optimal':<15} {'Improvement':<15}")
        print("  " + "-"*70)
        
        print(f"  {'Optimal (Oracle)':<25} {optimal_time:>14.3f} {'':<15} {'Baseline':<15}")
        print(f"  {'LightGBM ML':<25} {lgb_time:>14.3f} {f'+{(lgb_time/optimal_time-1)*100:.1f}%':<15} {f'vs All-PG: {(1-lgb_time/all_pg_time)*100:+.1f}%':<15}")
        print(f"  {'Always PostgreSQL':<25} {all_pg_time:>14.3f} {f'+{(all_pg_time/optimal_time-1)*100:.1f}%':<15} {'-':<15}")
        print(f"  {'Always DuckDB':<25} {all_duck_time:>14.3f} {f'+{(all_duck_time/optimal_time-1)*100:.1f}%':<15} {'-':<15}")
        
        # Test cost-threshold routing with different thresholds
        if 'aqd_feature_9' in test_data.columns or 'aqd_feature_10' in test_data.columns:
            # Use total cost feature if available
            cost_col = 'aqd_feature_10' if 'aqd_feature_10' in test_data.columns else 'aqd_feature_9'
            
            thresholds = [100, 500, 1000, 5000, 10000]
            print(f"\n  Cost-Threshold Routing (using {cost_col}):")
            
            for threshold in thresholds:
                cost_time = 0
                cost_routing = (test_data[cost_col] > threshold).astype(int)
                
                for idx, row in test_data.iterrows():
                    if cost_routing[idx] == 1:  # High cost -> DuckDB
                        cost_time += row['duckdb_time']
                    else:  # Low cost -> PostgreSQL
                        cost_time += row['postgres_time']
                
                print(f"  {'Threshold=' + str(threshold):<25} {cost_time:>14.3f} {f'+{(cost_time/optimal_time-1)*100:.1f}%':<15} {f'vs All-PG: {(1-cost_time/all_pg_time)*100:+.1f}%':<15}")
        
        # Analyze misrouted queries
        print("\n🔍 Misrouting Analysis:")
        misrouted = test_data[test_data['actual_routing'] != test_data['predicted_routing']]
        print(f"  Total misrouted: {len(misrouted)} / {total} ({len(misrouted)/total*100:.1f}%)")
        
        if len(misrouted) > 0:
            # Calculate time penalty for misrouting
            time_penalty = 0
            for idx, row in misrouted.iterrows():
                if row['predicted_routing'] == 1:  # Wrongly routed to PostgreSQL
                    time_penalty += row['postgres_time'] - row['duckdb_time']
                else:  # Wrongly routed to DuckDB
                    time_penalty += row['duckdb_time'] - row['postgres_time']
            
            print(f"  Total time penalty: {time_penalty:.3f}s")
            print(f"  Average penalty per misroute: {time_penalty/len(misrouted):.3f}s")
            
            # Show distribution of misrouted queries
            print(f"\n  Misrouting by query type:")
            if 'query_type' in misrouted.columns:
                for qtype in misrouted['query_type'].unique():
                    qtype_misrouted = misrouted[misrouted['query_type'] == qtype]
                    print(f"    {qtype}: {len(qtype_misrouted)} queries")
        
        # Save detailed results
        self.results['routing_accuracy'] = accuracy
        self.results['confusion_matrix'] = cm.tolist()
        self.results['precision'] = precision
        self.results['recall'] = recall
        self.results['f1_score'] = f1
        self.results['optimal_time'] = optimal_time
        self.results['lgb_time'] = lgb_time
        self.results['all_pg_time'] = all_pg_time
        self.results['all_duck_time'] = all_duck_time
        self.results['efficiency_vs_optimal'] = lgb_time / optimal_time
        self.results['improvement_vs_all_pg'] = (all_pg_time - lgb_time) / all_pg_time
        
        return test_data
    
    def analyze_features(self, X_train):
        """Analyze feature importance"""
        print("\n🔍 Analyzing feature importance...")
        
        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(feature_imp.head(20).to_string())
        
        # SHAP analysis for better interpretability
        try:
            print("\n📊 Running SHAP analysis...")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_train.sample(min(1000, len(X_train))))
            
            # Save SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_train.sample(min(1000, len(X_train))), 
                            show=False, max_display=20)
            
            plot_path = self.models_dir / 'shap_summary.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  SHAP summary plot saved to {plot_path}")
            
            # Get mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({
                'feature': X_train.columns,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            print("\nTop Features by SHAP Importance:")
            print(shap_importance.head(10).to_string())
            
            return feature_imp, shap_importance
            
        except Exception as e:
            print(f"  ⚠️ SHAP analysis failed: {e}")
            return feature_imp, None
    
    def save_model(self):
        """Save trained model and metadata"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        model_path = self.models_dir / 'lightgbm_model.txt'
        self.model.save_model(str(model_path))
        print(f"\n💾 Model saved to {model_path}")
        
        # Save model metadata
        metadata = {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'results': self.results,
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = self.models_dir / 'lightgbm_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to {metadata_path}")
        
        # Also save as pickle for Python use
        pickle_path = self.models_dir / 'lightgbm_model.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'results': self.results
            }, f)
        print(f"  Pickle saved to {pickle_path}")
    
    def plot_predictions(self, y_test, y_pred_test):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        plt.xlabel('Actual log(postgres_time / duckdb_time)')
        plt.ylabel('Predicted log(postgres_time / duckdb_time)')
        plt.title('LightGBM Model: Actual vs Predicted')
        
        # Add R² score
        plt.text(0.05, 0.95, f'R² = {self.results["test_r2"]:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top')
        
        plot_path = self.models_dir / 'predictions_scatter.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Prediction plot saved to {plot_path}")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("AQD LightGBM Model Training")
    print("=" * 60)
    
    trainer = AQDLightGBMTrainer()
    
    # Load data
    df = trainer.load_training_data()
    
    # Prepare features
    X, y = trainer.prepare_features(df)
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = trainer.train_model(X, y)
    
    # Analyze routing performance with confusion matrix and comparisons
    test_data = trainer.analyze_routing_performance(df, X_test, y_test, y_pred_test)
    
    # Analyze features
    feature_imp, shap_imp = trainer.analyze_features(X_train)
    
    # Save model
    trainer.save_model()
    
    # Plot results
    trainer.plot_predictions(y_test, y_pred_test)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"  Model saved to: {trainer.models_dir}")
    print(f"  Routing Accuracy: {trainer.results.get('routing_accuracy', 0):.1f}%")
    print(f"  Efficiency vs Optimal: {trainer.results.get('efficiency_vs_optimal', 0):.2f}x")
    print(f"  Improvement vs All-PostgreSQL: {trainer.results.get('improvement_vs_all_pg', 0)*100:.1f}%")

if __name__ == "__main__":
    main()