#!/usr/bin/env python3
"""
Train LightGBM model on all currently collected training data
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lightgbm_current_training.log'),
        logging.StreamHandler()
    ]
)

def load_all_training_data():
    """Load training data from all available sources"""
    all_data = []
    data_sources = []
    
    # Load original collection data
    try:
        with open('data/training_data_round1_partial.json', 'r') as f:
            original_data = json.load(f)
            logging.info(f"Loaded original collection: {len(original_data['data'])} records")
            
            for record in original_data['data']:
                # Convert to unified format
                unified_record = {
                    'query_id': record.get('query_id', f"orig_{len(all_data)}"),
                    'query_text': record.get('query_text', ''),
                    'query_type': record.get('query_type', 'mixed'),
                    'pg_execution_time': record.get('pg_execution_time', 0.0),
                    'duck_execution_time': record.get('duck_execution_time', 0.0),
                    'optimal_engine': record.get('optimal_engine', 'postgresql'),
                    'features': record.get('features', {}),
                    'dataset': 'original_collection'
                }
                all_data.append(unified_record)
            
            data_sources.append(f"Original: {len(original_data['data'])} records")
    
    except Exception as e:
        logging.warning(f"Could not load original collection: {e}")
    
    # Load comprehensive collection data
    comprehensive_files = [
        'data/comprehensive/dataset_1_small_oltp_oltp_partial_1000.json',
        'data/comprehensive/dataset_1_small_oltp_oltp_partial_2000.json', 
        'data/comprehensive/dataset_1_small_oltp_oltp_partial_3000.json'
    ]
    
    for file_path in comprehensive_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    comp_data = json.load(f)
                    records_added = len(comp_data['data'])
                    logging.info(f"Loaded {file_path}: {records_added} records")
                    
                    for record in comp_data['data']:
                        all_data.append(record)
                    
                    data_sources.append(f"{os.path.basename(file_path)}: {records_added} records")
        except Exception as e:
            logging.warning(f"Could not load {file_path}: {e}")
    
    logging.info(f"Total training data loaded: {len(all_data)} records")
    logging.info("Data sources:")
    for source in data_sources:
        logging.info(f"  - {source}")
    
    return all_data

def extract_features_from_record(record):
    """Extract features from a training record"""
    features = {}
    
    # Get features from the record
    record_features = record.get('features', {})
    
    # Query-based features
    query_text = record.get('query_text', '')
    features.update({
        'query_length': len(query_text),
        'num_select': query_text.upper().count('SELECT'),
        'num_join': query_text.upper().count('JOIN'),
        'num_where': query_text.upper().count('WHERE'),
        'num_group_by': query_text.upper().count('GROUP BY'),
        'num_order_by': query_text.upper().count('ORDER BY'),
        'num_having': query_text.upper().count('HAVING'),
        'num_distinct': query_text.upper().count('DISTINCT'),
        'num_union': query_text.upper().count('UNION'),
        'num_subquery': query_text.count('(SELECT'),
        'num_with': query_text.upper().count('WITH'),
        'num_case': query_text.upper().count('CASE'),
        'num_like': query_text.upper().count('LIKE'),
        'num_in': query_text.upper().count(' IN '),
        'num_between': query_text.upper().count('BETWEEN'),
    })
    
    # Plan-based features from record
    features.update({
        'startup_cost': record_features.get('startup_cost', 0),
        'total_cost': record_features.get('total_cost', 0),
        'plan_rows': record_features.get('plan_rows', 0),
        'plan_width': record_features.get('plan_width', 0),
        'num_tables': record_features.get('num_tables', 0),
        'num_joins_plan': record_features.get('num_joins', 0),
        'num_filters': record_features.get('num_filters', 0),
        'num_aggregates': record_features.get('num_aggregates', 0),
        'num_sorts': record_features.get('num_sorts', 0),
        'num_limits': record_features.get('num_limits', 0),
        'plan_depth': record_features.get('plan_depth', 0),
        'num_seqscan': record_features.get('num_seqscan', 0),
        'num_indexscan': record_features.get('num_indexscan', 0),
        'num_hashjoin': record_features.get('num_hashjoin', 0),
        'num_nestloop': record_features.get('num_nestloop', 0),
        'num_mergejoin': record_features.get('num_mergejoin', 0),
    })
    
    # Execution-based features
    pg_time = record.get('pg_execution_time', 0.0)
    duck_time = record.get('duck_execution_time', 0.0)
    
    features.update({
        'pg_execution_time': pg_time,
        'duck_execution_time': duck_time,
        'performance_ratio': duck_time / max(pg_time, 0.001),
        'time_difference': abs(pg_time - duck_time),
        'is_oltp': 1 if record.get('query_type') == 'oltp' else 0,
        'is_olap': 1 if record.get('query_type') == 'olap' else 0,
        'is_mixed': 1 if record.get('query_type') == 'mixed' else 0,
    })
    
    # Additional derived features
    features.update({
        'cost_per_row': features['total_cost'] / max(features['plan_rows'], 1),
        'scan_ratio': features['num_indexscan'] / max(features['num_seqscan'] + features['num_indexscan'], 1),
        'join_complexity': features['num_joins_plan'] / max(features['num_tables'], 1),
        'filter_selectivity': features['num_filters'] / max(features['query_length'], 1),
    })
    
    return features

def prepare_training_data(all_data):
    """Prepare training data for LightGBM"""
    logging.info("Preparing training data...")
    
    X_data = []
    y_data = []
    valid_records = []
    
    for record in all_data:
        try:
            features = extract_features_from_record(record)
            
            # Convert optimal engine to binary label
            optimal_engine = record.get('optimal_engine', 'postgresql')
            label = 0 if optimal_engine == 'postgresql' else 1  # 0=PostgreSQL, 1=DuckDB
            
            # Convert features to list maintaining consistent order
            feature_names = sorted(features.keys())
            feature_vector = [features[name] for name in feature_names]
            
            X_data.append(feature_vector)
            y_data.append(label)
            valid_records.append(record)
            
        except Exception as e:
            logging.warning(f"Error processing record {record.get('query_id', 'unknown')}: {e}")
            continue
    
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    
    logging.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    logging.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y, feature_names, valid_records

def train_lightgbm_model(X, y, feature_names):
    """Train LightGBM model with comprehensive evaluation"""
    logging.info("Training LightGBM model...")
    
    # Check class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    logging.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    
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
        'random_state': 42,
        'is_unbalance': True,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_names)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_score = None
        logging.warning("Could not calculate AUC score (single class in test set)")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importance(importance_type='gain')))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Detailed results
    results = {
        'test_accuracy': accuracy,
        'auc_score': auc_score,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_distribution': np.bincount(y_train).tolist(),
        'test_distribution': np.bincount(y_test).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_importance': feature_importance,
        'top_10_features': sorted_importance[:10],
        'model_params': params,
        'best_iteration': model.best_iteration
    }
    
    return model, results

def main():
    """Main execution function"""
    logging.info("=== Training LightGBM on Current Collected Data ===")
    
    try:
        # Load all available training data
        all_data = load_all_training_data()
        
        if len(all_data) < 100:
            logging.error(f"Insufficient training data: {len(all_data)} records")
            return
        
        # Prepare training data
        X, y, feature_names, valid_records = prepare_training_data(all_data)
        
        if len(X) == 0:
            logging.error("No valid training data after preprocessing")
            return
        
        # Train model
        model, results = train_lightgbm_model(X, y, feature_names)
        
        # Display results
        print("\n" + "="*80)
        print("🎯 LIGHTGBM MODEL TRAINING RESULTS ON CURRENT DATA")
        print("="*80)
        print(f"📊 Test Set Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        
        if results['auc_score']:
            print(f"📈 AUC Score: {results['auc_score']:.4f}")
        
        print(f"📉 Total Training Samples: {results['total_samples']:,}")
        print(f"🎯 Train/Test Split: {results['train_samples']:,}/{results['test_samples']:,}")
        
        # Distribution
        print(f"\n📊 Training Data Distribution:")
        print(f"   PostgreSQL Optimal: {results['train_distribution'][0]:,} ({results['train_distribution'][0]/results['train_samples']*100:.1f}%)")
        if len(results['train_distribution']) > 1:
            print(f"   DuckDB Optimal: {results['train_distribution'][1]:,} ({results['train_distribution'][1]/results['train_samples']*100:.1f}%)")
        
        print(f"\n🧪 Test Data Distribution:")
        print(f"   PostgreSQL Optimal: {results['test_distribution'][0]:,} ({results['test_distribution'][0]/results['test_samples']*100:.1f}%)")
        if len(results['test_distribution']) > 1:
            print(f"   DuckDB Optimal: {results['test_distribution'][1]:,} ({results['test_distribution'][1]/results['test_samples']*100:.1f}%)")
        
        # Classification report
        report = results['classification_report']
        print(f"\n📋 Detailed Classification Metrics:")
        if '0' in report:
            print(f"   PostgreSQL (0): Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}, F1={report['0']['f1-score']:.3f}")
        if '1' in report:
            print(f"   DuckDB (1):     Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}, F1={report['1']['f1-score']:.3f}")
        print(f"   Macro Avg:      Precision={report['macro avg']['precision']:.3f}, Recall={report['macro avg']['recall']:.3f}, F1={report['macro avg']['f1-score']:.3f}")
        
        # Confusion matrix
        cm = results['confusion_matrix']
        print(f"\n🎯 Confusion Matrix:")
        print(f"   Predicted:  PostgreSQL  DuckDB")
        if len(cm) >= 2:
            print(f"   PostgreSQL:     {cm[0][0]:4d}      {cm[0][1]:4d}")
            print(f"   DuckDB:         {cm[1][0]:4d}      {cm[1][1]:4d}")
        
        # Top features
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(results['top_10_features'], 1):
            print(f"   {i:2d}. {feature:<25}: {importance:8.1f}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_file = 'results/lightgbm_current_data_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model
        model_file = 'results/lightgbm_current_data_model.txt'
        model.save_model(model_file)
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"💾 Model saved to: {model_file}")
        print("="*80)
        
        logging.info(f"Training complete - Test Accuracy: {results['test_accuracy']:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()