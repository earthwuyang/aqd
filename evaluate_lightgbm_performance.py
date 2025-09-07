#!/usr/bin/env python3
"""
Quick LightGBM Model Evaluation - Fast training and performance assessment
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def evaluate_lightgbm_performance():
    """Quick training and evaluation of LightGBM model"""
    
    print("🚀 Starting Quick LightGBM Performance Evaluation")
    print("=" * 60)
    
    # Load enhanced training data
    print("📂 Loading training data...")
    with open('data/zsce_training_data_20250907_154000_enhanced.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"📊 Total records: {len(df)}")
    
    # Filter successful dual executions
    dual_success = df[(df['postgresql_success'] == True) & (df['duckdb_success'] == True)].copy()
    print(f"✅ Successful dual executions: {len(dual_success)}")
    
    if len(dual_success) == 0:
        print("❌ No successful dual executions found!")
        return
    
    # Feature columns
    feature_columns = [
        'query_length', 'word_count', 'has_select', 'has_from', 'has_where',
        'has_join', 'has_group_by', 'has_order_by', 'has_having', 'has_limit',
        'has_count', 'has_sum', 'has_avg', 'has_min', 'has_max',
        'num_joins', 'num_tables', 'num_conditions', 'num_aggregates',
        'num_equals', 'num_comparisons', 'num_likes', 'num_ins',
        'num_strings', 'num_numbers', 'has_subquery', 'has_union',
        'has_distinct', 'has_case'
    ]
    
    available_features = [col for col in feature_columns if col in dual_success.columns]
    print(f"🔧 Using {len(available_features)} features")
    
    # Prepare features and labels
    X = dual_success[available_features].fillna(0)
    y = dual_success['optimal_engine']
    
    # Add query type as feature if available
    if 'query_type' in dual_success.columns:
        query_type_encoded = pd.get_dummies(dual_success['query_type'], prefix='query_type')
        X = pd.concat([X, query_type_encoded], axis=1)
        print(f"➕ Added query type features, total features: {len(X.columns)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"🎯 Classes: {label_encoder.classes_}")
    print(f"📈 Class distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = sum(y_encoded == i)
        percentage = count / len(y_encoded) * 100
        print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train LightGBM with simple parameters for quick evaluation
    print("\n🤖 Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate performance metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("🎯 LIGHTGBM MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    print(f"🎯 Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n📊 Detailed Classification Report:")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"   {class_name}:")
            print(f"     Precision: {metrics['precision']:.4f}")
            print(f"     Recall: {metrics['recall']:.4f}")
            print(f"     F1-Score: {metrics['f1-score']:.4f}")
            print(f"     Support: {metrics['support']}")
    
    # Overall metrics
    if 'macro avg' in report:
        print(f"\n   Overall Performance:")
        print(f"     Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"     Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   Predicted →")
    print(f"   Actual ↓     {'':>10} {'':>10}")
    print(f"              {label_encoder.classes_[0]:>10} {label_encoder.classes_[1]:>10}")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"   {class_name:>10} {cm[i][0]:>10} {cm[i][1]:>10}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': list(X.columns),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<20}: {row['importance']:.4f}")
    
    # Performance analysis by query type
    if 'query_type' in dual_success.columns:
        print(f"\n📈 Performance by Query Type:")
        
        test_indices = X_test.index
        test_data = dual_success.loc[test_indices].copy()
        test_data['predicted_engine'] = label_encoder.inverse_transform(y_pred)
        test_data['actual_engine'] = label_encoder.inverse_transform(y_test)
        test_data['correct_prediction'] = test_data['predicted_engine'] == test_data['actual_engine']
        
        for query_type in test_data['query_type'].unique():
            subset = test_data[test_data['query_type'] == query_type]
            accuracy = subset['correct_prediction'].mean()
            print(f"   {query_type} queries: {accuracy:.4f} accuracy ({len(subset)} samples)")
    
    # Performance ratio analysis
    print(f"\n⚡ Query Performance Analysis:")
    ratios = dual_success['performance_ratio'].dropna()
    if len(ratios) > 0:
        print(f"   Average performance ratio (DuckDB/PostgreSQL): {ratios.mean():.3f}")
        print(f"   DuckDB faster queries: {sum(ratios < 0.95)} ({sum(ratios < 0.95)/len(ratios)*100:.1f}%)")
        print(f"   PostgreSQL faster queries: {sum(ratios > 1.05)} ({sum(ratios > 1.05)/len(ratios)*100:.1f}%)")
        print(f"   Similar performance queries: {sum((ratios >= 0.95) & (ratios <= 1.05))} ({sum((ratios >= 0.95) & (ratios <= 1.05))/len(ratios)*100:.1f}%)")
    
    print(f"\n🎉 Model evaluation completed!")
    print(f"📊 Key Result: {test_accuracy*100:.1f}% accuracy on {len(X_test)} test queries")
    
    return test_accuracy, model, scaler, label_encoder

if __name__ == "__main__":
    evaluate_lightgbm_performance()