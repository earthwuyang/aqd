#!/usr/bin/env python3
"""
Create a small test LightGBM model for debugging
"""
import lightgbm as lgb
import numpy as np
import json

# Create synthetic training data with 25 features
np.random.seed(42)
n_samples = 1000
n_features = 25  # Must match PREOPT_NUM_FEATURES

X = np.random.random((n_samples, n_features))
# Simple rule: route to DuckDB if first feature > 0.5
y = (X[:, 0] > 0.5).astype(int)

# Train a small model
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 10,  # Very small tree
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 1
}

train_data = lgb.Dataset(X, label=y)
model = lgb.train(params, train_data, num_boost_round=10)  # Only 10 trees

# Save model
model.save_model('lightgbm_models/test_model.txt')
print("Small test model saved to lightgbm_models/test_model.txt")

# Check model size
import os
size = os.path.getsize('lightgbm_models/test_model.txt')
print(f"Model size: {size} bytes ({size/1024:.2f} KB)")
