#!/usr/bin/env python3
"""Quick test of routing methods on small subset"""

import json
import subprocess
import numpy as np

# Load some test data
with open("data/execution_data/imdb_small_unified_training_data.json", 'r') as f:
    data = json.load(f)

# Test on 10 examples
test_examples = data[:10]

print("Testing routing methods on 10 examples:\n")

for i, ex in enumerate(test_examples):
    pg_time = ex.get('postgres_time', -1)
    duck_time = ex.get('duckdb_time', -1)
    
    if pg_time <= 0 or duck_time <= 0:
        continue
        
    optimal = "DuckDB" if duck_time < pg_time else "PostgreSQL"
    
    # Test GNN
    plan_json = ex.get('postgres_plan_json', {})
    # Extract the first plan if it's a list
    if isinstance(plan_json, list) and len(plan_json) > 0:
        plan_json = plan_json[0]
    
    try:
        result = subprocess.run(
            ['./build/gnn_predict_simple', 'models/rginn_routing_model.txt'],
            input=json.dumps(plan_json),
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            gnn_pred = float(result.stdout.strip())
            gnn_choice = "DuckDB" if gnn_pred > 0 else "PostgreSQL"
        else:
            gnn_choice = "Error"
            gnn_pred = 0
    except Exception as e:
        gnn_choice = f"Error: {e}"
        gnn_pred = 0
        
    print(f"Example {i}:")
    print(f"  PG time: {pg_time:.4f}s, Duck time: {duck_time:.4f}s")
    print(f"  Optimal: {optimal}")
    print(f"  GNN prediction: {gnn_pred:.4f} -> {gnn_choice}")
    print(f"  Correct: {gnn_choice == optimal}")
    print()