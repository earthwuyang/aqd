#!/usr/bin/env python3
"""Test GNN routing with gnn_trainer_real --predict"""

import json
import subprocess

# Load some test data
with open("data/execution_data/imdb_small_unified_training_data.json", 'r') as f:
    data = json.load(f)

# Test on 5 examples
test_examples = data[:5]

print("Testing GNN routing with gnn_trainer_real --predict:\n")

for i, ex in enumerate(test_examples):
    pg_time = ex.get('postgres_time', -1)
    duck_time = ex.get('duckdb_time', -1)
    
    if pg_time <= 0 or duck_time <= 0:
        continue
        
    optimal = "DuckDB" if duck_time < pg_time else "PostgreSQL"
    
    # Test GNN with gnn_trainer_real --predict
    plan_json = ex.get('postgres_plan_json', {})
    try:
        result = subprocess.run(
            ['./build/gnn_trainer_real', '--predict', 'models/rginn_routing_model.txt'],
            input=json.dumps(plan_json),
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            gnn_pred = result.stdout.strip()
            gnn_choice = "DuckDB" if gnn_pred == "1" else "PostgreSQL"
        else:
            gnn_choice = "Error"
            gnn_pred = "?"
    except Exception as e:
        gnn_choice = f"Error: {e}"
        gnn_pred = "?"
        
    print(f"Example {i}:")
    print(f"  PG time: {pg_time:.4f}s, Duck time: {duck_time:.4f}s")
    print(f"  Optimal: {optimal}")
    print(f"  GNN prediction: {gnn_pred} -> {gnn_choice}")
    print(f"  Correct: {gnn_choice == optimal}")
    print()