#!/usr/bin/env python3
"""Quick test of evaluation script"""

from evaluate_routing_methods import RoutingMethodEvaluator
import time

# Create evaluator
evaluator = RoutingMethodEvaluator()

# Load only 10 examples for quick test
evaluator.test_data = evaluator.test_data[:10]

print(f"Testing on {len(evaluator.test_data)} examples\n")

# Load LightGBM model
evaluator.load_lightgbm_model()

# Test each routing method
for i, ex in enumerate(evaluator.test_data):
    print(f"\nExample {i}:")
    print(f"  Dataset: {ex['dataset']}")
    print(f"  PG time: {ex['postgres_time']:.4f}s")
    print(f"  Duck time: {ex['duckdb_time']:.4f}s")
    
    pg_time = ex['postgres_time']
    duck_time = ex['duckdb_time']
    optimal = evaluator.compute_optimal_routing(pg_time, duck_time)
    
    # Test default routing
    start = time.time()
    default = evaluator.apply_default_routing(ex)
    print(f"  Default: {default} (took {time.time()-start:.3f}s)")
    
    # Test GNN routing
    start = time.time()
    gnn = evaluator.apply_gnn_routing(ex.get('postgres_plan_json', {}))
    print(f"  GNN: {gnn} (took {time.time()-start:.3f}s)")
    
    print(f"  Optimal: {optimal}")
    
    if i >= 2:  # Just test a few
        break

print("\nTest completed successfully!")