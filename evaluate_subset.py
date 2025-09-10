#!/usr/bin/env python3
"""Quick evaluation on subset of data"""

from evaluate_routing_methods import RoutingMethodEvaluator
import random

# Create evaluator
evaluator = RoutingMethodEvaluator()

# Load LightGBM model
evaluator.load_lightgbm_model()

# Use only 2000 test examples for quick evaluation
random.seed(42)
random.shuffle(evaluator.test_data)
evaluator.test_data = evaluator.test_data[:2000]

print(f"Quick evaluation on {len(evaluator.test_data)} examples\n")

# Evaluate all methods
results, ground_truth = evaluator.evaluate_all_methods()

# Print results
evaluator.print_results(results, ground_truth)

# Create plots
evaluator.create_comparison_plot(results, ground_truth)