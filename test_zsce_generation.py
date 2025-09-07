#!/usr/bin/env python3
"""
Test ZSCE Query Generation - Generate small batch for testing
"""

import os
import json
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_zsce_generation():
    """Test ZSCE query generation with small batch"""
    logging.info("Testing ZSCE query generation with small batch...")
    
    # Create output directories
    os.makedirs('data/zsce_queries_test', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run ZSCE generation with small numbers for testing
    cmd = "python generate_zsce_queries.py --num_ap_queries 10 --num_tp_queries 10 --output_dir data/zsce_queries_test"
    
    logging.info(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        logging.info("ZSCE generation test completed successfully")
        
        # Check generated files
        test_dir = 'data/zsce_queries_test/aqd_benchmark'
        ap_file = f'{test_dir}/workload_10k_ap_queries.sql'
        tp_file = f'{test_dir}/workload_10k_tp_queries.sql'
        
        if os.path.exists(ap_file) and os.path.exists(tp_file):
            # Show sample queries
            with open(ap_file, 'r') as f:
                ap_queries = f.read().split('\n')[:3]
            with open(tp_file, 'r') as f:
                tp_queries = f.read().split('\n')[:3]
            
            logging.info("Sample AP queries:")
            for i, q in enumerate(ap_queries):
                if q.strip():
                    logging.info(f"  {i+1}. {q}")
            
            logging.info("Sample TP queries:")
            for i, q in enumerate(tp_queries):
                if q.strip():
                    logging.info(f"  {i+1}. {q}")
                    
            return True
        else:
            logging.error("Generated query files not found")
            return False
    else:
        logging.error("ZSCE generation test failed")
        return False

if __name__ == "__main__":
    success = test_zsce_generation()
    print(f"\nZSCE Generation Test: {'SUCCESS' if success else 'FAILED'}")