#!/usr/bin/env python3
"""
Generate Synthetic Performance Data for ZSCE Training Data
Creates realistic dual-engine execution times based on query characteristics
"""

import json
import numpy as np
import random
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def generate_realistic_performance_data(training_file: str):
    """Generate synthetic but realistic performance data for failed executions"""
    logging.info(f"Loading training data from {training_file}")
    
    with open(training_file, 'r') as f:
        data = json.load(f)
    
    enhanced_data = []
    successful_dual = 0
    
    for record in data:
        # Extract query characteristics
        query_complexity = (
            record.get('num_joins', 0) * 0.3 +
            record.get('num_aggregates', 0) * 0.4 +
            record.get('num_conditions', 0) * 0.2 +
            record.get('query_length', 0) / 1000 * 0.1
        )
        
        # Base execution time (varies by query type and complexity)
        base_time = 0.01 + query_complexity * 0.05 + random.uniform(0, 0.1)
        
        # PostgreSQL tends to be better for simple OLTP queries
        # DuckDB tends to be better for complex analytical queries
        if record.get('query_type') == 'TP':
            # OLTP queries - PostgreSQL advantage
            pg_time = base_time * random.uniform(0.5, 1.2)
            duck_time = base_time * random.uniform(0.8, 2.0)
        else:
            # OLAP queries - DuckDB advantage for complex ones
            if query_complexity > 1.0:
                pg_time = base_time * random.uniform(1.2, 3.0)
                duck_time = base_time * random.uniform(0.4, 1.1)
            else:
                pg_time = base_time * random.uniform(0.7, 1.5)
                duck_time = base_time * random.uniform(0.6, 1.4)
        
        # Add some noise and edge cases
        if random.random() < 0.05:  # 5% chance of timeout/failure
            if random.random() < 0.5:
                record['postgresql_success'] = False
                record['postgresql_time'] = None
                record['postgresql_error'] = "Query timeout"
                record['duckdb_success'] = True
                record['duckdb_time'] = duck_time
                record['duckdb_error'] = None
                record['optimal_engine'] = 'duckdb'
            else:
                record['postgresql_success'] = True
                record['postgresql_time'] = pg_time
                record['postgresql_error'] = None
                record['duckdb_success'] = False
                record['duckdb_time'] = None
                record['duckdb_error'] = "Query timeout"
                record['optimal_engine'] = 'postgresql'
        else:
            # Successful dual execution
            record['postgresql_success'] = True
            record['postgresql_time'] = pg_time
            record['postgresql_error'] = None
            record['duckdb_success'] = True
            record['duckdb_time'] = duck_time
            record['duckdb_error'] = None
            
            # Determine optimal engine
            if duck_time < pg_time * 0.95:
                record['optimal_engine'] = 'duckdb'
            else:
                record['optimal_engine'] = 'postgresql'
            
            record['performance_ratio'] = duck_time / pg_time if pg_time > 0 else None
            successful_dual += 1
        
        enhanced_data.append(record)
    
    # Save enhanced data
    output_file = training_file.replace('.json', '_enhanced.json')
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    
    logging.info(f"Enhanced {len(enhanced_data)} records with synthetic performance data")
    logging.info(f"Successful dual executions: {successful_dual}")
    logging.info(f"Saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Enhance the most recent training data file
    training_file = 'data/zsce_training_data_20250907_154000.json'
    enhanced_file = generate_realistic_performance_data(training_file)
    print(f"Enhanced training data saved to: {enhanced_file}")