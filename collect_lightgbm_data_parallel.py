#!/usr/bin/env python3
"""
LightGBM Data Collection Script - Parallel Version
Collects features and execution times from both PostgreSQL and DuckDB engines
using parallel execution across different datasets for much faster collection
"""

import os
import sys
import csv
import json
import time
import psycopg2
import numpy as np
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import multiprocessing
import glob
import random

# Feature names - must match kernel
FEATURE_NAMES = [
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type"
]

# Global lock for file writing
write_lock = Lock()

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(processName)s/%(threadName)s] - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_connection(dbname, user="wuy", host="localhost", port=5432):
    """Create a new database connection"""
    return psycopg2.connect(
        dbname=dbname,
        user=user,
        host=host,
        port=port
    )

def warmup_connection(conn):
    """Warm up a connection with dummy queries"""
    with conn.cursor() as cur:
        for _ in range(3):
            cur.execute("SELECT 1")
            cur.fetchone()
    conn.commit()

def extract_features_manually(query):
    """Fallback manual feature extraction from query text"""
    query_lower = query.lower()
    
    features = {
        'num_tables': query_lower.count(' from ') + query_lower.count(' join '),
        'num_joins': query_lower.count(' join '),
        'query_depth': min(query_lower.count('('), 5),
        'complexity_score': len(query) // 100,
        'has_aggregates': 1 if any(agg in query_lower for agg in ['sum(', 'avg(', 'count(', 'max(', 'min(']) else 0,
        'has_group_by': 1 if ' group by ' in query_lower else 0,
        'has_order_by': 1 if ' order by ' in query_lower else 0,
        'has_limit': 1 if ' limit ' in query_lower else 0,
        'has_distinct': 1 if ' distinct ' in query_lower else 0,
        'has_window_functions': 1 if ' over(' in query_lower or ' over (' in query_lower else 0,
        'has_outer_joins': 1 if any(join in query_lower for join in [' left join', ' right join', ' full join']) else 0,
        'estimated_join_complexity': query_lower.count(' join ') * 2,
        'has_subqueries': 1 if 'select' in query_lower[10:] else 0,
        'has_correlated_subqueries': 0,
        'has_large_tables': 0,
        'all_tables_small': 0,
        'has_complex_expressions': 1 if 'case when' in query_lower else 0,
        'has_user_functions': 0,
        'has_text_operations': 1 if any(op in query_lower for op in ['like ', 'ilike ', '|| ']) else 0,
        'has_numeric_heavy_ops': 1 if any(op in query_lower for op in ['sum(', 'avg(', 'stddev(', 'variance(']) else 0,
        'num_aggregate_funcs': sum(1 for agg in ['sum(', 'avg(', 'count(', 'max(', 'min('] if agg in query_lower),
        'analytical_pattern': 1 if ('group by' in query_lower and 'sum(' in query_lower) else 0,
        'transactional_pattern': 1 if 'where' in query_lower and 'limit 1' in query_lower else 0,
        'etl_pattern': 0,
        'command_type': 0
    }
    
    return features

def collect_features(conn, query):
    """Collect pre-optimization features using observability GUCs"""
    features = {}
    
    with conn.cursor() as cur:
        try:
            # First, do an EXPLAIN to trigger feature extraction without execution
            cur.execute(f"EXPLAIN {query}")
            
            # Try to read features from observability GUC if available
            try:
                cur.execute("SHOW lgbm.last_features_json")
                features_json = cur.fetchone()[0]
                
                if features_json and features_json != '{}':
                    features = json.loads(features_json)
                else:
                    features = extract_features_manually(query)
            except:
                features = extract_features_manually(query)
                
        except Exception as e:
            # If EXPLAIN fails, use manual extraction
            features = extract_features_manually(query)
    
    conn.rollback()
    return features

def measure_execution_time(conn, query, engine, timeout_ms=60000):
    """Measure query execution time on specified engine"""
    with conn.cursor() as cur:
        try:
            # Configure engine
            if engine == 'postgres':
                cur.execute("SET LOCAL duckdb.force_execution = false")
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                except:
                    pass
            elif engine == 'duckdb':
                cur.execute("SET LOCAL duckdb.force_execution = true")
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                except:
                    pass
            
            # Set timeout
            cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
            
            # Measure execution
            start_time = time.perf_counter()
            cur.execute(query)
            
            # Fetch all results to ensure complete execution
            results = cur.fetchall()
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            # Check which engine actually ran (if GUC available)
            actual_engine = engine
            try:
                cur.execute("SHOW lgbm.last_routed_engine")
                actual_engine = cur.fetchone()[0]
            except:
                pass
            
            return execution_time_ms, actual_engine
            
        except Exception as e:
            # Query failed
            return -1, 'unknown'
        finally:
            conn.rollback()

def process_query_batch(batch_info):
    """Process a batch of queries for a single dataset"""
    dataset_name, db_name, queries, user, host, port, output_dir = batch_info
    
    logger = setup_logging()
    logger.info(f"Processing {len(queries)} queries for dataset {dataset_name}")
    
    results = []
    
    # Create connections for this worker
    try:
        conn_pg = create_connection(db_name, user, host, port)
        conn_duck = create_connection(db_name, user, host, port)
        conn_pg.autocommit = False
        conn_duck.autocommit = False
        
        # Warm up connections
        warmup_connection(conn_pg)
        warmup_connection(conn_duck)
        
        for i, (query_id, query) in enumerate(queries):
            if i % 10 == 0:
                logger.info(f"[{dataset_name}] Progress: {i}/{len(queries)}")
            
            try:
                # Collect features
                features = collect_features(conn_pg, query)
                
                # Measure on PostgreSQL
                pg_time, pg_engine = measure_execution_time(conn_pg, query, 'postgres')
                
                # Measure on DuckDB
                duck_time, duck_engine = measure_execution_time(conn_duck, query, 'duckdb')
                
                # Determine optimal engine
                if pg_time > 0 and duck_time > 0:
                    optimal_engine = 'duckdb' if duck_time < pg_time else 'postgres'
                else:
                    optimal_engine = 'unknown'
                
                # Create result row
                row = [query_id, len(query)]
                for feature_name in FEATURE_NAMES:
                    row.append(features.get(feature_name, 0))
                row.extend([pg_time, duck_time, optimal_engine])
                
                results.append(row)
                
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {e}")
                continue
        
        # Close connections
        conn_pg.close()
        conn_duck.close()
        
        logger.info(f"Completed {dataset_name}: collected {len(results)} samples")
        return dataset_name, results
        
    except Exception as e:
        logger.error(f"Failed to process dataset {dataset_name}: {e}")
        return dataset_name, []

def write_results_to_file(output_file, results, mode='a'):
    """Write results to CSV file with thread-safe locking"""
    with write_lock:
        file_exists = os.path.exists(output_file)
        
        with open(output_file, mode, newline='') as f:
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists or mode == 'w':
                header = ['query_id', 'query_length'] + FEATURE_NAMES + \
                        ['pg_time_ms', 'duck_time_ms', 'optimal_engine']
                writer.writerow(header)
            
            # Write data rows
            writer.writerows(results)

def collect_from_benchmark_dir_parallel(benchmark_dir="benchmark_queries", 
                                       output_dir="lightgbm_training_data",
                                       max_queries_per_dataset=100,
                                       num_workers=None,
                                       user="wuy",
                                       host="localhost", 
                                       port=5432):
    """Collect data from benchmark queries directory using parallel processing"""
    
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    logger.info(f"Starting parallel collection with {num_workers} workers")
    
    # Find all dataset directories
    dataset_dirs = [d for d in os.listdir(benchmark_dir) 
                   if os.path.isdir(os.path.join(benchmark_dir, d))]
    
    # Skip non-dataset directories
    skip_dirs = {'__pycache__', '.git', '.svn'}
    dataset_dirs = [d for d in dataset_dirs if d not in skip_dirs]
    
    logger.info(f"Found {len(dataset_dirs)} datasets to process")
    
    # Prepare batches for parallel processing
    batches = []
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(benchmark_dir, dataset_name)
        
        # Load queries
        all_queries = []
        
        # Try different query file patterns
        for pattern in ['workload_ap_queries.sql', 'workload_tp_queries.sql', 
                       'ap_queries.sql', 'tp_queries.sql', 'queries.sql']:
            query_file = os.path.join(dataset_path, pattern)
            if os.path.exists(query_file):
                with open(query_file, 'r') as f:
                    queries = [line.strip() for line in f 
                             if line.strip() and not line.startswith('--')]
                    all_queries.extend([(f"{dataset_name}_{i}", q) 
                                      for i, q in enumerate(queries)])
        
        if not all_queries:
            logger.warning(f"No queries found for dataset {dataset_name}")
            continue
        
        # Limit queries if specified
        if max_queries_per_dataset and len(all_queries) > max_queries_per_dataset:
            all_queries = random.sample(all_queries, max_queries_per_dataset)
        
        # Determine database name
        db_name = dataset_name
        
        # Special handling for certain datasets
        if dataset_name in ['tpch_sf1', 'tpch']:
            db_name = 'tpch_sf1'
        elif dataset_name in ['tpcds_sf1', 'tpcds']:
            db_name = 'tpcds_sf1'
        
        # Create batch info
        batch_info = (dataset_name, db_name, all_queries, user, host, port, output_dir)
        batches.append(batch_info)
    
    if not batches:
        logger.error("No valid datasets found to process")
        return
    
    # Process batches in parallel using ProcessPoolExecutor
    all_results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_dataset = {executor.submit(process_query_batch, batch): batch[0] 
                            for batch in batches}
        
        # Process completed tasks
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                dataset_name, results = future.result()
                all_results[dataset_name] = results
                
                # Write results immediately to avoid memory buildup
                if results:
                    output_file = os.path.join(output_dir, f"training_data_{dataset_name}.csv")
                    write_results_to_file(output_file, results, mode='w')
                    logger.info(f"Saved {len(results)} samples for {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Dataset {dataset_name} failed: {e}")
    
    # Summary
    total_samples = sum(len(results) for results in all_results.values())
    logger.info(f"\nCollection complete!")
    logger.info(f"Total datasets processed: {len(all_results)}")
    logger.info(f"Total samples collected: {total_samples}")
    
    # Combine all results into single file
    combined_file = os.path.join(output_dir, "training_data_combined.csv")
    all_rows = []
    for dataset_name, results in all_results.items():
        all_rows.extend(results)
    
    if all_rows:
        write_results_to_file(combined_file, all_rows, mode='w')
        logger.info(f"Combined data saved to {combined_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect LightGBM training data in parallel')
    parser.add_argument('--benchmark-dir', default='benchmark_queries',
                       help='Directory containing benchmark queries')
    parser.add_argument('--output-dir', default='lightgbm_training_data',
                       help='Output directory for training data')
    parser.add_argument('--max-queries', type=int, default=10000,
                       help='Maximum queries per dataset')
    parser.add_argument('--num-workers', type=int,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--user', default='wuy',
                       help='Database user')
    parser.add_argument('--host', default='localhost',
                       help='Database host')
    parser.add_argument('--port', type=int, default=5432,
                       help='Database port')
    
    args = parser.parse_args()
    
    collect_from_benchmark_dir_parallel(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        max_queries_per_dataset=args.max_queries,
        num_workers=args.num_workers,
        user=args.user,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()