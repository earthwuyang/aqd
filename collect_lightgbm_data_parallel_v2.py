#!/usr/bin/env python3
"""
LightGBM Data Collection Script - Parallel Version v2
With improved connection management and error recovery
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import glob
import random
from psycopg2 import pool

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

# Connection semaphore to limit concurrent connections
connection_semaphore = None

# Connection pools per database
connection_pools = {}
pool_lock = Lock()

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_connection_pool(dbname, user="wuy", host="localhost", port=5432, max_connections=5):
    """Get or create a connection pool for a database"""
    global connection_pools
    
    with pool_lock:
        if dbname not in connection_pools:
            try:
                connection_pools[dbname] = psycopg2.pool.ThreadedConnectionPool(
                    1,  # minconn
                    max_connections,  # maxconn
                    dbname=dbname,
                    user=user,
                    host=host,
                    port=port
                )
                logger = setup_logging()
                logger.info(f"Created connection pool for {dbname} with max {max_connections} connections")
            except Exception as e:
                logger = setup_logging()
                logger.error(f"Failed to create connection pool for {dbname}: {e}")
                return None
        
        return connection_pools[dbname]

def get_connection_from_pool(pool, timeout=30):
    """Get a connection from pool with retry logic"""
    logger = setup_logging()
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            conn = pool.getconn()
            if conn:
                conn.autocommit = False
                return conn
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to get connection (attempt {attempt + 1}): {e}")
                time.sleep(retry_delay)
            else:
                raise
    
    return None

def return_connection_to_pool(pool, conn):
    """Return connection to pool"""
    try:
        conn.rollback()  # Ensure clean state
        pool.putconn(conn)
    except:
        pass

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
    
    try:
        with conn.cursor() as cur:
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
    
    try:
        conn.rollback()
    except:
        pass
    
    return features

def measure_execution_time(conn, query, engine, timeout_ms=60000):
    """Measure query execution time on specified engine"""
    try:
        with conn.cursor() as cur:
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
        try:
            conn.rollback()
        except:
            pass

def process_single_query(query_info):
    """Process a single query with proper connection management"""
    query_id, query, db_name, user, host, port = query_info
    logger = setup_logging()
    
    # Get connection pool
    pool = get_connection_pool(db_name, user, host, port, max_connections=3)
    if not pool:
        return None
    
    result = None
    
    # Get connections from pool
    conn_features = None
    conn_pg = None
    conn_duck = None
    
    try:
        # Acquire semaphore before getting connections
        connection_semaphore.acquire()
        
        # Get connections with error handling
        conn_features = get_connection_from_pool(pool)
        conn_pg = get_connection_from_pool(pool)
        conn_duck = get_connection_from_pool(pool)
        
        if not all([conn_features, conn_pg, conn_duck]):
            logger.error(f"Failed to get connections for query {query_id}")
            return None
        
        # Collect features
        features = collect_features(conn_features, query)
        
        # Measure on PostgreSQL
        pg_time, pg_engine = measure_execution_time(conn_pg, query, 'postgres')
        
        # Small delay between engines
        time.sleep(0.1)
        
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
        
        result = row
        
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}")
    finally:
        # Return connections to pool
        if conn_features:
            return_connection_to_pool(pool, conn_features)
        if conn_pg:
            return_connection_to_pool(pool, conn_pg)
        if conn_duck:
            return_connection_to_pool(pool, conn_duck)
        
        # Release semaphore
        connection_semaphore.release()
    
    return result

def process_dataset_batch(dataset_info):
    """Process all queries for a dataset using thread pool"""
    dataset_name, queries, user, host, port, output_dir = dataset_info
    logger = setup_logging()
    logger.info(f"Processing {len(queries)} queries for dataset {dataset_name}")
    
    results = []
    
    # Use ThreadPoolExecutor for queries within a dataset
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Prepare query info for each query
        query_infos = []
        for query_id, query, db_name in queries:
            query_infos.append((query_id, query, db_name, user, host, port))
        
        # Submit all queries
        futures = [executor.submit(process_single_query, query_info) 
                  for query_info in query_infos]
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            if i % 50 == 0:
                logger.info(f"[{dataset_name}] Progress: {i}/{len(futures)}")
            
            try:
                result = future.result(timeout=120)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
    
    logger.info(f"Completed {dataset_name}: collected {len(results)} samples")
    return dataset_name, results

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
                                       num_workers=4,
                                       max_connections_per_db=10,
                                       user="wuy",
                                       host="localhost", 
                                       port=5432):
    """Collect data from benchmark queries directory using parallel processing"""
    
    global connection_semaphore
    connection_semaphore = Semaphore(max_connections_per_db)
    
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting parallel collection with {num_workers} dataset workers")
    
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
                    
                    # Determine database name
                    db_name = dataset_name
                    if dataset_name in ['tpch_sf1', 'tpch']:
                        db_name = 'tpch_sf1'
                    elif dataset_name in ['tpcds_sf1', 'tpcds']:
                        db_name = 'tpcds_sf1'
                    
                    all_queries.extend([(f"{dataset_name}_{pattern}_{i}", q, db_name) 
                                      for i, q in enumerate(queries)])
        
        if not all_queries:
            logger.warning(f"No queries found for dataset {dataset_name}")
            continue
        
        # Limit queries if specified
        if max_queries_per_dataset and len(all_queries) > max_queries_per_dataset:
            all_queries = random.sample(all_queries, max_queries_per_dataset)
        
        # Create batch info
        batch_info = (dataset_name, all_queries, user, host, port, output_dir)
        batches.append(batch_info)
    
    if not batches:
        logger.error("No valid datasets found to process")
        return
    
    # Process datasets sequentially, but queries within each dataset in parallel
    all_results = {}
    
    for batch in batches:
        dataset_name = batch[0]
        try:
            dataset_name, results = process_dataset_batch(batch)
            all_results[dataset_name] = results
            
            # Write results immediately
            if results:
                output_file = os.path.join(output_dir, f"training_data_{dataset_name}.csv")
                write_results_to_file(output_file, results, mode='w')
                logger.info(f"Saved {len(results)} samples for {dataset_name}")
                
        except Exception as e:
            logger.error(f"Dataset {dataset_name} failed: {e}")
        
        # Small delay between datasets
        time.sleep(1)
    
    # Close all connection pools
    with pool_lock:
        for db_name, pool in connection_pools.items():
            try:
                pool.closeall()
                logger.info(f"Closed connection pool for {db_name}")
            except:
                pass
    
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
    parser.add_argument('--max-queries', type=int, default=100,
                       help='Maximum queries per dataset')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel dataset workers')
    parser.add_argument('--max-connections', type=int, default=10,
                       help='Maximum connections per database')
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
        max_connections_per_db=args.max_connections,
        user=args.user,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()