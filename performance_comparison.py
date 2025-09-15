#!/usr/bin/env python3
"""
Performance Comparison Script
Compares performance of Always PostgreSQL, Always DuckDB, and LightGBM Routing
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
import random

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_queries_from_training_data(training_data_file, max_queries=50):
    """Load queries that succeeded on both PostgreSQL and DuckDB from training data"""
    logger = setup_logging()
    queries = []
    
    if not os.path.exists(training_data_file):
        logger.error(f"Training data file not found: {training_data_file}")
        return queries
    
    with open(training_data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include queries that succeeded on both engines
            pg_time = float(row.get('pg_time_ms', -1))
            duck_time = float(row.get('duck_time_ms', -1))
            
            if pg_time > 0 and duck_time > 0:
                # Look up the actual query from the query_id
                query_id = row['query_id']
                # For now, we'll create a simple query based on the features
                # In a real implementation, you'd look up the actual query
                query = generate_test_query_from_features(row)
                
                queries.append({
                    'query_id': query_id,
                    'query': query,
                    'expected_pg_time': pg_time,
                    'expected_duck_time': duck_time,
                    'expected_optimal': row.get('optimal_engine', 'unknown')
                })
                
                if len(queries) >= max_queries:
                    break
    
    logger.info(f"Loaded {len(queries)} queries from training data")
    return queries

def generate_test_query_from_features(row):
    """Generate a simple test query based on the features in the training data"""
    # This is a simplified approach - in practice, you'd have the actual queries
    num_tables = int(row.get('num_tables', 1))
    has_aggregates = int(row.get('has_aggregates', 0))
    has_joins = int(row.get('num_joins', 0)) > 0
    has_group_by = int(row.get('has_group_by', 0))
    has_order_by = int(row.get('has_order_by', 0))
    has_limit = int(row.get('has_limit', 0))
    
    # Generate a basic query structure
    if num_tables <= 1:
        table = 'lineitem'  # Use TPC-H table
        query = f"SELECT "
        
        if has_aggregates:
            query += "COUNT(*), SUM(l_quantity)"
        else:
            query += "l_orderkey, l_linenumber"
        
        query += f" FROM {table}"
        
        # Add some basic conditions
        query += " WHERE l_shipdate >= '1994-01-01'"
        
        if has_group_by and has_aggregates:
            query += " GROUP BY l_returnflag"
        
        if has_order_by:
            if has_aggregates:
                query += " ORDER BY COUNT(*)"
            else:
                query += " ORDER BY l_orderkey"
        
        if has_limit:
            query += " LIMIT 100"
    
    else:
        # Multi-table query with joins
        query = """SELECT o.o_orderkey, l.l_linenumber 
                   FROM orders o 
                   JOIN lineitem l ON o.o_orderkey = l.l_orderkey 
                   WHERE o.o_orderdate >= '1994-01-01'"""
        
        if has_limit:
            query += " LIMIT 50"
    
    return query

def run_performance_test(queries, db_name='tpch_sf1', user='wuy', host='localhost', port=5432):
    """Run performance comparison test"""
    logger = setup_logging()
    
    results = []
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        host=host,
        port=port
    )
    conn.autocommit = False
    
    try:
        # Load the LightGBM model first
        model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model_v3_20250915_160133.txt'
        threshold_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model_v3_20250915_160133_threshold.txt'
        
        with conn.cursor() as cur:
            logger.info("Loading LightGBM model into PostgreSQL...")
            
            # Set the model path
            try:
                cur.execute(f"SET lightgbm.model_path = '{model_path}'")
                logger.info(f"Set model path to: {model_path}")
            except Exception as e:
                logger.warning(f"Could not set model path: {e}")
            
            # Enable LightGBM routing
            try:
                cur.execute("SET lightgbm.enabled = true")
                logger.info("Enabled LightGBM routing")
            except Exception as e:
                logger.warning(f"Could not enable LightGBM: {e}")
        
        conn.commit()
        
        logger.info(f"Starting performance comparison on {len(queries)} queries...")
        
        for i, query_data in enumerate(queries):
            query_id = query_data['query_id']
            query = query_data['query']
            
            logger.info(f"Testing query {i+1}/{len(queries)}: {query_id}")
            
            result = {
                'query_id': query_id,
                'query': query[:100] + '...' if len(query) > 100 else query,
                'expected_optimal': query_data['expected_optimal']
            }
            
            # Test 1: Always PostgreSQL
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL duckdb.force_execution = false")
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                    cur.execute("SET LOCAL statement_timeout = 30000")  # 30 second timeout
                    
                    start_time = time.perf_counter()
                    cur.execute(query)
                    rows = cur.fetchall()
                    end_time = time.perf_counter()
                    
                    result['postgres_time_ms'] = (end_time - start_time) * 1000
                    result['postgres_rows'] = len(rows)
                    
            except Exception as e:
                logger.warning(f"PostgreSQL failed for {query_id}: {e}")
                result['postgres_time_ms'] = -1
                result['postgres_rows'] = -1
            
            conn.rollback()
            time.sleep(0.1)
            
            # Test 2: Always DuckDB
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL duckdb.force_execution = true")
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                    cur.execute("SET LOCAL statement_timeout = 30000")  # 30 second timeout
                    
                    start_time = time.perf_counter()
                    cur.execute(query)
                    rows = cur.fetchall()
                    end_time = time.perf_counter()
                    
                    result['duckdb_time_ms'] = (end_time - start_time) * 1000
                    result['duckdb_rows'] = len(rows)
                    
            except Exception as e:
                logger.warning(f"DuckDB failed for {query_id}: {e}")
                result['duckdb_time_ms'] = -1
                result['duckdb_rows'] = -1
            
            conn.rollback()
            time.sleep(0.1)
            
            # Test 3: LightGBM Routing
            try:
                with conn.cursor() as cur:
                    # Reset to allow LightGBM routing
                    cur.execute("SET LOCAL lightgbm.enabled = true")
                    cur.execute("SET LOCAL statement_timeout = 30000")  # 30 second timeout
                    
                    start_time = time.perf_counter()
                    cur.execute(query)
                    rows = cur.fetchall()
                    end_time = time.perf_counter()
                    
                    result['lightgbm_time_ms'] = (end_time - start_time) * 1000
                    result['lightgbm_rows'] = len(rows)
                    
                    # Try to get the routing decision
                    try:
                        cur.execute("SHOW lightgbm.last_decision")
                        decision = cur.fetchone()
                        result['lightgbm_decision'] = decision[0] if decision else 'unknown'
                    except:
                        result['lightgbm_decision'] = 'unknown'
                    
            except Exception as e:
                logger.warning(f"LightGBM routing failed for {query_id}: {e}")
                result['lightgbm_time_ms'] = -1
                result['lightgbm_rows'] = -1
                result['lightgbm_decision'] = 'error'
            
            conn.rollback()
            time.sleep(0.1)
            
            # Determine the actual best engine
            valid_times = []
            if result['postgres_time_ms'] > 0:
                valid_times.append(('postgres', result['postgres_time_ms']))
            if result['duckdb_time_ms'] > 0:
                valid_times.append(('duckdb', result['duckdb_time_ms']))
            
            if valid_times:
                best_engine, best_time = min(valid_times, key=lambda x: x[1])
                result['actual_optimal'] = best_engine
                result['speedup_vs_worst'] = max(valid_times, key=lambda x: x[1])[1] / best_time
            else:
                result['actual_optimal'] = 'unknown'
                result['speedup_vs_worst'] = 1.0
            
            results.append(result)
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"Completed {i+1}/{len(queries)} queries")
        
    finally:
        conn.close()
    
    return results

def analyze_results(results):
    """Analyze and summarize the performance results"""
    logger = setup_logging()
    
    # Filter valid results
    valid_results = [r for r in results 
                    if r['postgres_time_ms'] > 0 and r['duckdb_time_ms'] > 0 and r['lightgbm_time_ms'] > 0]
    
    if not valid_results:
        logger.error("No valid results to analyze")
        return
    
    logger.info(f"Analyzing {len(valid_results)} valid results out of {len(results)} total")
    
    # Calculate statistics
    postgres_times = [r['postgres_time_ms'] for r in valid_results]
    duckdb_times = [r['duckdb_time_ms'] for r in valid_results]
    lightgbm_times = [r['lightgbm_time_ms'] for r in valid_results]
    
    # Performance comparison
    postgres_wins = sum(1 for r in valid_results if r['actual_optimal'] == 'postgres')
    duckdb_wins = sum(1 for r in valid_results if r['actual_optimal'] == 'duckdb')
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(f"Total valid queries tested: {len(valid_results)}")
    logger.info(f"PostgreSQL optimal: {postgres_wins} ({postgres_wins/len(valid_results)*100:.1f}%)")
    logger.info(f"DuckDB optimal: {duckdb_wins} ({duckdb_wins/len(valid_results)*100:.1f}%)")
    
    logger.info("\nAverage Execution Times:")
    logger.info(f"Always PostgreSQL: {np.mean(postgres_times):.2f}ms")
    logger.info(f"Always DuckDB: {np.mean(duckdb_times):.2f}ms")
    logger.info(f"LightGBM Routing: {np.mean(lightgbm_times):.2f}ms")
    
    logger.info("\nMedian Execution Times:")
    logger.info(f"Always PostgreSQL: {np.median(postgres_times):.2f}ms")
    logger.info(f"Always DuckDB: {np.median(duckdb_times):.2f}ms")
    logger.info(f"LightGBM Routing: {np.median(lightgbm_times):.2f}ms")
    
    # LightGBM routing accuracy
    routing_decisions = [r['lightgbm_decision'] for r in valid_results if r['lightgbm_decision'] != 'unknown']
    if routing_decisions:
        postgres_decisions = sum(1 for d in routing_decisions if 'postgres' in str(d).lower())
        duckdb_decisions = sum(1 for d in routing_decisions if 'duckdb' in str(d).lower())
        
        logger.info("\nLightGBM Routing Decisions:")
        logger.info(f"Routed to PostgreSQL: {postgres_decisions}")
        logger.info(f"Routed to DuckDB: {duckdb_decisions}")
        
        # Calculate routing accuracy
        correct_decisions = 0
        for r in valid_results:
            if r['lightgbm_decision'] != 'unknown':
                decision = str(r['lightgbm_decision']).lower()
                actual_optimal = r['actual_optimal']
                if ('postgres' in decision and actual_optimal == 'postgres') or \
                   ('duckdb' in decision and actual_optimal == 'duckdb'):
                    correct_decisions += 1
        
        if routing_decisions:
            accuracy = correct_decisions / len(routing_decisions) * 100
            logger.info(f"Routing Accuracy: {accuracy:.1f}% ({correct_decisions}/{len(routing_decisions)})")
    
    # Performance gain analysis
    lightgbm_vs_postgres = [r['lightgbm_time_ms'] / r['postgres_time_ms'] for r in valid_results]
    lightgbm_vs_duckdb = [r['lightgbm_time_ms'] / r['duckdb_time_ms'] for r in valid_results]
    
    logger.info(f"\nLightGBM vs Always PostgreSQL (avg speedup): {1/np.mean(lightgbm_vs_postgres):.2f}x")
    logger.info(f"LightGBM vs Always DuckDB (avg speedup): {1/np.mean(lightgbm_vs_duckdb):.2f}x")
    
    logger.info("="*60)

def save_results(results, output_file):
    """Save results to CSV file"""
    logger = setup_logging()
    
    fieldnames = ['query_id', 'query', 'expected_optimal', 'actual_optimal',
                 'postgres_time_ms', 'duckdb_time_ms', 'lightgbm_time_ms',
                 'postgres_rows', 'duckdb_rows', 'lightgbm_rows',
                 'lightgbm_decision', 'speedup_vs_worst']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Performance Comparison: PostgreSQL vs DuckDB vs LightGBM Routing')
    parser.add_argument('--training-data', default='lightgbm_training_data_parallel/training_data_tpch_sf1.csv',
                       help='Training data CSV file containing successful queries')
    parser.add_argument('--max-queries', type=int, default=30,
                       help='Maximum number of queries to test')
    parser.add_argument('--database', default='tpch_sf1',
                       help='Database name to connect to')
    parser.add_argument('--user', default='wuy',
                       help='Database user')
    parser.add_argument('--host', default='localhost',
                       help='Database host')
    parser.add_argument('--port', type=int, default=5432,
                       help='Database port')
    parser.add_argument('--output', default=f'performance_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                       help='Output CSV file for results')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting Performance Comparison")
    logger.info(f"Training data: {args.training_data}")
    logger.info(f"Max queries: {args.max_queries}")
    logger.info(f"Database: {args.database}")
    
    # Load queries from training data
    queries = load_queries_from_training_data(args.training_data, args.max_queries)
    
    if not queries:
        logger.error("No queries loaded. Exiting.")
        return
    
    # Run performance test
    results = run_performance_test(
        queries,
        db_name=args.database,
        user=args.user,
        host=args.host,
        port=args.port
    )
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    save_results(results, args.output)
    
    logger.info("Performance comparison completed successfully!")

if __name__ == "__main__":
    main()