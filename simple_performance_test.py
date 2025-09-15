#!/usr/bin/env python3
"""
Simple Performance Test Script
Tests a few basic TPC-H queries to compare PostgreSQL, DuckDB, and LightGBM routing
"""

import os
import sys
import time
import psycopg2
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# Simple TPC-H queries that should work on both engines
TEST_QUERIES = [
    {
        'id': 'q1_simple_count',
        'query': "SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'",
        'description': 'Simple count query on lineitem'
    },
    {
        'id': 'q2_simple_agg',
        'query': "SELECT l_returnflag, COUNT(*), SUM(l_quantity) FROM lineitem WHERE l_shipdate <= '1998-12-01' GROUP BY l_returnflag ORDER BY l_returnflag LIMIT 10",
        'description': 'Simple aggregation with GROUP BY'
    },
    {
        'id': 'q3_simple_join',
        'query': "SELECT o.o_orderkey, l.l_linenumber FROM orders o JOIN lineitem l ON o.o_orderkey = l.l_orderkey WHERE o.o_orderdate >= '1995-01-01' LIMIT 20",
        'description': 'Simple join between orders and lineitem'
    },
    {
        'id': 'q4_date_filter',
        'query': "SELECT o_orderstatus, COUNT(*) FROM orders WHERE o_orderdate >= '1993-01-01' AND o_orderdate < '1994-01-01' GROUP BY o_orderstatus",
        'description': 'Date filter with aggregation'
    },
    {
        'id': 'q5_customer_simple',
        'query': "SELECT c_custkey, c_name FROM customer WHERE c_acctbal > 8000 ORDER BY c_acctbal DESC LIMIT 15",
        'description': 'Simple customer query with ORDER BY'
    }
]

def run_performance_test(db_name='tpch_sf1', user='wuy', host='localhost', port=5432):
    logger = setup_logging()
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        host=host,
        port=port
    )
    conn.autocommit = False
    
    results = []
    
    try:
        # Load the LightGBM model first (using working 50-feature model)
        model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model_v3.txt'
        
        with conn.cursor() as cur:
            logger.info("Loading LightGBM model into PostgreSQL...")
            
            # Set the model path
            try:
                cur.execute(f"SET lightgbm.model_path = '{model_path}'")
                logger.info(f"Set model path to: {model_path}")
            except Exception as e:
                logger.warning(f"Could not set model path: {e}")
            
            # Check current settings
            try:
                cur.execute("SHOW lightgbm.enabled")
                enabled = cur.fetchone()[0]
                logger.info(f"LightGBM enabled: {enabled}")
                
                cur.execute("SHOW duckdb.force_execution") 
                force_duck = cur.fetchone()[0]
                logger.info(f"DuckDB force execution: {force_duck}")
            except Exception as e:
                logger.warning(f"Could not check settings: {e}")
        
        conn.commit()
        
        logger.info(f"Starting performance test on {len(TEST_QUERIES)} queries...")
        
        for i, query_data in enumerate(TEST_QUERIES):
            query_id = query_data['id']
            query = query_data['query']
            description = query_data['description']
            
            logger.info(f"\\nTesting Query {i+1}/{len(TEST_QUERIES)}: {query_id}")
            logger.info(f"Description: {description}")
            
            result = {
                'query_id': query_id,
                'description': description,
                'query': query
            }
            
            # Test 1: Always PostgreSQL
            logger.info("  Running with PostgreSQL...")
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL duckdb.force_execution = false")
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                    cur.execute("SET LOCAL statement_timeout = 30000")  # 30 second timeout
                    
                    start_time = time.perf_counter()
                    cur.execute(query)
                    rows = cur.fetchall()
                    end_time = time.perf_counter()
                    
                    pg_time = (end_time - start_time) * 1000
                    result['postgres_time_ms'] = pg_time
                    result['postgres_rows'] = len(rows)
                    logger.info(f"    PostgreSQL: {pg_time:.2f}ms, {len(rows)} rows")
                    
            except Exception as e:
                logger.warning(f"    PostgreSQL failed: {e}")
                result['postgres_time_ms'] = -1
                result['postgres_rows'] = -1
            
            conn.rollback()
            time.sleep(0.1)
            
            # Test 2: Always DuckDB
            logger.info("  Running with DuckDB...")
            try:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL duckdb.force_execution = true")
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                    cur.execute("SET LOCAL statement_timeout = 30000")  # 30 second timeout
                    
                    start_time = time.perf_counter()
                    cur.execute(query)
                    rows = cur.fetchall()
                    end_time = time.perf_counter()
                    
                    duck_time = (end_time - start_time) * 1000
                    result['duckdb_time_ms'] = duck_time
                    result['duckdb_rows'] = len(rows)
                    logger.info(f"    DuckDB: {duck_time:.2f}ms, {len(rows)} rows")
                    
            except Exception as e:
                logger.warning(f"    DuckDB failed: {e}")
                result['duckdb_time_ms'] = -1
                result['duckdb_rows'] = -1
            
            conn.rollback()
            time.sleep(0.1)
            
            # Test 3: LightGBM Routing
            logger.info("  Running with LightGBM routing...")
            try:
                with conn.cursor() as cur:
                    # Reset to allow LightGBM routing
                    cur.execute("SET LOCAL lightgbm.enabled = true")
                    cur.execute("SET LOCAL statement_timeout = 30000")  # 30 second timeout
                    
                    start_time = time.perf_counter()
                    cur.execute(query)
                    rows = cur.fetchall()
                    end_time = time.perf_counter()
                    
                    lgbm_time = (end_time - start_time) * 1000
                    result['lightgbm_time_ms'] = lgbm_time
                    result['lightgbm_rows'] = len(rows)
                    
                    # Try to get the routing decision
                    routing_decision = 'unknown'
                    try:
                        cur.execute("SHOW lightgbm.last_decision")
                        decision = cur.fetchone()
                        routing_decision = decision[0] if decision else 'unknown'
                    except:
                        pass
                    
                    result['lightgbm_decision'] = routing_decision
                    logger.info(f"    LightGBM: {lgbm_time:.2f}ms, {len(rows)} rows, routed to: {routing_decision}")
                    
            except Exception as e:
                logger.warning(f"    LightGBM routing failed: {e}")
                result['lightgbm_time_ms'] = -1
                result['lightgbm_rows'] = -1
                result['lightgbm_decision'] = 'error'
            
            conn.rollback()
            time.sleep(0.1)
            
            # Determine best performance
            valid_times = []
            if result['postgres_time_ms'] > 0:
                valid_times.append(('postgres', result['postgres_time_ms']))
            if result['duckdb_time_ms'] > 0:
                valid_times.append(('duckdb', result['duckdb_time_ms']))
            if result['lightgbm_time_ms'] > 0:
                valid_times.append(('lightgbm', result['lightgbm_time_ms']))
            
            if valid_times:
                best_engine, best_time = min(valid_times, key=lambda x: x[1])
                result['best_engine'] = best_engine
                result['best_time_ms'] = best_time
                logger.info(f"    BEST: {best_engine} ({best_time:.2f}ms)")
            else:
                result['best_engine'] = 'none'
                result['best_time_ms'] = -1
            
            results.append(result)
    
    finally:
        conn.close()
    
    return results

def analyze_and_print_results(results):
    logger = setup_logging()
    
    logger.info("\\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON RESULTS")
    logger.info("="*80)
    
    valid_results = [r for r in results if r['postgres_time_ms'] > 0 and r['duckdb_time_ms'] > 0 and r['lightgbm_time_ms'] > 0]
    
    if not valid_results:
        logger.error("No valid results found!")
        return
    
    total_pg_time = sum(r['postgres_time_ms'] for r in valid_results)
    total_duck_time = sum(r['duckdb_time_ms'] for r in valid_results)  
    total_lgbm_time = sum(r['lightgbm_time_ms'] for r in valid_results)
    
    logger.info(f"Valid queries: {len(valid_results)}/{len(results)}")
    logger.info(f"")
    logger.info(f"TOTAL EXECUTION TIMES:")
    logger.info(f"  Always PostgreSQL: {total_pg_time:.2f}ms")
    logger.info(f"  Always DuckDB:     {total_duck_time:.2f}ms")
    logger.info(f"  LightGBM Routing:  {total_lgbm_time:.2f}ms")
    logger.info(f"")
    
    # Performance vs always PostgreSQL
    pg_vs_lgbm_speedup = total_pg_time / total_lgbm_time if total_lgbm_time > 0 else 1
    logger.info(f"LightGBM vs Always PostgreSQL: {pg_vs_lgbm_speedup:.2f}x speedup")
    
    # Performance vs always DuckDB  
    duck_vs_lgbm_speedup = total_duck_time / total_lgbm_time if total_lgbm_time > 0 else 1
    logger.info(f"LightGBM vs Always DuckDB:     {duck_vs_lgbm_speedup:.2f}x speedup")
    
    logger.info(f"")
    logger.info("ROUTING DECISIONS:")
    postgres_routes = sum(1 for r in valid_results if 'postgres' in str(r.get('lightgbm_decision', '')).lower())
    duckdb_routes = sum(1 for r in valid_results if 'duckdb' in str(r.get('lightgbm_decision', '')).lower())
    unknown_routes = len(valid_results) - postgres_routes - duckdb_routes
    
    logger.info(f"  Routed to PostgreSQL: {postgres_routes}")
    logger.info(f"  Routed to DuckDB:     {duckdb_routes}")
    logger.info(f"  Unknown routing:      {unknown_routes}")
    
    logger.info(f"")
    logger.info("PER-QUERY BREAKDOWN:")
    for r in results:
        logger.info(f"  {r['query_id']}: PG={r['postgres_time_ms']:.1f}ms, Duck={r['duckdb_time_ms']:.1f}ms, LightGBM={r['lightgbm_time_ms']:.1f}ms (routed to: {r.get('lightgbm_decision', 'unknown')})")
    
    logger.info("="*80)

def main():
    logger = setup_logging()
    logger.info("Starting Simple Performance Comparison Test")
    logger.info("Comparing: Always PostgreSQL vs Always DuckDB vs LightGBM Routing")
    
    # Run the performance test
    results = run_performance_test()
    
    # Analyze and print results
    analyze_and_print_results(results)
    
    logger.info("\\nPerformance test completed!")

if __name__ == "__main__":
    main()