#!/usr/bin/env python3
"""
Real performance benchmark comparing all routing methods
"""

import psycopg2
import time
import statistics
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Test queries that work on the financial database
TEST_QUERIES = [
    # Simple queries (should use PostgreSQL)
    "SELECT COUNT(*) FROM account WHERE account_id < 100",
    "SELECT * FROM district WHERE district_id = 1",
    "SELECT client_id, birth_number FROM client LIMIT 10",
    
    # Medium complexity (cost threshold dependent)
    "SELECT d.district_id, COUNT(c.client_id) FROM district d JOIN client c ON d.district_id = c.district_id GROUP BY d.district_id",
    "SELECT account_id, SUM(amount) FROM loan GROUP BY account_id HAVING SUM(amount) > 100000",
    
    # Complex analytical queries (should use DuckDB)
    """SELECT 
        d.A2 as district_name,
        COUNT(DISTINCT c.client_id) as num_clients,
        COUNT(DISTINCT a.account_id) as num_accounts,
        AVG(l.amount) as avg_loan_amount
    FROM district d
    LEFT JOIN client c ON d.district_id = c.district_id
    LEFT JOIN disp di ON c.client_id = di.client_id
    LEFT JOIN account a ON di.account_id = a.account_id
    LEFT JOIN loan l ON a.account_id = l.account_id
    GROUP BY d.A2
    ORDER BY num_clients DESC""",
    
    """WITH monthly_stats AS (
        SELECT 
            DATE_TRUNC('month', date) as month,
            COUNT(*) as num_loans,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM loan
        GROUP BY DATE_TRUNC('month', date)
    )
    SELECT * FROM monthly_stats
    ORDER BY month""",
]

def run_query_with_method(conn_params, query, method_code, method_name):
    """Execute a query with a specific routing method"""
    start = time.time()
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Set routing method
            cur.execute(f"SET aqd.routing_method = {method_code};")
            
            # Disable feature logging for fair comparison
            cur.execute("SET aqd.enable_feature_logging = off;")
            
            # Method-specific settings
            if method_name == 'cost_threshold':
                cur.execute("SET aqd.cost_threshold = 1000.0;")
            elif method_name == 'lightgbm':
                model_path = str(Path(__file__).parent / 'models' / 'lightgbm_model.txt')
                cur.execute(f"SET aqd.lightgbm_model_path = '{model_path}';")
            elif method_name == 'gnn':
                model_path = str(Path(__file__).parent / 'models' / 'rginn_routing_model.txt')
                cur.execute(f"SET aqd.gnn_model_path = '{model_path}';")
            
            # Execute query
            query_start = time.time()
            cur.execute(query)
            results = cur.fetchall()
            query_time = (time.time() - query_start) * 1000
            
            # Get routing info
            cur.execute("SHOW aqd.last_routed_engine;")
            engine = cur.fetchone()[0]
            
            cur.execute("SHOW aqd.last_decision_us;")
            routing_us = int(cur.fetchone()[0])
        
        conn.close()
        total_time = (time.time() - start) * 1000
        
        return {
            'success': True,
            'method': method_name,
            'engine': engine,
            'routing_us': routing_us,
            'query_ms': query_time,
            'total_ms': total_time,
            'result_count': len(results)
        }
    except Exception as e:
        return {
            'success': False,
            'method': method_name,
            'error': str(e),
            'total_ms': (time.time() - start) * 1000
        }

def run_benchmark():
    """Run comprehensive benchmark"""
    conn_params = {
        'host': 'localhost',
        'port': 5432,
        'user': 'wuy',
        'database': 'financial'
    }
    
    methods = {
        'default': 0,
        'cost_threshold': 1,
        'lightgbm': 2,
        'gnn': 3
    }
    
    print("\n" + "="*80)
    print("REAL PERFORMANCE EVALUATION - SERVER-SIDE ROUTING")
    print("="*80)
    print(f"Testing {len(TEST_QUERIES)} queries with {len(methods)} routing methods")
    print(f"Database: financial")
    print()
    
    # Create tasks - each query with each method
    tasks = []
    for query_idx, query in enumerate(TEST_QUERIES):
        for method_name, method_code in methods.items():
            tasks.append((query, query_idx, method_name, method_code))
    
    # Shuffle for cache fairness
    random.shuffle(tasks)
    
    # Run with concurrency
    results_by_method = {m: [] for m in methods.keys()}
    
    print("Running benchmark with interleaved methods...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for query, idx, method_name, method_code in tasks:
            future = executor.submit(run_query_with_method, conn_params, query, method_code, method_name)
            futures.append((future, method_name, idx))
        
        completed = 0
        for future, method_name, query_idx in futures:
            result = future.result()
            result['query_idx'] = query_idx
            results_by_method[method_name].append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  Completed {completed}/{len(tasks)} tasks")
    
    print("\n" + "-"*80)
    print("RESULTS BY ROUTING METHOD")
    print("-"*80)
    
    for method_name in methods.keys():
        results = results_by_method[method_name]
        successful = [r for r in results if r['success']]
        
        if not successful:
            print(f"\n{method_name.upper()}: All queries failed")
            continue
        
        query_times = [r['query_ms'] for r in successful]
        total_times = [r['total_ms'] for r in successful]
        routing_times = [r['routing_us'] for r in successful if 'routing_us' in r]
        
        # Count engine selection
        engines = [r.get('engine', 'unknown') for r in successful]
        pg_count = engines.count('postgres')
        duck_count = engines.count('duckdb')
        
        print(f"\n{method_name.upper()}:")
        print(f"  Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
        print(f"  Avg query time: {statistics.mean(query_times):.2f} ms")
        print(f"  P95 query time: {sorted(query_times)[int(0.95*len(query_times))]:.2f} ms" if len(query_times) > 1 else f"  Query time: {query_times[0]:.2f} ms")
        print(f"  Avg total time: {statistics.mean(total_times):.2f} ms")
        
        if routing_times:
            print(f"  Avg routing overhead: {statistics.mean(routing_times):.1f} µs")
        
        print(f"  Engine selection: PostgreSQL {pg_count}/{len(successful)} ({100*pg_count/len(successful):.0f}%), DuckDB {duck_count}/{len(successful)} ({100*duck_count/len(successful):.0f}%)")
    
    # Compare performance
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    
    baseline_method = 'default'
    baseline_results = [r for r in results_by_method[baseline_method] if r['success']]
    
    if baseline_results:
        baseline_avg = statistics.mean([r['total_ms'] for r in baseline_results])
        print(f"Baseline ({baseline_method}): {baseline_avg:.2f} ms avg total time")
        
        for method_name in methods.keys():
            if method_name == baseline_method:
                continue
            
            results = [r for r in results_by_method[method_name] if r['success']]
            if results:
                avg_time = statistics.mean([r['total_ms'] for r in results])
                improvement = ((baseline_avg - avg_time) / baseline_avg) * 100
                print(f"{method_name:15s}: {avg_time:.2f} ms ({improvement:+.1f}% vs baseline)")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("✓ All routing decisions happen server-side in the PostgreSQL kernel")
    print("✓ No client-side ML overhead in measurements")
    print("✓ Fair comparison with feature logging disabled for all methods")
    print("✓ Methods interleaved to avoid cache bias")
    print("✓ Actual routing overhead tracked via kernel GUCs")

if __name__ == "__main__":
    run_benchmark()