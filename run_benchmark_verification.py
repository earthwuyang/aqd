#!/usr/bin/env python3
"""
Benchmark Harness Verification Script
Compares routing methods and verifies LightGBM performance
"""

import os
import sys
import time
import json
import csv
import psycopg2
import numpy as np
import argparse
import logging
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict

class BenchmarkHarness:
    def __init__(self, dbname="test", user="postgres", host="localhost", port=5432):
        self.dbname = dbname
        self.user = user
        self.host = host
        self.port = port
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Persistent connections for each method
        self.connections = {}
        
        # Results storage
        self.results = defaultdict(list)
        
    @contextmanager
    def get_connection(self, method):
        """Get or create a persistent connection for a method"""
        if method not in self.connections:
            self.connections[method] = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                host=self.host,
                port=self.port
            )
            self.connections[method].autocommit = False
            self.logger.info(f"Created connection for method: {method}")
            
            # Warm up connection
            self.warmup_connection(self.connections[method])
        
        yield self.connections[method]
    
    def warmup_connection(self, conn):
        """Warm up a connection with dummy queries"""
        with conn.cursor() as cur:
            for _ in range(5):
                cur.execute("SELECT 1")
                cur.fetchone()
        conn.commit()
    
    def configure_method(self, conn, method, model_path=None, threshold=None):
        """Configure routing method for a connection"""
        with conn.cursor() as cur:
            if method == 'default':
                # No routing, use PostgreSQL default
                cur.execute("SET LOCAL duckdb.force_execution = false")
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                except:
                    pass  # LightGBM might not be configured
                
            elif method == 'always_postgres':
                # Force PostgreSQL
                cur.execute("SET LOCAL duckdb.force_execution = false")
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                except:
                    pass
                
            elif method == 'always_duckdb':
                # Force DuckDB
                cur.execute("SET LOCAL duckdb.force_execution = true")
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                except:
                    pass
                
            elif method == 'lightgbm':
                # Use LightGBM routing
                try:
                    cur.execute("SET LOCAL lightgbm.enable_plan_logging = false")  # OFF for performance
                except:
                    pass
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = true")
                    if model_path:
                        cur.execute(f"SET lightgbm.model_path = '{model_path}'")
                    if threshold is not None:
                        cur.execute(f"SET lightgbm.routing_threshold = {threshold}")
                except Exception as e:
                    self.logger.warning(f"Could not configure LightGBM: {e}")
                    # Fall back to default
                    cur.execute("SET LOCAL duckdb.force_execution = false")
                    
            elif method == 'cost':
                # Cost-based routing (if implemented)
                try:
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                except:
                    pass
                # Add cost-based configuration if available
                
            else:
                raise ValueError(f"Unknown method: {method}")
    
    def execute_query(self, conn, query, method, timeout_ms=60000):
        """Execute a query with specified method and measure performance"""
        result = {
            'method': method,
            'execution_time_ms': -1,
            'routed_engine': 'unknown',
            'routing_overhead_us': 0,
            'success': False,
            'error': None
        }
        
        with conn.cursor() as cur:
            try:
                # Configure method with model path and threshold if lightgbm
                if method == 'lightgbm':
                    self.configure_method(conn, method, 
                                        getattr(self, 'lgbm_model_path', None),
                                        getattr(self, 'lgbm_threshold', None))
                else:
                    self.configure_method(conn, method)
                
                # Set timeout
                cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                
                # Execute and measure
                start_time = time.perf_counter()
                cur.execute(query)
                
                # Fetch all results to ensure complete execution
                results = cur.fetchall()
                end_time = time.perf_counter()
                
                result['execution_time_ms'] = (end_time - start_time) * 1000
                result['success'] = True
                
                # Get routing information if available
                try:
                    cur.execute("SHOW lgbm.last_routed_engine")
                    result['routed_engine'] = cur.fetchone()[0]
                except:
                    pass
                
                try:
                    cur.execute("SHOW lgbm.last_decision_us")
                    result['routing_overhead_us'] = float(cur.fetchone()[0])
                except:
                    pass
                
                # Verify with EXPLAIN for LightGBM method
                if method == 'lightgbm' and result['success']:
                    try:
                        cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
                        plan = cur.fetchone()[0]
                        if 'DuckDBScan' in str(plan):
                            if result['routed_engine'] != 'duckdb':
                                self.logger.warning(f"EXPLAIN shows DuckDB but GUC shows {result['routed_engine']}")
                    except:
                        pass
                
            except Exception as e:
                result['error'] = str(e)[:200]
                self.logger.debug(f"Query failed with {method}: {result['error']}")
        
        conn.rollback()  # Don't keep results
        return result
    
    def run_comparison(self, query, methods=['default', 'lightgbm'], query_id=None):
        """Compare query performance across different methods"""
        query_results = {
            'query_id': query_id or hash(query),
            'query_length': len(query),
            'results': {}
        }
        
        # Interleave methods to reduce cache bias
        for round_num in range(2):  # Run each method twice
            for method in methods:
                with self.get_connection(method) as conn:
                    result = self.execute_query(conn, query, method)
                    
                    # Store best result
                    if method not in query_results['results'] or \
                       (result['success'] and result['execution_time_ms'] < 
                        query_results['results'][method]['execution_time_ms']):
                        query_results['results'][method] = result
                
                # Small delay between methods
                time.sleep(0.1)
        
        return query_results
    
    def analyze_results(self, all_results):
        """Analyze benchmark results and print summary"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Aggregate by method
        method_stats = defaultdict(lambda: {
            'total_time': 0,
            'successful': 0,
            'failed': 0,
            'routing_overhead': [],
            'engine_distribution': defaultdict(int)
        })
        
        for query_result in all_results:
            for method, result in query_result['results'].items():
                stats = method_stats[method]
                
                if result['success']:
                    stats['successful'] += 1
                    stats['total_time'] += result['execution_time_ms']
                    stats['engine_distribution'][result['routed_engine']] += 1
                    
                    if result['routing_overhead_us'] > 0:
                        stats['routing_overhead'].append(result['routing_overhead_us'])
                else:
                    stats['failed'] += 1
        
        # Print results
        methods = sorted(method_stats.keys())
        
        print(f"\n{'Method':<20} {'Success':<10} {'Failed':<10} {'Total Time (s)':<15} {'Avg Time (ms)':<15}")
        print("-" * 80)
        
        for method in methods:
            stats = method_stats[method]
            total = stats['successful'] + stats['failed']
            avg_time = stats['total_time'] / stats['successful'] if stats['successful'] > 0 else 0
            
            print(f"{method:<20} {stats['successful']:<10} {stats['failed']:<10} "
                  f"{stats['total_time']/1000:<15.2f} {avg_time:<15.2f}")
        
        # Compare to baseline
        if 'default' in method_stats and 'lightgbm' in method_stats:
            default_time = method_stats['default']['total_time']
            lgbm_time = method_stats['lightgbm']['total_time']
            
            if default_time > 0:
                improvement = (default_time - lgbm_time) / default_time * 100
                print(f"\nLightGBM vs Default: {improvement:+.1f}% "
                      f"({'faster' if improvement > 0 else 'slower'})")
        
        # Engine distribution for LightGBM
        if 'lightgbm' in method_stats:
            print(f"\nLightGBM Engine Distribution:")
            for engine, count in method_stats['lightgbm']['engine_distribution'].items():
                if count > 0:
                    print(f"  {engine}: {count}")
            
            # Routing overhead
            if method_stats['lightgbm']['routing_overhead']:
                overhead = method_stats['lightgbm']['routing_overhead']
                print(f"\nLightGBM Routing Overhead:")
                print(f"  Mean: {np.mean(overhead):.1f} μs")
                print(f"  Median: {np.median(overhead):.1f} μs")
                print(f"  Max: {np.max(overhead):.1f} μs")
        
        # Find queries where LightGBM made wrong decisions
        if 'lightgbm' in methods and ('always_postgres' in methods or 'always_duckdb' in methods):
            print(f"\nRouting Decision Analysis:")
            correct_routing = 0
            total_routable = 0
            
            for query_result in all_results:
                lgbm_result = query_result['results'].get('lightgbm', {})
                
                if lgbm_result.get('success'):
                    # Determine optimal engine
                    pg_time = float('inf')
                    duck_time = float('inf')
                    
                    if 'always_postgres' in query_result['results']:
                        if query_result['results']['always_postgres']['success']:
                            pg_time = query_result['results']['always_postgres']['execution_time_ms']
                    
                    if 'always_duckdb' in query_result['results']:
                        if query_result['results']['always_duckdb']['success']:
                            duck_time = query_result['results']['always_duckdb']['execution_time_ms']
                    
                    if pg_time != float('inf') and duck_time != float('inf'):
                        total_routable += 1
                        optimal = 'postgres' if pg_time < duck_time else 'duckdb'
                        
                        if lgbm_result['routed_engine'] == optimal:
                            correct_routing += 1
            
            if total_routable > 0:
                accuracy = correct_routing / total_routable * 100
                print(f"  Routing Accuracy: {correct_routing}/{total_routable} ({accuracy:.1f}%)")
        
        return method_stats
    
    def save_results(self, all_results, output_file="benchmark_results.json"):
        """Save detailed results to file"""
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"Detailed results saved to {output_file}")
    
    def close_all_connections(self):
        """Close all connections"""
        for conn in self.connections.values():
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Run benchmark verification for LightGBM routing')
    parser.add_argument('--query-file', required=True, help='File containing benchmark queries')
    parser.add_argument('--methods', default='default,lightgbm,always_postgres,always_duckdb',
                       help='Comma-separated list of methods to compare')
    parser.add_argument('--model-path', help='Path to LightGBM model file')
    parser.add_argument('--threshold', type=float, help='LightGBM routing threshold')
    parser.add_argument('--output', default='benchmark_results.json', help='Output file for results')
    parser.add_argument('--limit', type=int, help='Limit number of queries to process')
    parser.add_argument('--dbname', default='test', help='Database name')
    parser.add_argument('--user', default='wuy', help='Database user')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    
    args = parser.parse_args()
    
    # Read queries
    with open(args.query_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith('--')]
    
    if args.limit:
        queries = queries[:args.limit]
    
    print(f"Running benchmark with {len(queries)} queries")
    print(f"Methods: {args.methods}")
    
    # Create harness
    harness = BenchmarkHarness(
        dbname=args.dbname,
        user=args.user,
        host=args.host,
        port=args.port
    )
    
    # Store model path and threshold for LightGBM if provided
    harness.lgbm_model_path = args.model_path
    harness.lgbm_threshold = args.threshold
    
    # Run benchmark
    all_results = []
    methods = args.methods.split(',')
    
    for i, query in enumerate(queries):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(queries)}")
        
        try:
            result = harness.run_comparison(query, methods, query_id=i)
            all_results.append(result)
        except Exception as e:
            harness.logger.error(f"Failed to process query {i}: {e}")
    
    # Analyze and save results
    harness.analyze_results(all_results)
    harness.save_results(all_results, args.output)
    
    # Cleanup
    harness.close_all_connections()
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()