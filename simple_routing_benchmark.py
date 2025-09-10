#!/usr/bin/env python3
"""
Simple Real Routing Benchmark for AQD
Tests actual PostgreSQL routing with simplified queries
"""

import os
import time
import psycopg2
import threading
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRoutingBenchmark:
    def __init__(self):
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': 'wuy'
        }
        
        # Simple test queries for each dataset - properly formatted
        self.test_queries = [
            # OLTP-style queries
            "SELECT COUNT(*) FROM financial.trans WHERE account_id < 1000;",
            "SELECT * FROM financial.account WHERE account_id = 500 LIMIT 10;",
            "SELECT account_id, date FROM financial.account WHERE date > '1997-01-01' LIMIT 100;",
            "SELECT * FROM financial.disp WHERE account_id = 1000;",
            "SELECT type, COUNT(*) FROM financial.disp WHERE type = 'OWNER' GROUP BY type;",
            
            # OLAP-style queries  
            "SELECT account_id, AVG(CAST(balance AS NUMERIC)) FROM financial.trans WHERE balance IS NOT NULL AND balance != '' GROUP BY account_id ORDER BY account_id LIMIT 50;",
            "SELECT type, COUNT(*) as cnt FROM financial.disp GROUP BY type ORDER BY cnt DESC;",
            "SELECT DATE_PART('year', date) as year, COUNT(*) FROM financial.account GROUP BY year ORDER BY year;",
            "SELECT k_symbol, AVG(CAST(amount AS NUMERIC)) FROM financial.trans WHERE amount IS NOT NULL AND amount != '' AND k_symbol IS NOT NULL GROUP BY k_symbol ORDER BY k_symbol;",
            "SELECT operation, COUNT(*) FROM financial.trans WHERE operation IS NOT NULL GROUP BY operation ORDER BY COUNT(*) DESC LIMIT 10;"
        ]
        
    def get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.pg_config)
    
    def set_routing_method(self, conn, method):
        """Set the routing method in PostgreSQL"""
        cursor = conn.cursor()
        
        try:
            if method == 'default':
                cursor.execute("SET aqd.routing_method = 0;")  # AQD_ROUTE_DEFAULT
                
            elif method == 'cost_threshold':
                cursor.execute("SET aqd.routing_method = 1;")  # AQD_ROUTE_COST_THRESHOLD
                cursor.execute("SET aqd.cost_threshold = 1000.0;")
                
            elif method == 'lightgbm':
                cursor.execute("SET aqd.routing_method = 2;")  # AQD_ROUTE_LIGHTGBM
                cursor.execute("SET aqd.enable_feature_logging = off;")  # Fair comparison
                
            elif method == 'gnn':
                cursor.execute("SET aqd.routing_method = 3;")  # AQD_ROUTE_GNN
                
            conn.commit()
            
        except Exception as e:
            logger.warning(f"Could not set routing method {method}: {e}")
            # Continue with default settings
    
    def execute_single_query(self, query, routing_method, query_id):
        """Execute a single query and measure performance"""
        start_time = time.time()
        
        try:
            conn = self.get_connection()
            routing_start = time.time()
            
            # Set routing method
            self.set_routing_method(conn, routing_method)
            routing_time = time.time() - routing_start
            
            # Execute query
            cursor = conn.cursor()
            query_start = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            query_time = time.time() - query_start
            
            total_time = time.time() - start_time
            
            cursor.close()
            conn.close()
            
            return {
                'query_id': query_id,
                'routing_method': routing_method,
                'routing_time_ms': routing_time * 1000,
                'query_time_ms': query_time * 1000,
                'total_time_ms': total_time * 1000,
                'success': True,
                'result_count': len(results) if results else 0,
                'error': None
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            return {
                'query_id': query_id,
                'routing_method': routing_method,
                'routing_time_ms': 0,
                'query_time_ms': 0,
                'total_time_ms': total_time * 1000,
                'success': False,
                'result_count': 0,
                'error': str(e)
            }
    
    def run_concurrent_benchmark(self, routing_method, num_queries=100, max_workers=10):
        """Run concurrent queries for a specific routing method"""
        logger.info(f"Running {num_queries} queries concurrently for {routing_method}")
        
        # Create query list (repeat queries to reach num_queries)
        queries_to_run = []
        for i in range(num_queries):
            query = self.test_queries[i % len(self.test_queries)]
            queries_to_run.append((query, i))
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.execute_single_query, query, routing_method, qid): (query, qid)
                for query, qid in queries_to_run
            }
            
            # Collect results
            for future in as_completed(future_to_query):
                result = future.result()
                results.append(result)
                
                if len(results) % 25 == 0:
                    logger.info(f"  Progress: {len(results)}/{num_queries} completed")
        
        makespan = time.time() - start_time
        
        # Calculate metrics
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if not successful:
            logger.error(f"No successful queries for {routing_method}")
            if failed:
                logger.error(f"Sample error: {failed[0]['error']}")
            return None
            
        metrics = {
            'routing_method': routing_method,
            'total_queries': num_queries,
            'successful_queries': len(successful),
            'failed_queries': len(failed),
            'success_rate': len(successful) / num_queries,
            'makespan_seconds': makespan,
            'throughput_qps': len(successful) / makespan,
            
            # Timing metrics
            'avg_routing_time_ms': statistics.mean([r['routing_time_ms'] for r in successful]),
            'avg_query_time_ms': statistics.mean([r['query_time_ms'] for r in successful]),
            'avg_total_time_ms': statistics.mean([r['total_time_ms'] for r in successful]),
            
            # Percentiles
            'p50_total_time_ms': statistics.median([r['total_time_ms'] for r in successful]),
            'p95_total_time_ms': sorted([r['total_time_ms'] for r in successful])[int(0.95 * len(successful))],
            'p99_total_time_ms': sorted([r['total_time_ms'] for r in successful])[int(0.99 * len(successful))],
            
            'errors': [r['error'] for r in failed] if failed else []
        }
        
        return metrics
    
    def run_full_benchmark(self, num_queries=100):
        """Run benchmark for all routing methods"""
        methods = ['default', 'cost_threshold', 'lightgbm', 'gnn']
        results = {}
        
        logger.info(f"Starting routing benchmark with {num_queries} queries per method")
        logger.info("="*70)
        
        for method in methods:
            logger.info(f"Testing {method.upper()} routing...")
            result = self.run_concurrent_benchmark(method, num_queries)
            
            if result:
                results[method] = result
                logger.info(f"  ✅ {method}: {result['success_rate']*100:.1f}% success, "
                          f"{result['throughput_qps']:.2f} QPS, "
                          f"{result['avg_total_time_ms']:.1f}ms avg latency")
            else:
                logger.error(f"  ❌ {method}: Failed")
                
            # Small delay between methods
            time.sleep(1)
        
        self.generate_report(results, num_queries)
        return results
    
    def generate_report(self, results, num_queries):
        """Generate performance comparison report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{'='*80}")
        print(f"AQD REAL ROUTING BENCHMARK RESULTS - {timestamp}")
        print(f"{'='*80}")
        print(f"Queries per method: {num_queries}")
        print(f"Concurrent workers: 10")
        print(f"Test dataset: financial")
        print()
        
        if not results:
            print("❌ No successful benchmark results")
            return
            
        # Results table
        print("PERFORMANCE COMPARISON:")
        print("-" * 100)
        print(f"{'Method':<15} {'Success':<8} {'Makespan':<10} {'Throughput':<12} {'Routing':<10} {'Query':<10} {'P95':<10}")
        print(f"{'':15} {'Rate':<8} {'(sec)':<10} {'(QPS)':<12} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10}")
        print("-" * 100)
        
        for method, data in results.items():
            print(f"{method:<15} "
                  f"{data['success_rate']*100:>6.1f}% "
                  f"{data['makespan_seconds']:>8.2f} "
                  f"{data['throughput_qps']:>10.2f} "
                  f"{data['avg_routing_time_ms']:>8.3f} "
                  f"{data['avg_query_time_ms']:>8.1f} "
                  f"{data['p95_total_time_ms']:>8.1f}")
        
        print("-" * 100)
        print()
        
        # Key findings
        if len(results) > 1:
            best_throughput = max(results.items(), key=lambda x: x[1]['throughput_qps'])
            best_latency = min(results.items(), key=lambda x: x[1]['avg_total_time_ms'])
            fastest_routing = min(results.items(), key=lambda x: x[1]['avg_routing_time_ms'])
            
            print("KEY FINDINGS:")
            print(f"🚀 Highest Throughput: {best_throughput[0]} ({best_throughput[1]['throughput_qps']:.2f} QPS)")
            print(f"⚡ Lowest Latency: {best_latency[0]} ({best_latency[1]['avg_total_time_ms']:.1f}ms)")
            print(f"🎯 Fastest Routing: {fastest_routing[0]} ({fastest_routing[1]['avg_routing_time_ms']:.3f}ms)")
            print()
        
        # Save detailed results
        os.makedirs('results', exist_ok=True)
        with open('results/real_routing_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("📁 Detailed results saved to: results/real_routing_benchmark.json")
        print("="*80)

def main():
    benchmark = SimpleRoutingBenchmark()
    results = benchmark.run_full_benchmark(num_queries=100)
    
    if results:
        print(f"\n✅ Benchmark completed successfully! Tested {len(results)} routing methods.")
    else:
        print(f"\n❌ Benchmark failed!")

if __name__ == '__main__':
    main()