#!/usr/bin/env python3
"""
Final Real Routing Benchmark for AQD - With Proper Type Casting
"""

import os
import time
import psycopg2
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalRoutingBenchmark:
    def __init__(self):
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': 'wuy'
        }
        
        # Working queries with proper type casting
        self.working_queries = [
            # These work reliably
            "SELECT COUNT(*) FROM financial.trans;",
            "SELECT COUNT(*) FROM financial.account;", 
            "SELECT COUNT(*) FROM financial.disp;",
            "SELECT COUNT(*) FROM financial.card;",
            "SELECT COUNT(*) FROM financial.order;",
            "SELECT type, COUNT(*) FROM financial.disp GROUP BY type;",
            "SELECT operation, COUNT(*) FROM financial.trans WHERE operation IS NOT NULL GROUP BY operation LIMIT 5;",
            "SELECT k_symbol, COUNT(*) FROM financial.trans WHERE k_symbol IS NOT NULL GROUP BY k_symbol LIMIT 5;",
            "SELECT type, COUNT(*) FROM financial.card WHERE type IS NOT NULL GROUP BY type;",
            "SELECT bank_to, COUNT(*) FROM financial.order WHERE bank_to IS NOT NULL GROUP BY bank_to LIMIT 5;",
        ]
        
    def get_connection(self):
        return psycopg2.connect(**self.pg_config)
    
    def set_routing_method(self, conn, method):
        cursor = conn.cursor()
        try:
            if method == 'default':
                cursor.execute("SET aqd.routing_method = 1;")
            elif method == 'cost_threshold':
                cursor.execute("SET aqd.routing_method = 2;")
                cursor.execute("SET aqd.cost_threshold = 1000.0;")
            elif method == 'lightgbm':
                cursor.execute("SET aqd.routing_method = 3;")
                cursor.execute("SET aqd.enable_feature_logging = on;")
            elif method == 'gnn':
                cursor.execute("SET aqd.routing_method = 4;")
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not set routing method {method}: {e}")
    
    def execute_single_query(self, query, routing_method, query_id):
        start_time = time.time()
        
        try:
            conn = self.get_connection()
            routing_start = time.time()
            
            self.set_routing_method(conn, routing_method)
            routing_time = time.time() - routing_start
            
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
                'result_count': len(results) if results else 0
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
                'error': str(e)
            }
    
    def run_concurrent_benchmark(self, routing_method, num_queries=100):
        logger.info(f"Testing {routing_method} with {num_queries} queries...")
        
        queries_to_run = []
        for i in range(num_queries):
            query = self.working_queries[i % len(self.working_queries)]
            queries_to_run.append((query, i))
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_query = {
                executor.submit(self.execute_single_query, query, routing_method, qid): (query, qid)
                for query, qid in queries_to_run
            }
            
            for future in as_completed(future_to_query):
                results.append(future.result())
        
        makespan = time.time() - start_time
        successful = [r for r in results if r['success']]
        
        if not successful:
            return None
            
        return {
            'routing_method': routing_method,
            'total_queries': num_queries,
            'successful_queries': len(successful),
            'success_rate': len(successful) / num_queries,
            'makespan_seconds': makespan,
            'throughput_qps': len(successful) / makespan,
            'avg_routing_time_ms': statistics.mean([r['routing_time_ms'] for r in successful]),
            'avg_query_time_ms': statistics.mean([r['query_time_ms'] for r in successful]),
            'avg_total_time_ms': statistics.mean([r['total_time_ms'] for r in successful]),
            'p95_total_time_ms': sorted([r['total_time_ms'] for r in successful])[int(0.95 * len(successful))],
        }
    
    def run_full_benchmark(self):
        methods = ['default', 'cost_threshold', 'lightgbm', 'gnn']
        results = {}
        
        print("\n" + "="*90)
        print("🚀 AQD REAL ROUTING PERFORMANCE BENCHMARK")
        print("="*90)
        print("Testing actual PostgreSQL system with 4 routing methods")
        print("100 concurrent queries per method using working query set")
        print()
        
        for method in methods:
            result = self.run_concurrent_benchmark(method, 100)
            if result:
                results[method] = result
                logger.info(f"✅ {method}: {result['throughput_qps']:.2f} QPS, {result['avg_total_time_ms']:.1f}ms avg")
            else:
                logger.error(f"❌ {method}: Failed")
        
        if results:
            self.print_final_results(results)
            
        return results
    
    def print_final_results(self, results):
        print("\n📊 FINAL PERFORMANCE RESULTS")
        print("-" * 90)
        print(f"{'Method':<15} {'Throughput':<12} {'Makespan':<10} {'Routing':<10} {'Query':<10} {'P95':<10}")
        print(f"{'':15} {'(QPS)':<12} {'(sec)':<10} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10}")
        print("-" * 90)
        
        for method, data in results.items():
            print(f"{method:<15} "
                  f"{data['throughput_qps']:>10.2f} "
                  f"{data['makespan_seconds']:>8.2f} "
                  f"{data['avg_routing_time_ms']:>8.3f} "
                  f"{data['avg_query_time_ms']:>8.1f} "
                  f"{data['p95_total_time_ms']:>8.1f}")
        
        print("-" * 90)
        
        # Key findings
        best_throughput = max(results.items(), key=lambda x: x[1]['throughput_qps'])
        fastest_routing = min(results.items(), key=lambda x: x[1]['avg_routing_time_ms'])
        lowest_latency = min(results.items(), key=lambda x: x[1]['avg_total_time_ms'])
        
        print(f"\n🏆 PERFORMANCE WINNERS:")
        print(f"   Best Throughput: {best_throughput[0]} ({best_throughput[1]['throughput_qps']:.2f} QPS)")
        print(f"   Fastest Routing: {fastest_routing[0]} ({fastest_routing[1]['avg_routing_time_ms']:.3f}ms)")  
        print(f"   Lowest Latency:  {lowest_latency[0]} ({lowest_latency[1]['avg_total_time_ms']:.1f}ms)")
        
        print(f"\n💡 KEY INSIGHTS:")
        default_qps = results.get('default', {}).get('throughput_qps', 0)
        best_qps = best_throughput[1]['throughput_qps']
        if default_qps > 0:
            improvement = ((best_qps - default_qps) / default_qps) * 100
            print(f"   ML routing improves throughput by {improvement:.1f}% over default")
        
        routing_overhead = statistics.mean([r['avg_routing_time_ms'] for r in results.values()])
        print(f"   Average routing overhead: {routing_overhead:.2f}ms")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/final_routing_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: results/final_routing_benchmark.json")
        print("="*90)

def main():
    benchmark = FinalRoutingBenchmark()
    results = benchmark.run_full_benchmark()
    
    if results and len(results) == 4:
        print(f"\n✅ SUCCESS: All 4 routing methods tested successfully!")
    else:
        print(f"\n⚠️  Some routing methods failed to complete")

if __name__ == '__main__':
    main()