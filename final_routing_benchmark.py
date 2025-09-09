#!/usr/bin/env python3
"""
Final Routing Benchmark using generated AP/TP workloads
- Loads queries from data/benchmark_queries/<dataset>/{AP,TP} files
- Runs concurrent benchmarks at scales: 100, 200, ..., 1000
"""

import os
import time
import psycopg2
import statistics
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalRoutingBenchmark:
    def __init__(self, dataset='imdb_small', query_dir=None):
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': dataset,
            'user': 'wuy'
        }
        self.dataset = dataset
        base_dir = Path(__file__).resolve().parent
        self.query_base_dir = Path(query_dir) if query_dir else (base_dir / 'data' / 'benchmark_queries' / dataset)
        
        # Load AP + TP queries
        self.generated_queries = self._load_generated_queries()

    def _load_sql_file(self, path: Path):
        if not path.exists():
            return []
        text = path.read_text(encoding='utf-8', errors='ignore')
        queries = [q.strip() for q in text.split(';') if q.strip()]
        return queries

    def _load_generated_queries(self):
        ap_path = self.query_base_dir / 'workload_10k_ap_queries.sql'
        tp_path = self.query_base_dir / 'workload_10k_tp_queries.sql'
        ap = self._load_sql_file(ap_path)
        tp = self._load_sql_file(tp_path)
        queries = ap + tp
        if not queries:
            logger.warning(f"No generated queries found under {self.query_base_dir}")
        else:
            random.shuffle(queries)
        return queries
        
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
                # Attempt to load a trained LightGBM model if available
                base = Path(__file__).resolve().parent
                candidates = [
                    base / 'models' / 'real_classification_model.txt',
                    base / 'models' / 'classification_model.txt',
                    base / 'models' / 'routing_lightgbm.txt',
                    base / 'models' / 'lightgbm_model.txt'
                ]
                for path in candidates:
                    if path.exists():
                        cursor.execute(f"SET aqd.lightgbm_model_path = '{str(path)}';")
                        break
            elif method == 'gnn':
                cursor.execute("SET aqd.routing_method = 4;")
                # Attempt to load a trained GNN model if available
                base = Path(__file__).resolve().parent
                gnn_candidates = [
                    base / 'models' / 'gnn_model.txt',
                ]
                for path in gnn_candidates:
                    if path.exists():
                        cursor.execute(f"SET aqd.gnn_model_path = '{str(path)}';")
                        break
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
        logger.info(f"Testing {routing_method} on dataset '{self.dataset}' with {num_queries} concurrent queries...")
        
        if not self.generated_queries:
            logger.error("No queries loaded. Please generate queries first.")
            return None

        # Select a slice of queries; cycle if needed
        queries_to_run = []
        for i in range(num_queries):
            query = self.generated_queries[i % len(self.generated_queries)]
            queries_to_run.append((query, i))
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_queries) as executor:
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
    
    def run_full_benchmark(self, conc_levels=None):
        methods = ['default', 'cost_threshold', 'lightgbm', 'gnn']
        results = {m: {} for m in methods}
        conc_levels = conc_levels or [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        print("\n" + "="*90)
        print("🚀 AQD REAL ROUTING PERFORMANCE BENCHMARK")
        print("="*90)
        print(f"Dataset: {self.dataset}")
        print("Testing actual PostgreSQL system with 4 routing methods")
        print("Concurrent scales: 100..1000 using generated AP+TP workloads")
        print()
        for nq in conc_levels:
            print(f"\n--- Concurrency: {nq} ---")
            for method in methods:
                result = self.run_concurrent_benchmark(method, nq)
                if result:
                    results[method][str(nq)] = result
                    logger.info(f"✅ {method}@{nq}: {result['throughput_qps']:.2f} QPS, {result['avg_total_time_ms']:.1f}ms avg")
                else:
                    logger.error(f"❌ {method}@{nq}: Failed")

        self.print_final_results(results)
        return results
    
    def print_final_results(self, results):
        print("\n📊 FINAL PERFORMANCE RESULTS (by concurrency)")
        print("-" * 90)
        # Aggregate summary per method across conc levels
        for method, per_level in results.items():
            if not per_level:
                continue
            print(f"\nMethod: {method}")
            print(f"{'Conc':<6} {'QPS':>10} {'Makespan(s)':>14} {'Avg(ms)':>10} {'P95(ms)':>10} {'Success%':>10}")
            for level in sorted(per_level.keys(), key=lambda x: int(x)):
                d = per_level[level]
                print(f"{level:<6} "
                      f"{d['throughput_qps']:>10.2f} "
                      f"{d['makespan_seconds']:>14.2f} "
                      f"{d['avg_total_time_ms']:>10.1f} "
                      f"{d['p95_total_time_ms']:>10.1f} "
                      f"{(d['success_rate']*100):>9.1f}%")

        # Save results
        os.makedirs('results', exist_ok=True)
        out_path = f'results/final_routing_benchmark_{self.dataset}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {out_path}")
        print("="*90)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Final routing benchmark on generated workloads')
    parser.add_argument('--dataset', type=str, default='imdb_small', help='Dataset/database name to target')
    parser.add_argument('--query_dir', type=str, default=None, help='Directory with generated queries for the dataset')
    parser.add_argument('--levels', type=str, default='100,200,300,400,500,600,700,800,900,1000', help='Comma-separated concurrency levels')
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(',') if x.strip()]
    benchmark = FinalRoutingBenchmark(dataset=args.dataset, query_dir=args.query_dir)
    results = benchmark.run_full_benchmark(conc_levels=levels)
    
    print("\n✅ Benchmark complete")

if __name__ == '__main__':
    main()
