#!/usr/bin/env python3
"""
Real Concurrent Query Execution Benchmark for AQD Routing Methods
Tests actual PostgreSQL system with all four routing methods:
1. Default (pg_duckdb heuristic)
2. Cost-threshold
3. LightGBM-based (kernel inference)
4. GNN-based (kernel inference)
"""

import os
import sys
import time
import json
import psycopg2
import threading
import queue
import statistics
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AQDRoutingBenchmark:
    def __init__(self):
        self.pg_host = os.environ.get('AQD_PG_HOST', 'localhost')
        self.pg_port = int(os.environ.get('AQD_PG_PORT', '5432'))
        self.pg_user = os.environ.get('AQD_PG_USER', 'wuy')
        
        # Load trained LightGBM model
        base = Path(__file__).resolve().parent
        self.model_path = str(base / 'models' / 'lightgbm_model.txt')
        # Client-side LightGBM inference is not used; kernel performs inference
        self.lgb_model = None
        self.feature_names = []
        
        # Query sources
        self.query_base_dir = str(base / 'data' / 'benchmark_queries')
        self.exec_data_dir = str(base / 'data' / 'execution_data')
        self.results_dir = str(base / 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Routing methods to test
        self.routing_methods = ['default', 'cost_threshold', 'lightgbm', 'gnn']
        
    def load_lightgbm_model(self):
        """Deprecated: kernel performs inference; keep for compatibility."""
        self.lgb_model = None
        self.feature_names = []
    
    def get_connection(self, database: str):
        """Get PostgreSQL connection to a dataset database"""
        return psycopg2.connect(host=self.pg_host, port=self.pg_port, user=self.pg_user, database=database)
    
    def set_routing_method(self, conn, method):
        """Set the routing method in PostgreSQL with proper model paths and logging controls"""
        cursor = conn.cursor()

        # Kernel mapping: 0=default, 1=cost, 2=lightgbm, 3=gnn
        code = {'default': 0, 'cost_threshold': 1, 'lightgbm': 2, 'gnn': 3}[method]
        cursor.execute(f"SET aqd.routing_method = {code};")

        # Do not enable feature or plan logging in benchmark runs
        try:
            cursor.execute("SET aqd.enable_feature_logging = off;")
            cursor.execute("SET aqd.enable_plan_logging = off;")
        except Exception:
            pass

        # LightGBM: enable feature extraction so in-kernel predictor gets features
        if method == 'lightgbm':
            # Model path for LightGBM inference in kernel (features are extracted internally)
            cursor.execute(f"SET aqd.lightgbm_model_path = '{self.model_path}';")
            # Provide explicit library path to avoid dlopen search failures
            lgb_lib_candidates = [
                str(Path(__file__).resolve().parent / 'install' / 'lib' / 'libaqd_lgbm.so'),
                str(Path(__file__).resolve().parent / 'lib' / 'libaqd_lgbm.so'),
            ]
            for p in lgb_lib_candidates:
                if os.path.exists(p):
                    cursor.execute(f"SET aqd.lightgbm_library_path = '{p}';")
                    break

        # GNN: enable plan JSON logging for this session
        if method == 'gnn':
            gnn_path = str(Path(__file__).resolve().parent / 'models' / 'rginn_routing_model.txt')
            if os.path.exists(gnn_path):
                cursor.execute(f"SET aqd.gnn_model_path = '{gnn_path}';")
            # Provide explicit GNN library path
            gnn_lib_candidates = [
                str(Path(__file__).resolve().parent / 'install' / 'lib' / 'libaqd_gnn.so'),
                str(Path(__file__).resolve().parent / 'lib' / 'libaqd_gnn.so'),
            ]
            for p in gnn_lib_candidates:
                if os.path.exists(p):
                    cursor.execute(f"SET aqd.gnn_library_path = '{p}';")
                    break

        # Fine-tune cost threshold used for the baseline method
        if method == 'cost_threshold':
            cursor.execute("SET aqd.cost_threshold = 1000.0;")

        conn.commit()
        logger.info(f"Set routing method to: {method}")
    
    def extract_query_features(self, query_text):
        """Extract features for LightGBM prediction"""
        features = {}
        
        # Basic query structure features
        features['query_length'] = len(query_text)
        features['has_join'] = 1 if 'JOIN' in query_text.upper() else 0
        features['has_group_by'] = 1 if 'GROUP BY' in query_text.upper() else 0
        features['has_order_by'] = 1 if 'ORDER BY' in query_text.upper() else 0
        features['has_where'] = 1 if 'WHERE' in query_text.upper() else 0
        features['has_aggregation'] = 1 if any(agg in query_text.upper() for agg in ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX']) else 0
        
        # Query type classification (simple heuristic)
        if features['has_aggregation'] or features['has_group_by']:
            features['is_ap_query'] = 1
            features['is_tp_query'] = 0
        else:
            features['is_ap_query'] = 0
            features['is_tp_query'] = 1
            
        # Placeholder for other features that would come from actual execution
        for fname in self.feature_names:
            if fname not in features:
                features[fname] = 0.0
                
        return features
    
    def predict_routing(self, query_text):
        """Deprecated - kernel performs routing"""
        # Routing is now done server-side, not client-side
        return 'postgres'
    
    def _make_worker(self, database: str, routing_method: str):
        """Create one persistent connection per worker for a given method and DB."""
        conn = self.get_connection(database)
        self.set_routing_method(conn, routing_method)
        cur = conn.cursor()
        # Warm-up to ensure router/model initialization is done once per session
        try:
            cur.execute("SELECT 1;")
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

        def run_one(query_text: str, query_id: int):
            t0 = time.time()
            try:
                cur.execute(query_text)
                rows = cur.fetchall()
                exec_ms = (time.time() - t0) * 1000.0
                # Fetch server-side routing telemetry if available (robust via current_setting)
                engine_code = None
                route_us = None
                try:
                    cur.execute("SELECT current_setting('aqd.last_decision_engine_code', true)")
                    v = cur.fetchone()
                    if v and v[0] is not None and v[0] != '':
                        engine_code = int(v[0])
                    cur.execute("SELECT current_setting('aqd.last_decision_latency_us', true)")
                    v = cur.fetchone()
                    if v and v[0] is not None and v[0] != '':
                        route_us = float(v[0])
                except Exception:
                    pass
                return {
                    'query_id': query_id,
                    'routing_method': routing_method,
                    'routing_time_ms': (route_us / 1000.0) if route_us is not None else 0.0,
                    'query_time_ms': exec_ms,
                    'total_time_ms': exec_ms,
                    'engine_code': engine_code,  # 0=PostgreSQL, 1=DuckDB
                    'success': True,
                    'error': None,
                    'result_count': len(rows) if rows else 0
                }
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                return {
                    'query_id': query_id,
                    'routing_method': routing_method,
                    'routing_time_ms': 0.0,
                    'query_time_ms': 0.0,
                    'total_time_ms': (time.time() - t0) * 1000.0,
                    'success': False,
                    'error': str(e),
                    'result_count': 0
                }
        return run_one, conn
    
    def load_test_queries(self, dataset, query_type, limit=100):
        """Load queries from static generated SQL files (legacy path)"""
        query_file = Path(self.query_base_dir) / dataset / f'workload_10k_{query_type}_queries.sql'
        
        if not query_file.exists():
            logger.error(f"Query file not found: {query_file}")
            return []
        
        queries = []
        with open(query_file, 'r') as f:
            content = f.read()
            # Split queries (assuming they're separated by semicolons)
            raw_queries = content.split(';')
            
            for i, query in enumerate(raw_queries[:limit]):
                query = query.strip()
                if query and not query.startswith('--'):
                    queries.append({
                        'id': i,
                        'text': query + ';',
                        'dataset': dataset,
                        'type': query_type
                    })
                    
        logger.info(f"Loaded {len(queries)} queries from {dataset}/{query_type}")
        return queries

    def load_executed_queries(self, dataset: str = None, limit: int = 200):
        """Load queries that have been executed successfully by both PostgreSQL and DuckDB.
        Reads unified training data under data/execution_data/*_unified_training_data.json.
        """
        queries = []
        exec_dir = Path(self.exec_data_dir)
        if not exec_dir.exists():
            logger.error(f"Execution data directory not found: {exec_dir}")
            return []
        files = []
        if dataset:
            files = [exec_dir / f"{dataset}_unified_training_data.json"]
        else:
            files = sorted(exec_dir.glob("*_unified_training_data.json"))

        for fpath in files:
            if not fpath.exists():
                continue
            try:
                data = json.load(open(fpath, 'r'))
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
                continue
            for i, rec in enumerate(data):
                # Select only queries that succeeded on both engines
                ok_pg = rec.get('executed_postgres', rec.get('postgres_time') is not None)
                ok_duck = rec.get('executed_duckdb', rec.get('duckdb_time') is not None)
                if not (ok_pg and ok_duck):
                    continue
                qtxt = rec.get('query_text')
                dset = rec.get('dataset')
                if not qtxt or not dset:
                    continue
                queries.append({'id': len(queries), 'text': qtxt.strip().rstrip(';') + ';', 'dataset': dset, 'type': rec.get('query_type', 'ap')})
                if len(queries) >= limit:
                    break
            if len(queries) >= limit:
                break
        logger.info(f"Loaded {len(queries)} executed queries for benchmarking")
        return queries
    
    def run_concurrent_benchmark(self, queries, routing_method, max_workers=10):
        """Run concurrent queries for a specific routing method using persistent per-thread sessions."""
        logger.info(f"Running concurrent benchmark for {routing_method} with {len(queries)} queries")
        results = []
        start_time = time.time()

        # Build workers and persistent connections (assign DBs round-robin by dataset)
        datasets = list({q['dataset'] for q in queries}) or ['postgres']
        worker_funcs = []
        conns = []
        for i in range(max_workers):
            db = datasets[i % len(datasets)]
            fn, cn = self._make_worker(db, routing_method)
            worker_funcs.append(fn)
            conns.append(cn)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, q in enumerate(queries):
                    run_one = worker_funcs[i % max_workers]
                    futures.append(executor.submit(run_one, q['text'], q['id']))
                for future in as_completed(futures):
                    results.append(future.result())
                    if len(results) % 25 == 0:
                        logger.info(f"Completed {len(results)}/{len(queries)} for {routing_method}")
        finally:
            # Cleanly close persistent connections
            for cn in conns:
                try:
                    cn.close()
                except Exception:
                    pass

        total_makespan = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            return None
            
        routing_times = [r['routing_time_ms'] for r in successful_results]
        query_times = [r['query_time_ms'] for r in successful_results]
        total_times = [r['total_time_ms'] for r in successful_results]
        
        metrics = {
            'routing_method': routing_method,
            'total_queries': len(queries),
            'successful_queries': len(successful_results),
            'success_rate': len(successful_results) / len(queries),
            'makespan_seconds': total_makespan,
            'throughput_qps': len(successful_results) / total_makespan,
            'avg_routing_time_ms': statistics.mean(routing_times),
            'avg_query_time_ms': statistics.mean(query_times),
            'avg_total_time_ms': statistics.mean(total_times),
            'p95_total_time_ms': np.percentile(total_times, 95),
            'p99_total_time_ms': np.percentile(total_times, 99),
            'detailed_results': results
        }
        
        logger.info(f"Benchmark completed for {routing_method}:")
        logger.info(f"  Makespan: {total_makespan:.2f}s")
        logger.info(f"  Throughput: {metrics['throughput_qps']:.2f} QPS")
        logger.info(f"  Success rate: {metrics['success_rate']*100:.1f}%")
        logger.info(f"  Avg routing time: {metrics['avg_routing_time_ms']:.3f}ms")
        logger.info(f"  Avg query time: {metrics['avg_query_time_ms']:.1f}ms")
        
        return metrics
    
    def run_full_benchmark(self, dataset='financial', query_type='ap', num_queries=100, use_executed=True):
        """Run full benchmark comparing all routing methods"""
        logger.info(f"Starting full routing benchmark")
        logger.info(f"Dataset: {dataset}, Query type: {query_type}, Queries: {num_queries}")
        
        # Load queries from executed training data (preferred), or fallback to static files
        if use_executed:
            queries = self.load_executed_queries(dataset if dataset else None, num_queries)
        else:
            queries = self.load_test_queries(dataset, query_type, num_queries)
        if not queries:
            logger.error("No queries loaded for benchmark")
            return
            
        # Run benchmark for each routing method
        all_results = {}
        
        for method in self.routing_methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing routing method: {method.upper()}")
            logger.info(f"{'='*50}")
            
            result = self.run_concurrent_benchmark(queries, method)
            if result:
                all_results[method] = result
                
                # Save individual results
                result_file = Path(self.results_dir) / f'benchmark_{method}_{dataset}_{query_type}.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Generate comparison report
        self.generate_benchmark_report(all_results, dataset, query_type)
        
        return all_results
    
    def generate_benchmark_report(self, results, dataset, query_type):
        """Generate comprehensive benchmark report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# AQD Real Routing Methods Benchmark Report
Generated: {timestamp}
Dataset: {dataset}
Query Type: {query_type}

## Performance Comparison

| Method | Success Rate | Makespan (s) | Throughput (QPS) | Avg Routing (ms) | Avg Query (ms) | P95 Total (ms) |
|--------|-------------|--------------|------------------|------------------|----------------|----------------|
"""
        
        for method, data in results.items():
            report += f"| {method} | {data['success_rate']*100:.1f}% | {data['makespan_seconds']:.2f} | {data['throughput_qps']:.2f} | {data['avg_routing_time_ms']:.3f} | {data['avg_query_time_ms']:.1f} | {data['p95_total_time_ms']:.1f} |\n"
        
        report += f"""
## Key Findings

"""
        
        # Find best performer in each category
        if results:
            best_throughput = max(results.items(), key=lambda x: x[1]['throughput_qps'])
            best_latency = min(results.items(), key=lambda x: x[1]['avg_total_time_ms'])
            best_routing = min(results.items(), key=lambda x: x[1]['avg_routing_time_ms'])
            
            report += f"- **Highest Throughput**: {best_throughput[0]} ({best_throughput[1]['throughput_qps']:.2f} QPS)\n"
            report += f"- **Lowest Latency**: {best_latency[0]} ({best_latency[1]['avg_total_time_ms']:.1f}ms avg)\n"
            report += f"- **Fastest Routing**: {best_routing[0]} ({best_routing[1]['avg_routing_time_ms']:.3f}ms avg)\n"
        
        report += f"""
## System Information

- PostgreSQL with AQD modifications
- Concurrent execution with 10 worker threads
- LightGBM model: {self.model_path}
- Test queries: {len(results[list(results.keys())[0]]['detailed_results']) if results else 0}

## Routing Method Details

### Default
- Uses pg_duckdb built-in heuristics
- No ML prediction overhead
- Baseline performance

### Cost Threshold  
- Routes based on PostgreSQL optimizer cost estimates
- Threshold: 1000.0 cost units
- Simple decision logic

### LightGBM
- ML-based routing using trained LightGBM model
- Features are extracted internally in the kernel; no client inference
- Predicts optimal execution engine in-kernel via libaqd_lgbm.so

### GNN
- Graph neural network operating on the optimizer plan graph
- In-kernel inference via libaqd_gnn.so; no client JSON logging required
- Leverages plan structure and cost annotations
"""
        
        # Save report
        report_file = Path(self.results_dir) / f'real_routing_benchmark_report_{dataset}_{query_type}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Benchmark report saved to: {report_file}")
        print(f"\n{report}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AQD Routing Methods Benchmark')
    parser.add_argument('--dataset', default='imdb_small', help='Dataset to test (use executed queries from this dataset)')
    parser.add_argument('--query_type', default='ap', choices=['ap', 'tp'], help='Query type (used only for static files)')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to test')
    parser.add_argument('--use-executed', action='store_true', default=True, help='Use executed queries from data/execution_data (both engines successful)')
    
    args = parser.parse_args()
    
    benchmark = AQDRoutingBenchmark()
    results = benchmark.run_full_benchmark(args.dataset, args.query_type, args.num_queries, use_executed=args.use_executed)
    
    if results:
        logger.info("Benchmark completed successfully!")
    else:
        logger.error("Benchmark failed!")

if __name__ == '__main__':
    main()
