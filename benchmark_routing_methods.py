#!/usr/bin/env python3
"""
Real Concurrent Query Execution Benchmark for AQD Routing Methods
Tests actual PostgreSQL system with all four routing methods:
1. Default (pg_duckdb heuristic)
2. Cost-threshold 
3. LightGBM-based
4. GNN-based (placeholder for now)
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
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': 'wuy'
        }
        
        # Load trained LightGBM model
        self.model_path = '/home/wuy/DB/pg_duckdb_postgres/models/lightgbm_model.txt'
        self.load_lightgbm_model()
        
        # Query directories
        self.query_base_dir = '/home/wuy/DB/pg_duckdb_postgres/data/benchmark_queries'
        self.results_dir = '/home/wuy/DB/pg_duckdb_postgres/results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Routing methods to test
        self.routing_methods = ['default', 'cost_threshold', 'lightgbm', 'gnn']
        
    def load_lightgbm_model(self):
        """Load the trained LightGBM model"""
        try:
            import lightgbm as lgb
            self.lgb_model = lgb.Booster(model_file=self.model_path)
            logger.info(f"Loaded LightGBM model from {self.model_path}")
            
            # Load feature names
            with open('/home/wuy/DB/pg_duckdb_postgres/models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
                logger.info(f"Model expects {len(self.feature_names)} features")
                
        except Exception as e:
            logger.warning(f"Could not load LightGBM model: {e}")
            self.lgb_model = None
            self.feature_names = []
    
    def get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.pg_config)
    
    def set_routing_method(self, conn, method):
        """Set the routing method in PostgreSQL"""
        cursor = conn.cursor()
        
        if method == 'default':
            # Use pg_duckdb default heuristic
            cursor.execute("SET aqd.routing_method = 'default';")
            cursor.execute("SET pg_duckdb.force_execution = 'auto';")
            
        elif method == 'cost_threshold':
            # Use cost threshold method
            cursor.execute("SET aqd.routing_method = 'cost_threshold';")
            cursor.execute("SET aqd.cost_threshold = 1000.0;")  # Configurable threshold
            
        elif method == 'lightgbm':
            # Use LightGBM-based routing
            cursor.execute("SET aqd.routing_method = 'lightgbm';")
            cursor.execute("SET aqd.enable_feature_logging = on;")
            
        elif method == 'gnn':
            # Use GNN-based routing (placeholder - uses enhanced heuristics for now)
            cursor.execute("SET aqd.routing_method = 'gnn';")
            cursor.execute("SET aqd.enable_plan_analysis = on;")
            
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
        """Predict query routing using LightGBM"""
        if not self.lgb_model:
            return 'postgres'  # Default fallback
            
        features = self.extract_query_features(query_text)
        feature_vector = [features.get(fname, 0.0) for fname in self.feature_names]
        
        try:
            prediction = self.lgb_model.predict([feature_vector])[0]
            # If prediction > 0, prefer DuckDB; otherwise PostgreSQL
            return 'duckdb' if prediction > 0 else 'postgres'
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
            return 'postgres'
    
    def execute_query(self, query_text, routing_method, query_id):
        """Execute a single query and measure performance"""
        start_time = time.time()
        routing_start = time.time()
        
        try:
            conn = self.get_connection()
            self.set_routing_method(conn, routing_method)
            
            # Routing decision time
            if routing_method == 'lightgbm':
                predicted_engine = self.predict_routing(query_text)
                routing_time = time.time() - routing_start
            else:
                routing_time = time.time() - routing_start
            
            # Execute query
            cursor = conn.cursor()
            query_start = time.time()
            cursor.execute(query_text)
            results = cursor.fetchall()
            query_time = time.time() - query_start
            
            total_time = time.time() - start_time
            
            conn.close()
            
            return {
                'query_id': query_id,
                'routing_method': routing_method,
                'routing_time_ms': routing_time * 1000,
                'query_time_ms': query_time * 1000,
                'total_time_ms': total_time * 1000,
                'success': True,
                'error': None,
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
                'error': str(e),
                'result_count': 0
            }
    
    def load_test_queries(self, dataset, query_type, limit=100):
        """Load queries for testing"""
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
    
    def run_concurrent_benchmark(self, queries, routing_method, max_workers=10):
        """Run concurrent queries for a specific routing method"""
        logger.info(f"Running concurrent benchmark for {routing_method} with {len(queries)} queries")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.execute_query, q['text'], routing_method, q['id']): q 
                for q in queries
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                result = future.result()
                results.append(result)
                
                if len(results) % 10 == 0:
                    logger.info(f"Completed {len(results)}/{len(queries)} queries for {routing_method}")
        
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
    
    def run_full_benchmark(self, dataset='financial', query_type='ap', num_queries=100):
        """Run full benchmark comparing all routing methods"""
        logger.info(f"Starting full routing benchmark")
        logger.info(f"Dataset: {dataset}, Query type: {query_type}, Queries: {num_queries}")
        
        # Load test queries
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
- ML-based routing using trained model
- Extracts {len(self.feature_names)} query features
- Predicts optimal execution engine

### GNN (Placeholder)
- Currently uses enhanced heuristics
- Future: Graph neural network on query plans
- Plan structure analysis
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
    parser.add_argument('--dataset', default='financial', help='Dataset to test')
    parser.add_argument('--query_type', default='ap', choices=['ap', 'tp'], help='Query type')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to test')
    
    args = parser.parse_args()
    
    benchmark = AQDRoutingBenchmark()
    results = benchmark.run_full_benchmark(args.dataset, args.query_type, args.num_queries)
    
    if results:
        logger.info("Benchmark completed successfully!")
    else:
        logger.error("Benchmark failed!")

if __name__ == '__main__':
    main()