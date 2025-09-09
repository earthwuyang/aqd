#!/usr/bin/env python3
"""
Integrated Routing Benchmark - Mixed Workload from All Datasets
- Loads queries from all datasets in data/benchmark_queries/
- Mixes queries from different datasets randomly
- Tests different query counts: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
- Compares routing methods: default, cost_threshold, lightgbm, gnn
- Uses fixed concurrency level for fair comparison
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
import pandas as pd
from typing import List, Dict, Tuple
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedRoutingBenchmark:
    def __init__(self):
        """Initialize benchmark with mixed queries from all datasets"""
        self.base_dir = Path(__file__).resolve().parent
        self.queries_dir = self.base_dir / 'data' / 'benchmark_queries'
        
        # Load all queries from all datasets
        self.all_queries = self._load_all_queries()
        logger.info(f"Loaded {len(self.all_queries)} total queries from all datasets")

    def _load_sql_file(self, path: Path) -> List[str]:
        """Load SQL queries from a file"""
        if not path.exists():
            return []
        text = path.read_text(encoding='utf-8', errors='ignore')
        queries = [q.strip() for q in text.split(';') if q.strip()]
        return queries

    def _load_all_queries(self) -> List[Tuple[str, str, str]]:
        """Load all queries from all datasets and mix them
        Returns list of tuples: (query, dataset, query_type)"""
        all_queries = []
        
        if not self.queries_dir.exists():
            logger.error(f"Queries directory not found: {self.queries_dir}")
            return []
        
        # Iterate through all dataset directories
        for dataset_dir in self.queries_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            
            # Load AP queries
            ap_file = dataset_dir / 'workload_10k_ap_queries.sql'
            if ap_file.exists():
                ap_queries = self._load_sql_file(ap_file)
                for query in ap_queries[:1000]:  # Limit to 1000 per type per dataset
                    all_queries.append((query, dataset_name, 'AP'))
                logger.info(f"  Loaded {len(ap_queries[:1000])} AP queries from {dataset_name}")
            
            # Load TP queries  
            tp_file = dataset_dir / 'workload_10k_tp_queries.sql'
            if tp_file.exists():
                tp_queries = self._load_sql_file(tp_file)
                for query in tp_queries[:1000]:  # Limit to 1000 per type per dataset
                    all_queries.append((query, dataset_name, 'TP'))
                logger.info(f"  Loaded {len(tp_queries[:1000])} TP queries from {dataset_name}")
        
        # Shuffle all queries to mix datasets
        random.shuffle(all_queries)
        return all_queries
    
    def get_connection(self, dataset: str):
        """Get a connection to the specified dataset"""
        pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': dataset,
            'user': 'wuy'
        }
        return psycopg2.connect(**pg_config)
    
    def set_routing_method(self, conn, method: str):
        """Set the routing method for this connection"""
        cursor = conn.cursor()
        try:
            if method == 'default':
                cursor.execute("SET aqd.routing_method = 0;")  # AQD_ROUTE_DEFAULT
            elif method == 'cost_threshold':
                cursor.execute("SET aqd.routing_method = 1;")  # AQD_ROUTE_COST_THRESHOLD
                cursor.execute("SET aqd.cost_threshold = 1000.0;")
            elif method == 'lightgbm':
                cursor.execute("SET aqd.routing_method = 2;")  # AQD_ROUTE_LIGHTGBM
                # Load trained LightGBM model
                model_path = self.base_dir / 'models' / 'lightgbm_model.txt'
                if model_path.exists():
                    cursor.execute(f"SET aqd.lightgbm_model_path = '{str(model_path)}';")
            elif method == 'gnn':
                cursor.execute("SET aqd.routing_method = 3;")  # AQD_ROUTE_GNN
                # Load trained GNN model - will use default path if not specified
                model_path = self.base_dir / 'models' / 'rginn_routing_model.txt'
                if model_path.exists():
                    cursor.execute(f"SET aqd.gnn_model_path = '{str(model_path)}';")
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not set routing method {method}: {e}")
    
    def execute_single_query(self, query_info: Tuple[str, str, str], routing_method: str, query_id: int):
        """Execute a single query with specified routing method"""
        query, dataset, query_type = query_info
        start_time = time.time()
        
        try:
            conn = self.get_connection(dataset)
            self.set_routing_method(conn, routing_method)
            cursor = conn.cursor()
            
            cursor.execute(query)
            cursor.fetchall()
            
            latency = time.time() - start_time
            cursor.close()
            conn.close()
            
            return {
                'query_id': query_id,
                'dataset': dataset,
                'query_type': query_type,
                'routing_method': routing_method,
                'latency': latency,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'query_id': query_id,
                'dataset': dataset,
                'query_type': query_type,
                'routing_method': routing_method,
                'latency': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def run_concurrent_benchmark(self, routing_method: str, num_queries: int, max_workers: int = 10):
        """Run benchmark with specified concurrency"""
        if not self.all_queries:
            logger.error("No queries available for benchmarking")
            return {}
        
        # Select queries (cycling if needed)
        if num_queries <= len(self.all_queries):
            selected_queries = self.all_queries[:num_queries]
        else:
            # Cycle through queries if we need more than available
            selected_queries = []
            for i in range(num_queries):
                selected_queries.append(self.all_queries[i % len(self.all_queries)])
        
        results = []
        
        logger.info(f"Running {num_queries} queries with {routing_method} routing (concurrency: {max_workers})")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.execute_single_query, query_info, routing_method, i): i 
                for i, query_info in enumerate(selected_queries)
            }
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"  Progress: {completed}/{num_queries} queries completed")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        latencies = [r['latency'] for r in successful_results]
        
        if latencies:
            stats = {
                'routing_method': routing_method,
                'concurrency': max_workers,
                'num_queries': num_queries,
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results),
                'total_time': total_time,
                'makespan': total_time,
                'mean_latency': statistics.mean(latencies),
                'median_latency': statistics.median(latencies),
                'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
                'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else max(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'throughput': len(successful_results) / total_time,
                'queries_per_second': num_queries / total_time
            }
            
            # Breakdown by dataset
            dataset_counts = {}
            dataset_latencies = {}
            for r in successful_results:
                ds = r['dataset']
                if ds not in dataset_counts:
                    dataset_counts[ds] = 0
                    dataset_latencies[ds] = []
                dataset_counts[ds] += 1
                dataset_latencies[ds].append(r['latency'])
            
            stats['dataset_breakdown'] = {
                ds: {
                    'count': dataset_counts[ds],
                    'mean_latency': statistics.mean(dataset_latencies[ds])
                }
                for ds in dataset_counts
            }
            
            # Breakdown by query type
            type_counts = {'AP': 0, 'TP': 0}
            type_latencies = {'AP': [], 'TP': []}
            for r in successful_results:
                qt = r['query_type']
                type_counts[qt] += 1
                type_latencies[qt].append(r['latency'])
            
            stats['query_type_breakdown'] = {
                qt: {
                    'count': type_counts[qt],
                    'mean_latency': statistics.mean(type_latencies[qt]) if type_latencies[qt] else 0
                }
                for qt in type_counts
            }
        else:
            stats = {
                'routing_method': routing_method,
                'concurrency': max_workers,
                'num_queries': num_queries,
                'successful': 0,
                'failed': len(results),
                'total_time': total_time,
                'makespan': total_time,
                'error': 'All queries failed'
            }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Integrated routing benchmark with mixed workload')
    parser.add_argument('--methods', nargs='+', 
                       default=['default', 'cost_threshold', 'lightgbm', 'gnn'],
                       help='Routing methods to test')
    parser.add_argument('--query-counts', nargs='+', type=int,
                       default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                       help='Number of queries to test')
    parser.add_argument('--concurrency', type=int, default=1000,
                       help='Fixed concurrency level for all tests')
    parser.add_argument('--output', default='integrated_benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = IntegratedRoutingBenchmark()
    
    if len(benchmark.all_queries) == 0:
        logger.error("No queries loaded! Please ensure benchmark queries exist.")
        return
    
    all_results = []
    summary_results = []
    
    logger.info(f"\n{'='*80}")
    logger.info("INTEGRATED ROUTING BENCHMARK")
    logger.info(f"{'='*80}")
    logger.info(f"Total available queries: {len(benchmark.all_queries)}")
    logger.info(f"Methods to test: {args.methods}")
    logger.info(f"Query counts to test: {args.query_counts}")
    logger.info(f"Fixed concurrency level: {args.concurrency}")
    
    # Run benchmarks
    for num_queries in args.query_counts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {num_queries} queries")
        logger.info(f"{'='*60}")
        
        for method in args.methods:
            logger.info(f"\nRunning {method} with {num_queries} queries...")
            
            stats = benchmark.run_concurrent_benchmark(
                routing_method=method,
                num_queries=num_queries,
                max_workers=args.concurrency
            )
            
            all_results.append(stats)
            
            # Print immediate results
            if 'error' not in stats:
                logger.info(f"  Successful: {stats['successful']}/{stats['num_queries']}")
                logger.info(f"  Makespan: {stats['makespan']:.2f}s")
                logger.info(f"  Mean latency: {stats['mean_latency']:.3f}s")
                logger.info(f"  P95 latency: {stats['p95_latency']:.3f}s")
                logger.info(f"  Throughput: {stats['throughput']:.1f} queries/s")
                
                # Show query type breakdown
                if 'query_type_breakdown' in stats:
                    logger.info(f"  Query types:")
                    for qt, info in stats['query_type_breakdown'].items():
                        logger.info(f"    {qt}: {info['count']} queries, {info['mean_latency']:.3f}s mean")
            else:
                logger.error(f"  Error: {stats['error']}")
            
            # Add to summary
            summary_results.append({
                'method': method,
                'num_queries': num_queries,
                'concurrency': args.concurrency,
                'makespan': stats.get('makespan', -1),
                'mean_latency': stats.get('mean_latency', -1),
                'p95_latency': stats.get('p95_latency', -1),
                'throughput': stats.get('throughput', 0),
                'success_rate': stats['successful'] / stats['num_queries'] if stats.get('num_queries', 0) > 0 else 0
            })
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nDetailed results saved to {output_path}")
    
    # Save summary CSV
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv = output_path.with_suffix('.csv')
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Summary saved to {summary_csv}")
        
        # Print comparison table
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*80)
        
        # Pivot table for easy comparison
        pivot = summary_df.pivot_table(
            values=['makespan', 'throughput'],
            index='num_queries',
            columns='method',
            aggfunc='mean'
        )
        
        print("\nMakespan (seconds) by method and number of queries:")
        print(pivot['makespan'].to_string())
        
        print("\nThroughput (queries/second) by method and number of queries:")
        print(pivot['throughput'].to_string())
        
        # Find best method per query count
        logger.info("\nBest performing method by number of queries:")
        for nq in args.query_counts:
            nq_data = summary_df[summary_df['num_queries'] == nq]
            if not nq_data.empty:
                best = nq_data.loc[nq_data['makespan'].idxmin()]
                logger.info(f"  {nq} queries: {best['method']} (makespan: {best['makespan']:.2f}s, throughput: {best['throughput']:.1f} q/s)")
    
    logger.info("\n" + "="*80)
    logger.info("Benchmark complete!")

if __name__ == "__main__":
    main()