#!/usr/bin/env python3
"""
AQD Comprehensive Benchmarking Framework

Tests all routing methods and generates performance comparison reports.
Includes concurrent query testing and makespan/latency analysis.
"""

import os
import sys
import time
import json
import threading
import concurrent.futures
import subprocess
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class AQDBenchmarkRunner:
    """
    Comprehensive benchmarking system for AQD routing methods
    """
    
    def __init__(self, 
                 postgres_config: Dict,
                 benchmark_config: Dict,
                 output_dir: str = "/tmp/aqd_benchmarks"):
        
        self.postgres_config = postgres_config
        self.benchmark_config = benchmark_config
        self.output_dir = output_dir
        
        # Routing methods to test
        self.routing_methods = {
            0: "Default (pg_duckdb heuristic)",
            1: "Cost Threshold", 
            2: "LightGBM ML",
            3: "Graph Neural Network"
        }
        
        # Benchmark results
        self.results = {
            'single_query': {},
            'concurrent_query': {},
            'makespan_analysis': {},
            'routing_accuracy': {},
            'resource_usage': {}
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def setup_postgres_routing(self, routing_method: int) -> bool:
        """Configure PostgreSQL to use specific routing method"""
        try:
            with psycopg2.connect(**self.postgres_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET aqd.routing_method = {routing_method};")
                    cur.execute("SET aqd.enable_feature_logging = true;")
                    
                    # Method-specific configuration
                    if routing_method == 1:  # Cost threshold
                        threshold = self.benchmark_config.get('cost_threshold', 10000)
                        cur.execute(f"SET aqd.cost_threshold = {threshold};")
                    elif routing_method == 2:  # LightGBM
                        model_path = self.benchmark_config.get('lightgbm_model_path', '')
                        if model_path:
                            cur.execute(f"SET aqd.lightgbm_model_path = '{model_path}';")
                    elif routing_method == 3:  # GNN
                        model_path = self.benchmark_config.get('gnn_model_path', '')
                        if model_path:
                            cur.execute(f"SET aqd.gnn_model_path = '{model_path}';")
                    
                    conn.commit()
            
            print(f"Configured routing method: {self.routing_methods[routing_method]}")
            return True
            
        except Exception as e:
            print(f"Error configuring routing method {routing_method}: {e}")
            return False
    
    def execute_query_with_timing(self, query: str, query_id: str = None) -> Dict:
        """Execute a single query and measure performance"""
        start_time = time.time()
        
        try:
            with psycopg2.connect(**self.postgres_config) as conn:
                with conn.cursor() as cur:
                    # Execute with timing
                    execution_start = time.time()
                    cur.execute(query)
                    results = cur.fetchall()
                    execution_end = time.time()
                    
                    # Get execution plan if possible
                    try:
                        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                        explain_result = cur.fetchone()[0]
                        planning_time = explain_result[0].get('Planning Time', 0)
                        execution_time = explain_result[0].get('Execution Time', 0)
                    except:
                        planning_time = 0
                        execution_time = (execution_end - execution_start) * 1000
                    
                    return {
                        'success': True,
                        'query_id': query_id,
                        'execution_time_ms': execution_time,
                        'planning_time_ms': planning_time,
                        'total_time_ms': execution_time + planning_time,
                        'result_count': len(results),
                        'wall_clock_time_ms': (execution_end - execution_start) * 1000,
                        'error': None
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'query_id': query_id,
                'execution_time_ms': None,
                'planning_time_ms': None,
                'total_time_ms': (end_time - start_time) * 1000,
                'result_count': 0,
                'wall_clock_time_ms': (end_time - start_time) * 1000,
                'error': str(e)
            }
    
    def run_single_query_benchmark(self, queries: List[Tuple[str, str]]) -> Dict:
        """Benchmark single query execution across all routing methods"""
        print("\n=== Single Query Benchmark ===")
        
        results = {}
        
        for method_id, method_name in self.routing_methods.items():
            print(f"\nTesting {method_name}...")
            
            # Configure routing method
            if not self.setup_postgres_routing(method_id):
                continue
            
            method_results = []
            
            # Execute each query
            for query_id, query in tqdm(queries, desc=f"Method {method_id}"):
                result = self.execute_query_with_timing(query, query_id)
                result['routing_method'] = method_id
                result['method_name'] = method_name
                method_results.append(result)
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
            
            results[method_id] = method_results
            
            # Calculate summary statistics
            successful_queries = [r for r in method_results if r['success']]
            if successful_queries:
                exec_times = [r['total_time_ms'] for r in successful_queries]
                print(f"  Avg execution time: {statistics.mean(exec_times):.2f}ms")
                print(f"  Median execution time: {statistics.median(exec_times):.2f}ms")
                print(f"  Success rate: {len(successful_queries)/len(method_results)*100:.1f}%")
        
        return results
    
    def run_concurrent_query_benchmark(self, 
                                     queries: List[Tuple[str, str]], 
                                     max_workers: int = 4,
                                     batch_size: int = 10) -> Dict:
        """Benchmark concurrent query execution"""
        print(f"\n=== Concurrent Query Benchmark (workers={max_workers}) ===")
        
        results = {}
        
        for method_id, method_name in self.routing_methods.items():
            print(f"\nTesting concurrent execution: {method_name}...")
            
            # Configure routing method
            if not self.setup_postgres_routing(method_id):
                continue
            
            # Prepare query batches
            query_batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
            
            concurrent_results = []
            total_makespan = 0
            
            for batch_idx, batch in enumerate(tqdm(query_batches, desc=f"Concurrent {method_id}")):
                batch_start_time = time.time()
                
                # Execute batch concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_query = {
                        executor.submit(self.execute_query_with_timing, query, f"{query_id}_batch{batch_idx}"): 
                        (query_id, query) for query_id, query in batch
                    }
                    
                    batch_results = []
                    for future in concurrent.futures.as_completed(future_to_query):
                        try:
                            result = future.result(timeout=30)  # 30 second timeout per query
                            batch_results.append(result)
                        except concurrent.futures.TimeoutError:
                            query_id, query = future_to_query[future]
                            batch_results.append({
                                'success': False,
                                'query_id': query_id,
                                'error': 'Timeout',
                                'total_time_ms': 30000
                            })
                        except Exception as e:
                            query_id, query = future_to_query[future]
                            batch_results.append({
                                'success': False,
                                'query_id': query_id,
                                'error': str(e),
                                'total_time_ms': 0
                            })
                
                batch_end_time = time.time()
                batch_makespan = (batch_end_time - batch_start_time) * 1000
                total_makespan += batch_makespan
                
                # Add batch info to results
                for result in batch_results:
                    result['routing_method'] = method_id
                    result['method_name'] = method_name
                    result['batch_id'] = batch_idx
                    result['batch_makespan_ms'] = batch_makespan
                
                concurrent_results.extend(batch_results)
            
            results[method_id] = {
                'query_results': concurrent_results,
                'total_makespan_ms': total_makespan,
                'avg_batch_makespan_ms': total_makespan / len(query_batches),
                'total_queries': len(queries),
                'successful_queries': len([r for r in concurrent_results if r['success']])
            }
            
            print(f"  Total makespan: {total_makespan:.0f}ms")
            print(f"  Avg batch makespan: {total_makespan/len(query_batches):.0f}ms")
            print(f"  Success rate: {results[method_id]['successful_queries']/len(queries)*100:.1f}%")
        
        return results
    
    def analyze_routing_accuracy(self) -> Dict:
        """Analyze routing decision accuracy by comparing predicted vs actual performance"""
        print("\n=== Routing Accuracy Analysis ===")
        
        accuracy_results = {}
        
        # This would require access to AQD feature logs to compare
        # predicted routing decisions vs actual performance
        try:
            feature_log_path = "/tmp/aqd_features.csv"
            if os.path.exists(feature_log_path):
                df = pd.read_csv(feature_log_path)
                
                # Calculate routing accuracy for each method
                for method_id in self.routing_methods.keys():
                    method_data = df[df.get('routing_method', 0) == method_id]
                    
                    if len(method_data) > 0:
                        # Simple accuracy calculation based on whether PostgreSQL or DuckDB was faster
                        correct_decisions = 0
                        total_decisions = 0
                        
                        for _, row in method_data.iterrows():
                            if pd.notna(row.get('postgres_time_ms')) and pd.notna(row.get('duckdb_time_ms')):
                                pg_faster = row['postgres_time_ms'] < row['duckdb_time_ms']
                                # This would need to be matched with actual routing decisions
                                total_decisions += 1
                        
                        accuracy = correct_decisions / max(total_decisions, 1)
                        accuracy_results[method_id] = {
                            'accuracy': accuracy,
                            'total_decisions': total_decisions,
                            'method_name': self.routing_methods[method_id]
                        }
                        
                        print(f"  {self.routing_methods[method_id]}: {accuracy*100:.1f}% accuracy")
        
        except Exception as e:
            print(f"Warning: Could not analyze routing accuracy: {e}")
        
        return accuracy_results
    
    def generate_performance_plots(self) -> None:
        """Generate comprehensive performance visualization plots"""
        print("\n=== Generating Performance Plots ===")
        
        # Set up the plotting environment
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AQD Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Single Query Latency Distribution
        if 'single_query' in self.results and self.results['single_query']:
            ax = axes[0, 0]
            data_for_plot = []
            labels = []
            
            for method_id, method_results in self.results['single_query'].items():
                successful = [r for r in method_results if r['success']]
                if successful:
                    times = [r['total_time_ms'] for r in successful]
                    data_for_plot.append(times)
                    labels.append(f"Method {method_id}")
            
            if data_for_plot:
                ax.boxplot(data_for_plot, labels=labels)
                ax.set_title('Single Query Latency Distribution')
                ax.set_ylabel('Execution Time (ms)')
                ax.set_yscale('log')
        
        # Plot 2: Concurrent Query Makespan
        if 'concurrent_query' in self.results and self.results['concurrent_query']:
            ax = axes[0, 1]
            methods = []
            makespans = []
            
            for method_id, results in self.results['concurrent_query'].items():
                methods.append(f"Method {method_id}")
                makespans.append(results['total_makespan_ms'])
            
            if methods:
                bars = ax.bar(methods, makespans, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax.set_title('Concurrent Query Makespan')
                ax.set_ylabel('Total Makespan (ms)')
                
                # Add value labels on bars
                for bar, value in zip(bars, makespans):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(makespans)*0.01,
                           f'{value:.0f}ms', ha='center', va='bottom')
        
        # Plot 3: Success Rate Comparison
        ax = axes[0, 2]
        methods = []
        success_rates = []
        
        for method_id in self.routing_methods.keys():
            if method_id in self.results.get('single_query', {}):
                method_results = self.results['single_query'][method_id]
                success_rate = len([r for r in method_results if r['success']]) / len(method_results) * 100
                methods.append(f"Method {method_id}")
                success_rates.append(success_rate)
        
        if methods:
            bars = ax.bar(methods, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_title('Query Success Rate')
            ax.set_ylabel('Success Rate (%)')
            ax.set_ylim(0, 105)
            
            # Add value labels
            for bar, value in zip(bars, success_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Latency vs Query Complexity (scatter plot)
        ax = axes[1, 0]
        if 'single_query' in self.results:
            for method_id, method_results in self.results['single_query'].items():
                successful = [r for r in method_results if r['success']]
                if successful:
                    # Use result count as a proxy for complexity
                    complexities = [r.get('result_count', 1) for r in successful]
                    times = [r['total_time_ms'] for r in successful]
                    ax.scatter(complexities, times, label=f"Method {method_id}", alpha=0.6)
            
            ax.set_title('Latency vs Query Complexity')
            ax.set_xlabel('Result Count (proxy for complexity)')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_yscale('log')
            ax.legend()
        
        # Plot 5: Routing Method Performance Comparison
        ax = axes[1, 1]
        if 'single_query' in self.results:
            method_names = []
            avg_times = []
            
            for method_id, method_results in self.results['single_query'].items():
                successful = [r for r in method_results if r['success']]
                if successful:
                    avg_time = statistics.mean([r['total_time_ms'] for r in successful])
                    method_names.append(self.routing_methods[method_id][:15])  # Truncate long names
                    avg_times.append(avg_time)
            
            if method_names:
                bars = ax.bar(method_names, avg_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax.set_title('Average Query Execution Time')
                ax.set_ylabel('Avg Execution Time (ms)')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for bar, value in zip(bars, avg_times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.01,
                           f'{value:.1f}ms', ha='center', va='bottom')
        
        # Plot 6: Routing Accuracy (if available)
        ax = axes[1, 2]
        if 'routing_accuracy' in self.results and self.results['routing_accuracy']:
            methods = []
            accuracies = []
            
            for method_id, acc_data in self.results['routing_accuracy'].items():
                methods.append(f"Method {method_id}")
                accuracies.append(acc_data['accuracy'] * 100)
            
            if methods:
                bars = ax.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax.set_title('Routing Decision Accuracy')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 105)
                
                # Add value labels
                for bar, value in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Routing Accuracy\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Routing Decision Accuracy')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'aqd_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to: {plot_path}")
        
        plt.close()
    
    def export_detailed_results(self) -> str:
        """Export detailed benchmark results to JSON and CSV"""
        print("\n=== Exporting Detailed Results ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export JSON
        json_file = os.path.join(self.output_dir, f"aqd_benchmark_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Export CSV for easier analysis
        csv_data = []
        
        # Single query results
        if 'single_query' in self.results:
            for method_id, method_results in self.results['single_query'].items():
                for result in method_results:
                    csv_row = {
                        'benchmark_type': 'single_query',
                        'routing_method': method_id,
                        'method_name': self.routing_methods.get(method_id, 'Unknown'),
                        'query_id': result.get('query_id', ''),
                        'success': result['success'],
                        'execution_time_ms': result.get('execution_time_ms'),
                        'planning_time_ms': result.get('planning_time_ms'),
                        'total_time_ms': result.get('total_time_ms'),
                        'result_count': result.get('result_count', 0),
                        'error': result.get('error', '')
                    }
                    csv_data.append(csv_row)
        
        # Concurrent query results
        if 'concurrent_query' in self.results:
            for method_id, concurrent_data in self.results['concurrent_query'].items():
                for result in concurrent_data.get('query_results', []):
                    csv_row = {
                        'benchmark_type': 'concurrent_query',
                        'routing_method': method_id,
                        'method_name': self.routing_methods.get(method_id, 'Unknown'),
                        'query_id': result.get('query_id', ''),
                        'batch_id': result.get('batch_id', ''),
                        'success': result['success'],
                        'execution_time_ms': result.get('execution_time_ms'),
                        'total_time_ms': result.get('total_time_ms'),
                        'batch_makespan_ms': result.get('batch_makespan_ms'),
                        'error': result.get('error', '')
                    }
                    csv_data.append(csv_row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(self.output_dir, f"aqd_benchmark_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
            print(f"CSV results exported to: {csv_file}")
        
        print(f"JSON results exported to: {json_file}")
        
        return json_file
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        print("\n=== Generating Summary Report ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"aqd_benchmark_report_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write("# AQD Performance Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Single Query Performance
            if 'single_query' in self.results:
                f.write("## Single Query Performance\n\n")
                f.write("| Method | Avg Time (ms) | Median Time (ms) | Success Rate | Queries Tested |\n")
                f.write("|--------|---------------|------------------|--------------|----------------|\n")
                
                for method_id, method_results in self.results['single_query'].items():
                    successful = [r for r in method_results if r['success']]
                    if successful:
                        avg_time = statistics.mean([r['total_time_ms'] for r in successful])
                        median_time = statistics.median([r['total_time_ms'] for r in successful])
                        success_rate = len(successful) / len(method_results) * 100
                        
                        f.write(f"| {self.routing_methods[method_id]} | {avg_time:.2f} | {median_time:.2f} | {success_rate:.1f}% | {len(method_results)} |\n")
                
                f.write("\n")
            
            # Concurrent Query Performance
            if 'concurrent_query' in self.results:
                f.write("## Concurrent Query Performance\n\n")
                f.write("| Method | Total Makespan (ms) | Avg Batch Makespan (ms) | Success Rate | Total Queries |\n")
                f.write("|--------|---------------------|--------------------------|--------------|---------------|\n")
                
                for method_id, concurrent_data in self.results['concurrent_query'].items():
                    total_makespan = concurrent_data['total_makespan_ms']
                    avg_makespan = concurrent_data['avg_batch_makespan_ms']
                    success_rate = concurrent_data['successful_queries'] / concurrent_data['total_queries'] * 100
                    total_queries = concurrent_data['total_queries']
                    
                    f.write(f"| {self.routing_methods[method_id]} | {total_makespan:.0f} | {avg_makespan:.0f} | {success_rate:.1f}% | {total_queries} |\n")
                
                f.write("\n")
            
            # Best performing method
            if 'single_query' in self.results:
                best_method_id = None
                best_avg_time = float('inf')
                
                for method_id, method_results in self.results['single_query'].items():
                    successful = [r for r in method_results if r['success']]
                    if successful:
                        avg_time = statistics.mean([r['total_time_ms'] for r in successful])
                        if avg_time < best_avg_time:
                            best_avg_time = avg_time
                            best_method_id = method_id
                
                if best_method_id is not None:
                    f.write(f"## Best Performing Method\n\n")
                    f.write(f"**{self.routing_methods[best_method_id]}** achieved the lowest average query latency of {best_avg_time:.2f}ms.\n\n")
            
            # Performance improvement analysis
            if 'single_query' in self.results and len(self.results['single_query']) > 1:
                f.write("## Performance Improvements\n\n")
                
                # Compare against default method (method 0)
                if 0 in self.results['single_query']:
                    default_results = self.results['single_query'][0]
                    default_successful = [r for r in default_results if r['success']]
                    
                    if default_successful:
                        default_avg = statistics.mean([r['total_time_ms'] for r in default_successful])
                        
                        for method_id, method_results in self.results['single_query'].items():
                            if method_id != 0:
                                successful = [r for r in method_results if r['success']]
                                if successful:
                                    method_avg = statistics.mean([r['total_time_ms'] for r in successful])
                                    improvement = (default_avg - method_avg) / default_avg * 100
                                    
                                    if improvement > 0:
                                        f.write(f"- **{self.routing_methods[method_id]}** is {improvement:.1f}% faster than default method\n")
                                    else:
                                        f.write(f"- **{self.routing_methods[method_id]}** is {abs(improvement):.1f}% slower than default method\n")
                
                f.write("\n")
            
            # System configuration
            f.write("## System Configuration\n\n")
            f.write(f"- Benchmark queries: {len(self.benchmark_config.get('queries', []))}\n")
            f.write(f"- Concurrent workers: {self.benchmark_config.get('concurrent_workers', 4)}\n")
            f.write(f"- PostgreSQL config: {self.postgres_config['host']}:{self.postgres_config['port']}\n")
            f.write(f"- Output directory: {self.output_dir}\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- Performance plots: `aqd_performance_comparison.png`\n")
            f.write("- Detailed results: `aqd_benchmark_results_*.json`\n")
            f.write("- CSV data: `aqd_benchmark_results_*.csv`\n")
            f.write("- This report: `aqd_benchmark_report_*.md`\n")
        
        print(f"Summary report generated: {report_file}")
        return report_file
    
    def run_comprehensive_benchmark(self, test_queries: List[Tuple[str, str]]) -> Dict:
        """Run the complete benchmark suite"""
        print("Starting AQD Comprehensive Benchmark Suite")
        print(f"Testing {len(test_queries)} queries across {len(self.routing_methods)} routing methods")
        
        start_time = time.time()
        
        # Run single query benchmark
        self.results['single_query'] = self.run_single_query_benchmark(test_queries)
        
        # Run concurrent query benchmark
        concurrent_workers = self.benchmark_config.get('concurrent_workers', 4)
        self.results['concurrent_query'] = self.run_concurrent_query_benchmark(
            test_queries[:20],  # Use subset for concurrent testing
            max_workers=concurrent_workers
        )
        
        # Analyze routing accuracy
        self.results['routing_accuracy'] = self.analyze_routing_accuracy()
        
        # Generate visualizations
        self.generate_performance_plots()
        
        # Export detailed results
        json_file = self.export_detailed_results()
        
        # Generate summary report
        report_file = self.generate_summary_report()
        
        total_time = time.time() - start_time
        
        print(f"\n=== Benchmark Complete ===")
        print(f"Total benchmark time: {total_time:.1f} seconds")
        print(f"Results available in: {self.output_dir}")
        print(f"Summary report: {report_file}")
        
        return self.results


def create_test_queries() -> List[Tuple[str, str]]:
    """Create a set of test queries for benchmarking"""
    queries = []
    
    # Simple analytical queries (OLAP)
    queries.extend([
        ("olap_1", "SELECT COUNT(*), AVG(amount) FROM transactions WHERE date > '2023-01-01'"),
        ("olap_2", "SELECT category, SUM(amount) FROM sales GROUP BY category ORDER BY SUM(amount) DESC"),
        ("olap_3", "SELECT EXTRACT(MONTH FROM date) as month, COUNT(*) FROM orders WHERE status = 'completed' GROUP BY month"),
        ("olap_4", "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id HAVING COUNT(*) > 5"),
        ("olap_5", "SELECT p.name, SUM(oi.quantity * oi.price) as total_sales FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.name"),
    ])
    
    # Simple transactional queries (OLTP)
    queries.extend([
        ("oltp_1", "SELECT * FROM users WHERE id = 12345"),
        ("oltp_2", "SELECT * FROM products WHERE sku = 'ABC123' LIMIT 1"),
        ("oltp_3", "SELECT u.name, u.email FROM users u WHERE u.active = true LIMIT 10"),
        ("oltp_4", "SELECT * FROM orders WHERE customer_id = 67890 AND status = 'pending'"),
        ("oltp_5", "SELECT COUNT(*) FROM inventory WHERE product_id = 123 AND quantity > 0"),
    ])
    
    # Mixed complexity queries
    queries.extend([
        ("mixed_1", "SELECT c.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name"),
        ("mixed_2", "WITH monthly_sales AS (SELECT EXTRACT(MONTH FROM date) as month, SUM(amount) as total FROM sales GROUP BY month) SELECT * FROM monthly_sales WHERE total > 10000"),
        ("mixed_3", "SELECT p.category, AVG(r.rating) as avg_rating, COUNT(r.id) as review_count FROM products p JOIN reviews r ON p.id = r.product_id GROUP BY p.category HAVING COUNT(r.id) > 100"),
    ])
    
    return queries


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='AQD Comprehensive Benchmark Runner')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', default=5432, type=int, help='PostgreSQL port')  
    parser.add_argument('--database', default='benchmark_datasets', help='PostgreSQL database')
    parser.add_argument('--user', default='postgres', help='PostgreSQL user')
    parser.add_argument('--password', default='postgres', help='PostgreSQL password')
    parser.add_argument('--output-dir', default='/tmp/aqd_benchmarks', help='Output directory')
    parser.add_argument('--concurrent-workers', default=4, type=int, help='Concurrent workers for testing')
    parser.add_argument('--cost-threshold', default=10000, type=float, help='Cost threshold for routing method 1')
    
    args = parser.parse_args()
    
    # Configuration
    postgres_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    benchmark_config = {
        'concurrent_workers': args.concurrent_workers,
        'cost_threshold': args.cost_threshold,
        'lightgbm_model_path': '/tmp/aqd_model.txt',  # Would be set to actual model path
        'gnn_model_path': '/tmp/aqd_gnn_model.txt'
    }
    
    # Create test queries
    test_queries = create_test_queries()
    benchmark_config['queries'] = test_queries
    
    print("AQD Comprehensive Benchmark Runner")
    print(f"PostgreSQL: {args.host}:{args.port}/{args.database}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test queries: {len(test_queries)}")
    
    # Run benchmark
    try:
        benchmark_runner = AQDBenchmarkRunner(
            postgres_config=postgres_config,
            benchmark_config=benchmark_config,
            output_dir=args.output_dir
        )
        
        results = benchmark_runner.run_comprehensive_benchmark(test_queries)
        
        print("\nBenchmark completed successfully!")
        print(f"Check {args.output_dir} for detailed results and visualizations.")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()