#!/usr/bin/env python3
"""
Collect dual execution time data for training R-GIN model.
This script executes queries on both PostgreSQL and DuckDB engines,
collects execution times and query plans for training the routing model.
"""

import json
import psycopg2
import time
import os
import sys
import argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DualExecutionCollector:
    def __init__(self, db_config: Dict, output_dir: str = "dual_execution_data", flush_interval: int = 10):
        """
        Initialize the dual execution collector.
        
        Args:
            db_config: Database connection configuration
            output_dir: Directory to store collected data
            flush_interval: Number of queries before flushing to disk
        """
        self.db_config = db_config
        self.output_dir = output_dir
        self.postgres_conn = None
        self.duckdb_conn = None
        self.flush_interval = flush_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Stats tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.postgres_timeouts = 0
        self.duckdb_timeouts = 0
        
        # Buffer for incremental saves
        self.current_dataset = None
        self.results_buffer = []
        
    def connect(self):
        """Establish database connections."""
        try:
            # Connect to PostgreSQL
            self.postgres_conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                user=self.db_config.get('user', os.environ.get('USER')),
                password=self.db_config.get('password', ''),
                database=self.db_config.get('database', 'postgres')
            )
            self.postgres_conn.autocommit = True
            
            logger.info("Connected to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connections."""
        if self.postgres_conn:
            self.postgres_conn.close()
        if self.duckdb_conn:
            self.duckdb_conn.close()
    
    def execute_postgres_query(self, query: str, timeout: int = 30) -> Tuple[float, Optional[Dict]]:
        """
        Execute query on PostgreSQL and return execution time and plan.
        
        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds
            
        Returns:
            Tuple of (execution_time_ms, query_plan_json)
        """
        cur = self.postgres_conn.cursor()
        
        try:
            # Set statement timeout
            cur.execute(f"SET statement_timeout = {timeout * 1000}")
            
            # Get query plan (may fail if query is invalid)
            try:
                cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
                plan_json = cur.fetchone()[0][0]
            except psycopg2.errors.GroupingError as e:
                # Skip queries with GROUP BY errors
                logger.debug(f"Query has GROUP BY error, skipping: {str(e)[:100]}")
                return -1, None
            except Exception as e:
                logger.debug(f"Failed to get query plan: {str(e)[:100]}")
                plan_json = None
            
            # Execute query and measure time
            start_time = time.perf_counter()
            cur.execute(query)
            _ = cur.fetchall()  # Fetch all results to ensure complete execution
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            return execution_time_ms, plan_json
            
        except psycopg2.errors.QueryCanceled:
            logger.warning(f"PostgreSQL query timeout: {query[:100]}...")
            self.postgres_timeouts += 1
            return timeout * 1000, None
        except psycopg2.errors.GroupingError as e:
            # Skip queries with GROUP BY errors silently
            logger.debug(f"Query GROUP BY error: {str(e)[:100]}")
            return -1, None
        except Exception as e:
            logger.debug(f"PostgreSQL execution error: {str(e)[:100]}")
            return -1, None
        finally:
            cur.close()
    
    def execute_duckdb_query(self, query: str, timeout: int = 30) -> float:
        """
        Execute query on DuckDB through pg_duckdb extension.
        
        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds
            
        Returns:
            Execution time in milliseconds
        """
        cur = self.postgres_conn.cursor()
        
        try:
            # Set statement timeout
            cur.execute(f"SET statement_timeout = {timeout * 1000}")
            
            # Force DuckDB execution through pg_duckdb
            # This assumes pg_duckdb extension is installed and configured
            cur.execute("SET duckdb.force_execution = true")
            
            # Execute query and measure time
            start_time = time.perf_counter()
            cur.execute(query)
            _ = cur.fetchall()  # Fetch all results
            end_time = time.perf_counter()
            
            # Reset to PostgreSQL execution
            cur.execute("SET duckdb.force_execution = false")
            
            execution_time_ms = (end_time - start_time) * 1000
            
            return execution_time_ms
            
        except psycopg2.errors.QueryCanceled:
            logger.warning(f"DuckDB query timeout: {query[:100]}...")
            self.duckdb_timeouts += 1
            return timeout * 1000
        except Exception as e:
            logger.error(f"DuckDB execution error: {e}")
            return -1
        finally:
            cur.close()
    
    def collect_query_data(self, query: str, dataset: str, query_type: str, 
                          query_index: int) -> Optional[Dict]:
        """
        Collect execution data for a single query.
        
        Args:
            query: SQL query to execute
            dataset: Dataset name
            query_type: Query type (AP/TP)
            query_index: Query index
            
        Returns:
            Dictionary with collected data or None if failed
        """
        self.total_queries += 1
        
        try:
            # Execute on PostgreSQL
            postgres_time, plan_json = self.execute_postgres_query(query)
            if postgres_time < 0:
                self.failed_queries += 1
                return None
            
            # Execute on DuckDB
            duckdb_time = self.execute_duckdb_query(query)
            if duckdb_time < 0:
                self.failed_queries += 1
                return None
            
            # Calculate log time difference for training target
            # Positive value means PostgreSQL is slower (prefer DuckDB)
            # Negative value means DuckDB is slower (prefer PostgreSQL)
            if postgres_time > 0 and duckdb_time > 0:
                log_time_diff = np.log(postgres_time / duckdb_time)
            else:
                log_time_diff = 0.0
            
            # Collect all data
            result = {
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset,
                'query_type': query_type,
                'query_index': query_index,
                'query_text': query,
                'postgres_time_ms': postgres_time,
                'duckdb_time_ms': duckdb_time,
                'log_time_difference': log_time_diff,
                'postgres_plan': plan_json,
                'optimal_engine': 'duckdb' if duckdb_time < postgres_time else 'postgres',
                'speedup_ratio': max(postgres_time, duckdb_time) / min(postgres_time, duckdb_time) if min(postgres_time, duckdb_time) > 0 else 1.0
            }
            
            self.successful_queries += 1
            
            # Log progress
            if self.successful_queries % 10 == 0:
                logger.info(f"Collected {self.successful_queries}/{self.total_queries} queries")
                self.print_stats()
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting data for query: {e}")
            self.failed_queries += 1
            return None
    
    def collect_from_sql_files(self, ap_file: str, tp_file: str, dataset: str, 
                              sample_size: Optional[int] = None) -> List[Dict]:
        """
        Collect execution data from SQL query files with incremental flushing.
        
        Args:
            ap_file: Path to AP queries SQL file
            tp_file: Path to TP queries SQL file
            dataset: Dataset name
            sample_size: Optional number of queries to sample
            
        Returns:
            List of collected data dictionaries
        """
        self.current_dataset = dataset
        self.results_buffer = []
        all_results = []
        
        # Load and process AP queries
        if os.path.exists(ap_file):
            with open(ap_file, 'r') as f:
                ap_queries = [q.strip() for q in f.read().split('\n') if q.strip()]
            
            if sample_size and len(ap_queries) > sample_size // 2:
                ap_queries = random.sample(ap_queries, sample_size // 2)
            
            logger.info(f"Processing {len(ap_queries)} AP queries from {dataset}")
            for i, query in enumerate(ap_queries):
                result = self.collect_query_data(query, dataset, 'AP', i)
                if result:
                    self.results_buffer.append(result)
                    all_results.append(result)
                    
                    # Flush buffer if reached interval
                    if len(self.results_buffer) >= self.flush_interval:
                        self.flush_buffer(dataset)
        else:
            logger.warning(f"AP queries file not found: {ap_file}")
        
        # Load and process TP queries
        if os.path.exists(tp_file):
            with open(tp_file, 'r') as f:
                tp_queries = [q.strip() for q in f.read().split('\n') if q.strip()]
            
            if sample_size and len(tp_queries) > sample_size // 2:
                tp_queries = random.sample(tp_queries, sample_size // 2)
            
            logger.info(f"Processing {len(tp_queries)} TP queries from {dataset}")
            for i, query in enumerate(tp_queries):
                result = self.collect_query_data(query, dataset, 'TP', i)
                if result:
                    self.results_buffer.append(result)
                    all_results.append(result)
                    
                    # Flush buffer if reached interval
                    if len(self.results_buffer) >= self.flush_interval:
                        self.flush_buffer(dataset)
        else:
            logger.warning(f"TP queries file not found: {tp_file}")
        
        # Final flush for any remaining results
        if self.results_buffer:
            self.flush_buffer(dataset, final=True)
        
        return all_results
    
    def flush_buffer(self, dataset: str, final: bool = False):
        """Flush buffered results to disk."""
        if not self.results_buffer:
            return
            
        output_file = f"{dataset}.json"
        output_path = os.path.join(self.output_dir, output_file)
        
        # Load existing data if file exists
        existing_data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # Append new results
        all_data = existing_data + self.results_buffer
        
        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        logger.info(f"Flushed {len(self.results_buffer)} new results to {output_path} (total: {len(all_data)})")
        
        # Update summary if final flush
        if final:
            base_name = output_file.replace('.json', '')
            self.save_summary(all_data, f"{base_name}_summary.json")
        
        # Clear buffer
        self.results_buffer = []
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save collected results to file."""
        output_path = os.path.join(self.output_dir, output_file)
        
        # Save as JSON array for better structure
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
        
        # Also save summary statistics (remove .json first to avoid double extension)
        base_name = output_file.replace('.json', '')
        self.save_summary(results, f"{base_name}_summary.json")
    
    def save_summary(self, results: List[Dict], summary_file: str):
        """Save summary statistics."""
        if not results:
            return
        
        postgres_times = [r['postgres_time_ms'] for r in results if r['postgres_time_ms'] > 0]
        duckdb_times = [r['duckdb_time_ms'] for r in results if r['duckdb_time_ms'] > 0]
        speedups = [r['speedup_ratio'] for r in results]
        
        summary = {
            'total_queries': len(results),
            'postgres_preferred': sum(1 for r in results if r['optimal_engine'] == 'postgres'),
            'duckdb_preferred': sum(1 for r in results if r['optimal_engine'] == 'duckdb'),
            'postgres_stats': {
                'mean_ms': np.mean(postgres_times),
                'median_ms': np.median(postgres_times),
                'std_ms': np.std(postgres_times),
                'min_ms': np.min(postgres_times),
                'max_ms': np.max(postgres_times)
            },
            'duckdb_stats': {
                'mean_ms': np.mean(duckdb_times),
                'median_ms': np.median(duckdb_times),
                'std_ms': np.std(duckdb_times),
                'min_ms': np.min(duckdb_times),
                'max_ms': np.max(duckdb_times)
            },
            'speedup_stats': {
                'mean': np.mean(speedups),
                'median': np.median(speedups),
                'max': np.max(speedups)
            },
            'collection_stats': {
                'total_attempts': self.total_queries,
                'successful': self.successful_queries,
                'failed': self.failed_queries,
                'postgres_timeouts': self.postgres_timeouts,
                'duckdb_timeouts': self.duckdb_timeouts
            }
        }
        
        summary_path = os.path.join(self.output_dir, summary_file)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_path}")
    
    def print_stats(self):
        """Print collection statistics."""
        success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        logger.info(f"Stats: {self.successful_queries}/{self.total_queries} successful ({success_rate:.1f}%), "
                   f"PG timeouts: {self.postgres_timeouts}, DuckDB timeouts: {self.duckdb_timeouts}")


def main():
    parser = argparse.ArgumentParser(description='Collect dual execution time data')
    parser.add_argument('--queries-dir', default='benchmark_queries',
                       help='Directory containing query JSON files')
    parser.add_argument('--output-dir', default='dual_execution_data',
                       help='Output directory for collected data')
    parser.add_argument('--datasets', nargs='+',
                       default=['tpch_sf1', 'tpcds_sf1', 'Airline', 'financial', 'Carcinogenesis', 'Credit', 'employee', 'Hepatitis_std', 'geneea'],
                       help='Datasets to process')
    parser.add_argument('--sample-size', type=int,
                       help='Number of queries to sample per dataset')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Query timeout in seconds')
    parser.add_argument('--host', default='localhost',
                       help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=5432,
                       help='PostgreSQL port')
    parser.add_argument('--user', default=os.environ.get('USER'),
                       help='PostgreSQL user')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable GNN plan logging during collection')
    parser.add_argument('--flush-interval', type=int, default=10,
                       help='Number of queries before flushing to disk (default: 10)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process each dataset
    overall_stats = {
        'total_queries': 0,
        'postgres_preferred': 0,
        'duckdb_preferred': 0,
        'total_speedups': []
    }
    
    for dataset in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*60}")
        
        # Database configuration
        db_config = {
            'host': args.host,
            'port': args.port,
            'user': args.user,
            'database': dataset
        }
        
        # Initialize collector with flush interval
        collector = DualExecutionCollector(db_config, args.output_dir, args.flush_interval)
        
        try:
            # Connect to database
            collector.connect()
            
            # Enable GNN plan logging if requested
            if args.enable_logging:
                cur = collector.postgres_conn.cursor()
                cur.execute("SET gnn.plan_logging_enabled = on")
                cur.close()
                logger.info("Enabled GNN plan logging")
            
            # Look for query files in subdirectory
            dataset_dir = os.path.join(args.queries_dir, dataset)
            ap_file = os.path.join(dataset_dir, 'workload_ap_queries.sql')
            tp_file = os.path.join(dataset_dir, 'workload_tp_queries.sql')
            
            if not os.path.exists(dataset_dir):
                logger.warning(f"Query directory not found: {dataset_dir}")
                continue
            
            # Collect data (with incremental flushing)
            results = collector.collect_from_sql_files(ap_file, tp_file, dataset, args.sample_size)
            
            # Note: Results are already saved incrementally by flush_buffer
            # This final save is now redundant but kept for compatibility
            # collector.save_results(results, f"{dataset}.json")
            
            # Update overall statistics
            if results:
                overall_stats['total_queries'] += len(results)
                overall_stats['postgres_preferred'] += sum(1 for r in results if r['optimal_engine'] == 'postgres')
                overall_stats['duckdb_preferred'] += sum(1 for r in results if r['optimal_engine'] == 'duckdb')
                overall_stats['total_speedups'].extend([r['speedup_ratio'] for r in results])
            
            # Print final stats for this dataset
            logger.info(f"\nFinal stats for {dataset}:")
            collector.print_stats()
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
        finally:
            collector.disconnect()
    
    # Print overall summary (no combined file)
    if overall_stats['total_queries'] > 0:
        logger.info("\n" + "="*60)
        logger.info("OVERALL SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total queries collected: {overall_stats['total_queries']}")
        logger.info(f"PostgreSQL preferred: {overall_stats['postgres_preferred']} "
                   f"({overall_stats['postgres_preferred']/overall_stats['total_queries']*100:.1f}%)")
        logger.info(f"DuckDB preferred: {overall_stats['duckdb_preferred']} "
                   f"({overall_stats['duckdb_preferred']/overall_stats['total_queries']*100:.1f}%)")
        
        if overall_stats['total_speedups']:
            logger.info(f"Average speedup potential: {np.mean(overall_stats['total_speedups']):.2f}x")
            logger.info(f"Maximum speedup observed: {np.max(overall_stats['total_speedups']):.2f}x")


if __name__ == '__main__':
    main()