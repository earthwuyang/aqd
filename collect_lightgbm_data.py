#!/usr/bin/env python3
"""
Collect lightweight feature data for training LightGBM routing model.
This script executes queries on both PostgreSQL and DuckDB engines,
extracts simple features, and saves to CSV format for incremental training.
"""

import csv
import psycopg2
import time
import os
import sys
import argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class LightGBMDataCollector:
    def __init__(self, db_config: Dict, output_dir: str = "lightgbm_training_data", flush_interval: int = 50):
        """
        Initialize the LightGBM data collector.
        
        Args:
            db_config: Database connection configuration
            output_dir: Directory to store collected CSV data
            flush_interval: Number of queries before flushing to disk
        """
        self.db_config = db_config
        self.output_dir = output_dir
        self.postgres_conn = None
        self.flush_interval = flush_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Stats tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.postgres_timeouts = 0
        self.duckdb_timeouts = 0
        
        # CSV writers and files
        self.csv_files = {}
        self.csv_writers = {}
        
        # Define CSV columns - matching LightGBM feature extraction
        self.csv_columns = [
            'timestamp',
            'dataset',
            'query_type',
            'query_index',
            'query_hash',
            # Basic query structure features
            'num_tables',
            'num_joins', 
            'query_depth',
            'complexity_score',
            # Boolean features
            'has_aggregates',
            'has_group_by',
            'has_order_by',
            'has_limit',
            'has_distinct',
            'has_window_functions',
            # Join features
            'has_outer_joins',
            'estimated_join_complexity',
            # Subquery features
            'has_subqueries',
            'has_correlated_subqueries',
            # Table characteristics
            'has_large_tables',
            'all_tables_small',
            # Operation complexity
            'has_complex_expressions',
            'has_user_functions',
            'has_text_operations',
            'has_numeric_heavy_ops',
            # Aggregate function count
            'num_aggregate_funcs',
            # Pattern scores
            'analytical_pattern',
            'transactional_pattern',
            'etl_pattern',
            # Command type
            'command_type',
            # Execution times and target
            'postgres_time_ms',
            'duckdb_time_ms',
            'log_time_difference',
            'optimal_engine',
            'speedup_ratio'
        ]
        
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
        """Close database connections and CSV files."""
        if self.postgres_conn:
            self.postgres_conn.close()
        
        # Close CSV files
        for file_handle in self.csv_files.values():
            file_handle.close()
    
    def get_csv_writer(self, dataset: str):
        """Get or create CSV writer for dataset."""
        if dataset not in self.csv_writers:
            filename = f"{dataset}_features.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(filepath)
            
            # Open file in append mode
            file_handle = open(filepath, 'a', newline='', encoding='utf-8')
            csv_writer = csv.writer(file_handle)
            
            # Write header if new file
            if not file_exists:
                csv_writer.writerow(self.csv_columns)
                logger.info(f"Created new CSV file: {filepath}")
            else:
                logger.info(f"Appending to existing CSV file: {filepath}")
            
            self.csv_files[dataset] = file_handle
            self.csv_writers[dataset] = csv_writer
            
        return self.csv_writers[dataset]
    
    def extract_lightweight_features(self, query: str) -> Dict:
        """
        Extract lightweight features from query using PostgreSQL parser.
        This mimics what the query_analyzer.c does in the kernel.
        """
        cur = self.postgres_conn.cursor()
        features = {}
        
        try:
            # Enable feature extraction mode
            cur.execute("SET gnn.feature_extraction_enabled = on")
            
            # Get EXPLAIN output to trigger feature extraction
            # This will internally use the query analyzer
            cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
            plan_result = cur.fetchone()
            
            # For now, we'll extract simple heuristic features
            # In a full implementation, this would call the C function
            features.update(self._extract_heuristic_features(query))
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {str(e)[:200]}")
            # Return default features if extraction fails
            return self._get_default_features()
        finally:
            cur.close()
    
    def _extract_heuristic_features(self, query: str) -> Dict:
        """Extract simple heuristic features from query text."""
        query_upper = query.upper()
        
        # Count basic elements
        num_joins = (query_upper.count(' JOIN ') + 
                    query_upper.count(' LEFT ') + 
                    query_upper.count(' RIGHT ') + 
                    query_upper.count(' FULL ') +
                    query_upper.count(' INNER '))
        
        num_tables = max(1, query_upper.count(' FROM ') + num_joins)
        
        # Extract features
        features = {
            'num_tables': num_tables,
            'num_joins': num_joins,
            'query_depth': query_upper.count('(') + 1,  # Rough nesting estimate
            'has_aggregates': any(agg in query_upper for agg in ['SUM(', 'COUNT(', 'AVG(', 'MIN(', 'MAX(']),
            'has_group_by': 'GROUP BY' in query_upper,
            'has_order_by': 'ORDER BY' in query_upper,
            'has_limit': 'LIMIT ' in query_upper,
            'has_distinct': 'DISTINCT' in query_upper,
            'has_window_functions': 'OVER (' in query_upper,
            'has_outer_joins': any(join in query_upper for join in ['LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']),
            'has_subqueries': query_upper.count('SELECT') > 1,
            'has_correlated_subqueries': 'EXISTS (' in query_upper,
            'has_complex_expressions': any(op in query_upper for op in ['CASE ', 'COALESCE', 'CAST(']),
            'has_user_functions': '()' in query and any(func in query_upper for func in ['SUBSTRING', 'REGEXP', 'EXTRACT']),
            'has_text_operations': any(op in query_upper for op in ['LIKE ', 'ILIKE ', '~', 'REGEXP']),
            'has_numeric_heavy_ops': any(op in query_upper for op in ['*', '/', '+', '-']) and 'SELECT' in query_upper,
            'num_aggregate_funcs': sum(query_upper.count(agg) for agg in ['SUM(', 'COUNT(', 'AVG(', 'MIN(', 'MAX(']),
        }
        
        # Estimate table characteristics (simplified)
        features['has_large_tables'] = num_tables > 3  # Heuristic
        features['all_tables_small'] = num_tables <= 2  # Heuristic
        
        # Estimate complexity
        complexity = (num_tables * 2 + num_joins * 3 + 
                     features['num_aggregate_funcs'] * 2 +
                     (5 if features['has_window_functions'] else 0))
        features['complexity_score'] = complexity
        
        # Pattern detection
        features['analytical_pattern'] = (features['has_aggregates'] and 
                                        (features['has_group_by'] or features['has_window_functions']))
        features['transactional_pattern'] = (num_tables <= 2 and num_joins <= 1 and 
                                           not features['has_aggregates'])
        features['etl_pattern'] = (num_tables > 5 or 'INSERT' in query_upper or 'UPDATE' in query_upper)
        
        # Command type (simplified)
        if 'SELECT' in query_upper:
            features['command_type'] = 1  # CMD_SELECT
        elif 'INSERT' in query_upper:
            features['command_type'] = 3  # CMD_INSERT  
        elif 'UPDATE' in query_upper:
            features['command_type'] = 2  # CMD_UPDATE
        elif 'DELETE' in query_upper:
            features['command_type'] = 4  # CMD_DELETE
        else:
            features['command_type'] = 0  # CMD_UNKNOWN
        
        # Set remaining features with estimates
        features['estimated_join_complexity'] = num_joins
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Return default feature values."""
        return {col: 0 for col in self.csv_columns[5:30]}  # Feature columns
    
    def execute_postgres_query(self, query: str, timeout: int = 120) -> float:
        """Execute query on PostgreSQL and return execution time."""
        cur = self.postgres_conn.cursor()
        
        try:
            # Set statement timeout
            cur.execute(f"SET statement_timeout = {timeout * 1000}")
            
            # Execute query and measure time
            start_time = time.perf_counter()
            cur.execute(query)
            _ = cur.fetchall()  # Fetch all results
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            return execution_time_ms
            
        except psycopg2.errors.QueryCanceled:
            logger.warning(f"PostgreSQL query timeout: {query[:100]}...")
            self.postgres_timeouts += 1
            return timeout * 1000
        except psycopg2.errors.GroupingError as e:
            logger.debug(f"Query GROUP BY error: {str(e)[:100]}")
            return -1
        except Exception as e:
            logger.debug(f"PostgreSQL execution error: {str(e)[:100]}")
            return -1
        finally:
            cur.close()
    
    def execute_duckdb_query(self, query: str, timeout: int = 120) -> float:
        """Execute query on DuckDB through pg_duckdb extension."""
        cur = self.postgres_conn.cursor()
        
        try:
            # Set statement timeout
            cur.execute(f"SET statement_timeout = {timeout * 1000}")
            
            # Force DuckDB execution
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
            logger.debug(f"DuckDB execution error: {str(e)[:100]}")
            return -1
        finally:
            cur.close()
    
    def collect_query_data(self, query: str, dataset: str, query_type: str, 
                          query_index: int) -> bool:
        """
        Collect lightweight features and execution data for a single query.
        
        Returns:
            True if successful, False otherwise
        """
        self.total_queries += 1
        
        try:
            # Extract lightweight features
            features = self.extract_lightweight_features(query)
            
            # Execute on both engines
            postgres_time = self.execute_postgres_query(query)
            if postgres_time < 0:
                self.failed_queries += 1
                return False
            
            duckdb_time = self.execute_duckdb_query(query)
            if duckdb_time < 0:
                self.failed_queries += 1
                return False
            
            # Calculate target values
            if postgres_time > 0 and duckdb_time > 0:
                log_time_diff = np.log(postgres_time / duckdb_time)
                optimal_engine = 'duckdb' if duckdb_time < postgres_time else 'postgres'
                speedup_ratio = max(postgres_time, duckdb_time) / min(postgres_time, duckdb_time)
            else:
                log_time_diff = 0.0
                optimal_engine = 'postgres'
                speedup_ratio = 1.0
            
            # Create CSV row
            row_data = [
                datetime.now().isoformat(),
                dataset,
                query_type,
                query_index,
                hash(query) % 1000000,  # Simple query hash
            ]
            
            # Add feature values in order
            for col in self.csv_columns[5:30]:  # Feature columns
                row_data.append(features.get(col, 0))
            
            # Add execution results
            row_data.extend([
                postgres_time,
                duckdb_time,
                log_time_diff,
                optimal_engine,
                speedup_ratio
            ])
            
            # Write to CSV
            csv_writer = self.get_csv_writer(dataset)
            csv_writer.writerow(row_data)
            
            # Flush periodically
            if self.successful_queries % self.flush_interval == 0:
                self.csv_files[dataset].flush()
            
            self.successful_queries += 1
            
            # Log progress
            if self.successful_queries % 25 == 0:
                logger.info(f"Collected {self.successful_queries}/{self.total_queries} queries")
                self.print_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data for query: {e}")
            self.failed_queries += 1
            return False
    
    def collect_from_sql_files(self, ap_file: str, tp_file: str, dataset: str, 
                              sample_size: Optional[int] = None):
        """
        Collect feature data from SQL query files.
        
        Args:
            ap_file: Path to AP queries SQL file
            tp_file: Path to TP queries SQL file
            dataset: Dataset name
            sample_size: Optional number of queries to sample
        """
        # Process AP queries
        if os.path.exists(ap_file):
            with open(ap_file, 'r') as f:
                ap_queries = [q.strip() for q in f.read().split('\n') if q.strip()]
            
            if sample_size and len(ap_queries) > sample_size // 2:
                ap_queries = random.sample(ap_queries, sample_size // 2)
            
            logger.info(f"Processing {len(ap_queries)} AP queries from {dataset}")
            for i, query in enumerate(ap_queries):
                self.collect_query_data(query, dataset, 'AP', i)
        else:
            logger.warning(f"AP queries file not found: {ap_file}")
        
        # Process TP queries
        if os.path.exists(tp_file):
            with open(tp_file, 'r') as f:
                tp_queries = [q.strip() for q in f.read().split('\n') if q.strip()]
            
            if sample_size and len(tp_queries) > sample_size // 2:
                tp_queries = random.sample(tp_queries, sample_size // 2)
            
            logger.info(f"Processing {len(tp_queries)} TP queries from {dataset}")
            for i, query in enumerate(tp_queries):
                self.collect_query_data(query, dataset, 'TP', i)
        else:
            logger.warning(f"TP queries file not found: {tp_file}")
        
        # Final flush
        if dataset in self.csv_files:
            self.csv_files[dataset].flush()
    
    def print_stats(self):
        """Print collection statistics."""
        success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        logger.info(f"Stats: {self.successful_queries}/{self.total_queries} successful ({success_rate:.1f}%), "
                   f"PG timeouts: {self.postgres_timeouts}, DuckDB timeouts: {self.duckdb_timeouts}")


def main():
    parser = argparse.ArgumentParser(description='Collect lightweight features for LightGBM training')
    parser.add_argument('--queries-dir', default='benchmark_queries',
                       help='Directory containing query SQL files')
    parser.add_argument('--output-dir', default='lightgbm_training_data',
                       help='Output directory for CSV files')
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
    parser.add_argument('--flush-interval', type=int, default=50,
                       help='Number of queries before flushing to disk (default: 50)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process each dataset
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
        
        # Initialize collector
        collector = LightGBMDataCollector(db_config, args.output_dir, args.flush_interval)
        
        try:
            # Connect to database
            collector.connect()
            
            # Look for query files
            dataset_dir = os.path.join(args.queries_dir, dataset)
            ap_file = os.path.join(dataset_dir, 'workload_ap_queries.sql')
            tp_file = os.path.join(dataset_dir, 'workload_tp_queries.sql')
            
            if not os.path.exists(dataset_dir):
                logger.warning(f"Query directory not found: {dataset_dir}")
                continue
            
            # Collect data to CSV
            collector.collect_from_sql_files(ap_file, tp_file, dataset, args.sample_size)
            
            # Print final stats
            logger.info(f"\nFinal stats for {dataset}:")
            collector.print_stats()
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
        finally:
            collector.disconnect()
    
    logger.info("\nData collection completed. CSV files are ready for LightGBM training.")


if __name__ == '__main__':
    main()