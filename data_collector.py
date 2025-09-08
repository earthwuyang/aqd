#!/usr/bin/env python3
"""
AQD Data Collection Pipeline

Collects execution metrics from both PostgreSQL and DuckDB for ML training.
Executes the same queries on both engines and records performance differences.
"""

import os
import sys
import time
import json
import csv
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import subprocess
import logging
import hashlib

import psycopg2
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import our benchmark query generators
from generate_benchmark_queries import (
    GenQueryAP, GenQueryTP, DatabaseIntrospector,
    POSTGRESQL_CONFIG, DUCKDB_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AQDDataCollector:
    """
    Collects training data by executing queries on both PostgreSQL and DuckDB
    and recording execution metrics along with query features.
    """
    
    def __init__(self, 
                 postgres_config: Dict,
                 duckdb_path: str,
                 output_dir: str = "/tmp/aqd_training_data"):
        self.postgres_config = postgres_config
        self.duckdb_path = duckdb_path
        self.output_dir = output_dir
        
        # Execution state
        self.pg_conn = None
        self.duck_conn = None
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'postgres_faster': 0,
            'duckdb_faster': 0,
            'start_time': datetime.now()
        }
        
        # Collected data
        self.training_data = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def connect_databases(self):
        """Connect to both PostgreSQL and DuckDB"""
        try:
            # Connect to PostgreSQL with modified config for feature logging
            pg_config = self.postgres_config.copy()
            self.pg_conn = psycopg2.connect(**pg_config)
            self.pg_conn.autocommit = True
            
            # Enable AQD feature logging
            with self.pg_conn.cursor() as cur:
                cur.execute("SET aqd.enable_feature_logging = true;")
                cur.execute("SET aqd.feature_log_path = '/tmp/aqd_features.csv';")
                cur.execute("SET aqd.routing_method = 0;")  # Use default for data collection
                
            logger.info("Connected to PostgreSQL with AQD feature logging enabled")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
            
        try:
            # Connect to DuckDB
            self.duck_conn = duckdb.connect(self.duckdb_path)
            logger.info(f"Connected to DuckDB: {self.duckdb_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise
    
    def disconnect_databases(self):
        """Disconnect from databases"""
        if self.pg_conn:
            self.pg_conn.close()
            self.pg_conn = None
            
        if self.duck_conn:
            self.duck_conn.close()
            self.duck_conn = None
            
        logger.info("Disconnected from databases")
    
    def execute_query_postgres(self, query: str, timeout: int = 60) -> Dict:
        """Execute query on PostgreSQL and collect metrics"""
        start_time = time.time()
        
        try:
            with self.pg_conn.cursor() as cur:
                # Execute query with timing
                cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query)
                explain_result = cur.fetchone()[0]
                
                execution_time = explain_result[0]['Execution Time']
                planning_time = explain_result[0]['Planning Time']
                
                # Extract buffer statistics
                buffer_stats = {
                    'shared_hit': 0,
                    'shared_read': 0,
                    'shared_written': 0,
                    'local_hit': 0,
                    'local_read': 0,
                    'local_written': 0,
                    'temp_read': 0,
                    'temp_written': 0
                }
                
                # Parse buffer usage from explain output
                def extract_buffers(node):
                    if 'Shared Hit Blocks' in node:
                        buffer_stats['shared_hit'] += node['Shared Hit Blocks']
                    if 'Shared Read Blocks' in node:
                        buffer_stats['shared_read'] += node['Shared Read Blocks']
                    if 'Shared Written Blocks' in node:
                        buffer_stats['shared_written'] += node['Shared Written Blocks']
                    
                    # Recursively process child nodes
                    for child_key in ['Plans', 'Plan']:
                        if child_key in node:
                            if isinstance(node[child_key], list):
                                for child in node[child_key]:
                                    extract_buffers(child)
                            else:
                                extract_buffers(node[child_key])
                
                extract_buffers(explain_result[0]['Plan'])
                
                return {
                    'success': True,
                    'execution_time_ms': execution_time,
                    'planning_time_ms': planning_time,
                    'total_time_ms': execution_time + planning_time,
                    'buffer_stats': buffer_stats,
                    'explain_json': explain_result,
                    'error': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'execution_time_ms': None,
                'planning_time_ms': None,
                'total_time_ms': time.time() - start_time,
                'buffer_stats': {},
                'explain_json': None,
                'error': str(e)
            }
    
    def execute_query_duckdb(self, query: str, timeout: int = 60) -> Dict:
        """Execute query on DuckDB and collect metrics"""
        start_time = time.time()
        
        try:
            # Enable profiling
            self.duck_conn.execute("PRAGMA enable_profiling;")
            
            # Execute query
            query_start = time.time()
            result = self.duck_conn.execute(query).fetchall()
            query_end = time.time()
            
            execution_time_ms = (query_end - query_start) * 1000
            
            # Get profiling information
            prof_info = self.duck_conn.execute("PRAGMA profiling_output;").fetchone()
            
            return {
                'success': True,
                'execution_time_ms': execution_time_ms,
                'planning_time_ms': 0,  # DuckDB doesn't separate planning time
                'total_time_ms': execution_time_ms,
                'result_rows': len(result) if result else 0,
                'profiling_info': prof_info[0] if prof_info else None,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'execution_time_ms': None,
                'planning_time_ms': None,
                'total_time_ms': (time.time() - start_time) * 1000,
                'result_rows': 0,
                'profiling_info': None,
                'error': str(e)
            }
    
    def collect_query_features(self, query: str) -> Dict:
        """Extract features from PostgreSQL feature logger"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        
        try:
            # Read the latest features from AQD feature log
            features_file = "/tmp/aqd_features.csv"
            if os.path.exists(features_file):
                df = pd.read_csv(features_file)
                
                # Find the row corresponding to our query
                matching_rows = df[df['query_hash'] == query_hash]
                if len(matching_rows) > 0:
                    latest_row = matching_rows.iloc[-1]
                    return latest_row.to_dict()
                    
        except Exception as e:
            logger.warning(f"Failed to read query features: {e}")
        
        return {}
    
    def collect_single_query(self, query: str, query_id: str, dataset: str) -> Optional[Dict]:
        """Collect data for a single query on both engines"""
        logger.debug(f"Collecting data for query {query_id} on dataset {dataset}")
        
        # Execute on PostgreSQL
        pg_result = self.execute_query_postgres(query)
        
        # Execute on DuckDB (same query)
        duck_result = self.execute_query_duckdb(query)
        
        # Collect features
        features = self.collect_query_features(query)
        
        # Check if both executions were successful
        if not pg_result['success'] and not duck_result['success']:
            logger.warning(f"Both engines failed for query {query_id}")
            self.stats['failed_queries'] += 1
            return None
        
        # Calculate performance gap (log-transformed as per paper)
        if pg_result['success'] and duck_result['success']:
            pg_time = pg_result['total_time_ms']
            duck_time = duck_result['total_time_ms']
            
            if pg_time > 0 and duck_time > 0:
                # Log-transformed gap as in AQD paper
                log_gap = np.log(pg_time / duck_time)
            else:
                log_gap = 0.0
        else:
            # If one failed, assign large penalty
            if pg_result['success']:
                log_gap = -10.0  # PostgreSQL much better
            else:
                log_gap = 10.0   # DuckDB much better
        
        # Compile training record
        training_record = {
            'query_id': query_id,
            'dataset': dataset,
            'query_text': query[:500],  # Truncate for storage
            'query_hash': hashlib.sha256(query.encode()).hexdigest(),
            'timestamp': datetime.now().isoformat(),
            
            # Target variable (what we want to predict)
            'log_time_gap': log_gap,
            
            # PostgreSQL metrics
            'pg_success': pg_result['success'],
            'pg_execution_time_ms': pg_result.get('execution_time_ms'),
            'pg_planning_time_ms': pg_result.get('planning_time_ms'),
            'pg_total_time_ms': pg_result.get('total_time_ms'),
            'pg_error': pg_result.get('error'),
            
            # DuckDB metrics
            'duck_success': duck_result['success'],
            'duck_execution_time_ms': duck_result.get('execution_time_ms'),
            'duck_total_time_ms': duck_result.get('total_time_ms'),
            'duck_error': duck_result.get('error'),
            
            # Features from PostgreSQL AQD feature logger
            'features': features
        }
        
        # Update statistics
        self.stats['total_queries'] += 1
        if pg_result['success'] and duck_result['success']:
            self.stats['successful_queries'] += 1
            if pg_result['total_time_ms'] < duck_result['total_time_ms']:
                self.stats['postgres_faster'] += 1
            else:
                self.stats['duckdb_faster'] += 1
        
        return training_record
    
    def collect_dataset_queries(self, 
                               queries: List[Tuple[str, str]], 
                               dataset: str,
                               max_queries: Optional[int] = None) -> List[Dict]:
        """Collect data for all queries in a dataset"""
        logger.info(f"Collecting data for dataset '{dataset}' ({len(queries)} queries)")
        
        dataset_records = []
        query_count = min(len(queries), max_queries) if max_queries else len(queries)
        
        for i, (query_id, query) in enumerate(tqdm(queries[:query_count], desc=f"Dataset {dataset}")):
            try:
                record = self.collect_single_query(query, f"{dataset}_{i:06d}", dataset)
                if record:
                    dataset_records.append(record)
                    self.training_data.append(record)
                
                # Periodic save
                if len(dataset_records) % 100 == 0:
                    self.save_intermediate_data(dataset)
                    
            except KeyboardInterrupt:
                logger.info("Collection interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error collecting query {i}: {e}")
                continue
        
        logger.info(f"Collected {len(dataset_records)} records for dataset '{dataset}'")
        return dataset_records
    
    def save_intermediate_data(self, dataset_suffix: str = ""):
        """Save intermediate results to prevent data loss"""
        if not self.training_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aqd_training_data_{dataset_suffix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
            
        logger.info(f"Saved {len(self.training_data)} records to {filepath}")
    
    def export_training_data(self) -> str:
        """Export collected data in format suitable for ML training"""
        if not self.training_data:
            logger.warning("No training data to export")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export as JSON
        json_file = os.path.join(self.output_dir, f"aqd_training_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        
        # Export as CSV for easier analysis
        csv_file = os.path.join(self.output_dir, f"aqd_training_{timestamp}.csv")
        
        # Flatten the data for CSV export
        flattened_data = []
        for record in self.training_data:
            flat_record = {
                'query_id': record['query_id'],
                'dataset': record['dataset'],
                'query_hash': record['query_hash'],
                'log_time_gap': record['log_time_gap'],
                'pg_success': record['pg_success'],
                'pg_total_time_ms': record['pg_total_time_ms'],
                'duck_success': record['duck_success'],
                'duck_total_time_ms': record['duck_total_time_ms'],
            }
            
            # Add features as separate columns
            features = record.get('features', {})
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    flat_record[f'feature_{key}'] = value
            
            flattened_data.append(flat_record)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(csv_file, index=False)
        
        # Export statistics
        stats_file = os.path.join(self.output_dir, f"aqd_collection_stats_{timestamp}.json")
        collection_time = (datetime.now() - self.stats['start_time']).total_seconds()
        
        final_stats = self.stats.copy()
        final_stats['collection_duration_seconds'] = collection_time
        final_stats['collection_rate_queries_per_second'] = self.stats['total_queries'] / collection_time
        final_stats['success_rate'] = self.stats['successful_queries'] / max(self.stats['total_queries'], 1)
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2, default=str)
        
        logger.info(f"Exported training data:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Stats: {stats_file}")
        
        return json_file
    
    def print_statistics(self):
        """Print collection statistics"""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        print(f"\n=== AQD Data Collection Statistics ===")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total queries: {self.stats['total_queries']}")
        print(f"Successful queries: {self.stats['successful_queries']}")
        print(f"Failed queries: {self.stats['failed_queries']}")
        print(f"Success rate: {self.stats['successful_queries']/max(self.stats['total_queries'],1)*100:.1f}%")
        print(f"PostgreSQL faster: {self.stats['postgres_faster']} ({self.stats['postgres_faster']/max(self.stats['successful_queries'],1)*100:.1f}%)")
        print(f"DuckDB faster: {self.stats['duckdb_faster']} ({self.stats['duckdb_faster']/max(self.stats['successful_queries'],1)*100:.1f}%)")
        print(f"Collection rate: {self.stats['total_queries']/duration:.2f} queries/sec")
        print(f"Training records: {len(self.training_data)}")


def main():
    """Main data collection pipeline"""
    # Configuration
    DATASETS_TO_COLLECT = [
        'accidents', 'financial', 'Basketball_men', 'northwind', 
        'sakila', 'employees', 'world', 'imdb_small'
    ]
    QUERIES_PER_DATASET = 2000  # 1000 TP + 1000 AP queries
    
    # Initialize data collector
    collector = AQDDataCollector(
        postgres_config=POSTGRESQL_CONFIG,
        duckdb_path=DUCKDB_PATH,
        output_dir="/tmp/aqd_training_data"
    )
    
    try:
        # Connect to databases
        collector.connect_databases()
        
        # Initialize database introspector
        introspector = DatabaseIntrospector()
        introspector.connect_duckdb()
        
        # Collect data for each dataset
        for dataset in DATASETS_TO_COLLECT:
            logger.info(f"Starting collection for dataset: {dataset}")
            
            try:
                # Generate queries for this dataset
                logger.info(f"Generating queries for {dataset}")
                
                # Get schema information
                tables = introspector.get_schema_tables(dataset)
                if not tables:
                    logger.warning(f"No tables found for dataset {dataset}, skipping")
                    continue
                
                # Generate AP queries (analytical)
                ap_queries = []
                for i in range(QUERIES_PER_DATASET // 2):
                    # This would use the actual query generator
                    # For now, create simple placeholder queries
                    query = f"SELECT COUNT(*), AVG(CASE WHEN RANDOM() > 0.5 THEN 1 ELSE 0 END) FROM {dataset}.{tables[0]} LIMIT 1000"
                    ap_queries.append((f"ap_{i}", query))
                
                # Generate TP queries (transactional)
                tp_queries = []
                for i in range(QUERIES_PER_DATASET // 2):
                    query = f"SELECT * FROM {dataset}.{tables[0]} WHERE RANDOM() > 0.9 LIMIT 10"
                    tp_queries.append((f"tp_{i}", query))
                
                all_queries = ap_queries + tp_queries
                
                # Collect data for this dataset
                dataset_records = collector.collect_dataset_queries(all_queries, dataset)
                logger.info(f"Collected {len(dataset_records)} records for {dataset}")
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {e}")
                continue
        
        # Export all collected data
        output_file = collector.export_training_data()
        logger.info(f"Data collection complete. Training data exported to: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
    finally:
        collector.print_statistics()
        collector.disconnect_databases()


if __name__ == "__main__":
    main()