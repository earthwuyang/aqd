#!/usr/bin/env python3
"""
Dual Execution Data Collection for AQD Training
Executes queries on both PostgreSQL and DuckDB, collecting:
1. PostgreSQL execution time and AQD features
2. DuckDB execution time
3. Training data for ML models

Based on methodology from AQD paper implementation.
"""

import os
import sys
import time
import json
import glob
import psycopg2
import duckdb
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import threading
import queue
import traceback
from contextlib import contextmanager

# Configuration
POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'wuy'
}
from pathlib import Path
# Resolve paths relative to this repository
BASE_DIR = Path(__file__).resolve().parent
DUCKDB_PATH = str(BASE_DIR / 'data' / 'benchmark_datasets.db')
QUERIES_BASE_DIR = str(BASE_DIR / 'data' / 'benchmark_queries')
OUTPUT_DIR = str(BASE_DIR / 'data' / 'execution_data')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExecutionResult:
    """Container for query execution results"""
    def __init__(self, query_text, dataset, query_type, query_index):
        self.query_text = query_text
        self.dataset = dataset
        self.query_type = query_type  # 'AP' or 'TP'
        self.query_index = query_index
        
        # Execution metrics
        self.postgres_time = None
        self.duckdb_time = None
        self.postgres_error = None
        self.duckdb_error = None
        
        # AQD features from PostgreSQL
        self.aqd_features = {}
        self.postgres_plan = None
        
        # Status
        self.executed_postgres = False
        self.executed_duckdb = False
        
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'query_text': self.query_text,
            'dataset': self.dataset,
            'query_type': self.query_type,
            'query_index': self.query_index,
            'postgres_time': self.postgres_time,
            'duckdb_time': self.duckdb_time,
            'time_ratio': self.duckdb_time / self.postgres_time if self.postgres_time and self.duckdb_time and self.postgres_time > 0 else None,
            'log_time_difference': np.log(self.postgres_time / self.duckdb_time) if self.postgres_time and self.duckdb_time and self.duckdb_time > 0 else None,
            'postgres_error': self.postgres_error,
            'duckdb_error': self.duckdb_error,
            'aqd_features': self.aqd_features,
            'postgres_plan': self.postgres_plan,
            'executed_postgres': self.executed_postgres,
            'executed_duckdb': self.executed_duckdb,
            'timestamp': datetime.now().isoformat()
        }


class DualExecutionCollector:
    """Collects dual execution data for AQD training"""
    
    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds
        self.pg_conn = None
        self.duck_conn = None
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'postgres_successes': 0,
            'duckdb_successes': 0,
            'both_successful': 0,
            'postgres_timeouts': 0,
            'duckdb_timeouts': 0,
            'start_time': datetime.now()
        }
        
    def connect_databases(self):
        """Connect to both PostgreSQL and DuckDB"""
        try:
            # Connect to PostgreSQL with AQD features enabled
            self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
            self.pg_conn.autocommit = True
            
            # Enable AQD feature logging (CSV)
            cursor = self.pg_conn.cursor()
            cursor.execute("SET aqd.enable_feature_logging = on;")
            os.makedirs(BASE_DIR / 'data' / 'execution_data', exist_ok=True)
            feature_csv = str(BASE_DIR / 'data' / 'execution_data' / 'aqd_features.csv')
            plan_jsonl = str(BASE_DIR / 'data' / 'execution_data' / 'aqd_plans.jsonl')
            cursor.execute(f"SET aqd.feature_log_path = '{feature_csv}';")
            try:
                cursor.execute(f"SET aqd.plan_log_path = '{plan_jsonl}';")
                cursor.execute("SET aqd.enable_plan_logging = on;")
            except Exception:
                # Older kernels may not have plan logging GUCs; ignore
                pass
            cursor.execute("SET aqd.log_format = 0;")
            cursor.close()
            
            logger.info("Connected to PostgreSQL with AQD enabled")
            
            # Connect to DuckDB
            self.duck_conn = duckdb.connect(DUCKDB_PATH)
            logger.info("Connected to DuckDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    @contextmanager
    def timeout_context(self, seconds):
        """Context manager for query timeouts"""
        def timeout_handler():
            raise TimeoutError(f"Query timeout after {seconds} seconds")
        
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    
    def modify_query_for_schema(self, query_text, dataset_name):
        """Modify query to use schema-qualified table names and handle reserved keywords"""
        import re

        reserved_table_keywords = ['order', 'user', 'table', 'index', 'view', 'trigger', 'function', 'procedure']

        q = query_text
        # 1) Schema-qualify table names in common clauses
        table_patterns = [
            (r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', rf'FROM "{dataset_name}".\1'),
            (r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', rf'JOIN "{dataset_name}".\1'),
            (r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', rf'INTO "{dataset_name}".\1'),
            (r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', rf'UPDATE "{dataset_name}".\1'),
            (r'\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', rf'DELETE FROM "{dataset_name}".\1'),
        ]
        for pattern, replacement in table_patterns:
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        # 2) Quote reserved keywords when used as table identifiers globally (SELECT list, joins, predicates)
        for keyword in reserved_table_keywords:
            # schema-qualified case: "schema".order. -> "schema"."order".
            q = re.sub(rf'("{dataset_name}")\.({keyword})\.', rf'\1."{keyword}".', q, flags=re.IGNORECASE)
            # unqualified case: order. -> "order".
            q = re.sub(rf'(?<!")\b({keyword})\.', rf'"{keyword}".', q, flags=re.IGNORECASE)

        return q
    
    def execute_postgres_query(self, query_text, dataset_name):
        """Execute query on PostgreSQL and extract AQD features"""
        # Ensure connection to the correct PostgreSQL database (each dataset in its own DB)
        try:
            if self.pg_conn:
                try:
                    self.pg_conn.close()
                except Exception:
                    pass
            pg_cfg = POSTGRESQL_CONFIG.copy()
            pg_cfg['database'] = dataset_name
            self.pg_conn = psycopg2.connect(**pg_cfg)
            self.pg_conn.autocommit = True
            # Enable AQD feature logging into repo execution_data directory
            cursor = self.pg_conn.cursor()
            cursor.execute("SET aqd.enable_feature_logging = on;")
            feature_csv = str(BASE_DIR / 'data' / 'execution_data' / 'aqd_features.csv')
            plan_jsonl = str(BASE_DIR / 'data' / 'execution_data' / 'aqd_plans.jsonl')
            cursor.execute(f"SET aqd.feature_log_path = '{feature_csv}';")
            try:
                cursor.execute(f"SET aqd.plan_log_path = '{plan_jsonl}';")
                cursor.execute("SET aqd.enable_plan_logging = on;")
            except Exception:
                pass
            cursor.execute("SET aqd.log_format = 0;")
            cursor.close()
        except Exception as e:
            return None, f"Failed to connect to PostgreSQL DB '{dataset_name}': {e}", None, None
        
        try:
            with self.timeout_context(self.timeout_seconds):
                cursor = self.pg_conn.cursor()
                
                # Note: CSV log is append-only; we'll read the last line after execution
                
                # Quote reserved table keywords without schema qualification
                import re
                reserved_table_keywords = ['order', 'user', 'table', 'index', 'view', 'trigger', 'function', 'procedure']
                modified_query = query_text
                for keyword in reserved_table_keywords:
                    modified_query = re.sub(rf'(?<!")\b({keyword})\.', rf'"{keyword}".', modified_query, flags=re.IGNORECASE)

                # Heuristic rewrites for PostgreSQL: cast numeric comparisons and aggregates
                # Cast column side of comparisons like t.c >= 123 to handle TEXT-typed numeric columns
                def _repl_cmp(m):
                    tbl, col, op, num = m.groups()
                    return f"CAST(NULLIF({tbl}.{col}::text, '') AS DOUBLE PRECISION) {op} {num}"

                modified_query = re.sub(
                    r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\.(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=|<=|>=|<|>)\s*([0-9]+(?:\.[0-9]+)?)",
                    _repl_cmp,
                    modified_query,
                )

                # Cast aggregation arguments (AVG/SUM/MIN/MAX) similarly
                def _repl_agg(m):
                    func, tbl, col = m.groups()
                    return f"{func}(CAST(NULLIF({tbl}.{col}::text, '') AS DOUBLE PRECISION))"

                modified_query = re.sub(
                    r"\b(AVG|SUM|MIN|MAX)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\)",
                    _repl_agg,
                    modified_query,
                    flags=re.IGNORECASE,
                )
                
                # Get file sizes before execution to know where new data starts
                csv_path = str(BASE_DIR / 'data' / 'execution_data' / 'aqd_features.csv')
                plan_path = str(BASE_DIR / 'data' / 'execution_data' / 'aqd_plans.jsonl')
                
                csv_size_before = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
                plan_size_before = os.path.getsize(plan_path) if os.path.exists(plan_path) else 0
                
                # Execute query with timing
                start_time = time.perf_counter()
                cursor.execute(modified_query)
                result = cursor.fetchall()  # Fetch all results to ensure complete execution
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                
                # Get the query plan using EXPLAIN (ANALYZE FALSE to avoid re-execution)
                postgres_plan_json = None
                try:
                    cursor.execute(f"EXPLAIN (FORMAT JSON, ANALYZE FALSE, COSTS TRUE, VERBOSE FALSE) {modified_query}")
                    plan_result = cursor.fetchone()
                    if plan_result and plan_result[0]:
                        postgres_plan_json = json.dumps(plan_result[0])
                except Exception as e:
                    logger.debug(f"Failed to get EXPLAIN plan: {e}")
                
                cursor.close()
                
                # Read AQD features from CSV log: take last non-empty line and map feature_* to aqd_feature_*
                aqd_features = {}
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, 'r') as f:
                            lines = [ln.strip() for ln in f if ln.strip()]
                        if len(lines) >= 2:
                            header = lines[0].split(',')
                            last = lines[-1].split(',')
                            if len(last) == len(header):
                                row = dict(zip(header, last))
                                for k, v in row.items():
                                    if k.startswith('feature_'):
                                        try:
                                            aqd_features[f'aqd_{k}'] = float(v)
                                        except ValueError:
                                            pass
                                # Include complexity_score as a named feature
                                if 'complexity_score' in row:
                                    try:
                                        aqd_features['aqd_complexity_score'] = float(row['complexity_score'])
                                    except ValueError:
                                        pass
                    except Exception as e:
                        logger.warning(f"Failed to parse AQD CSV features: {e}")

                # Read the new plan JSONL line that was just written
                plan_json = None
                if os.path.exists(plan_path):
                    try:
                        # Only read the new content added after query execution
                        with open(plan_path, 'rb') as f:
                            f.seek(plan_size_before)
                            new_content = f.read().decode('utf-8', errors='ignore')
                        
                        # Find complete JSON lines in the new content
                        lines = [ln.strip() for ln in new_content.split('\n') if ln.strip()]
                        
                        # Take the last complete line (should be our query's plan)
                        for line in reversed(lines):
                            try:
                                # Validate it's a complete JSON
                                json.loads(line)
                                plan_json = line
                                break
                            except:
                                continue
                                
                        if not plan_json and lines:
                            # Fallback: just take the last line if we couldn't parse any
                            plan_json = lines[-1]
                            
                    except Exception as e:
                        logger.warning(f"Failed to read plan JSONL: {e}")
                
                # Use the directly obtained plan if available, otherwise fallback to JSONL
                if postgres_plan_json:
                    plan_to_return = postgres_plan_json
                else:
                    plan_to_return = plan_json

                return execution_time, None, aqd_features, plan_to_return
                
        except TimeoutError:
            return None, "Query timeout", None, None
        except psycopg2.Error as e:
            return None, str(e), None, None
        except Exception as e:
            return None, f"Unexpected error: {str(e)}", None, None
    
    def execute_duckdb_query(self, query_text, dataset_name):
        """Execute query on DuckDB"""
        if not self.duck_conn:
            return None, "No DuckDB connection"
            
        try:
            with self.timeout_context(self.timeout_seconds):
                # Modify query for schema qualification (same as PostgreSQL)
                modified_query = self.modify_query_for_schema(query_text, dataset_name)
                
                # Execute query with timing
                start_time = time.perf_counter()
                result = self.duck_conn.execute(modified_query).fetchall()
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                return execution_time, None
                
        except TimeoutError:
            return None, "Query timeout"
        except Exception as e:
            return None, str(e)
    
    def execute_dual_query(self, query_text, dataset, query_type, query_index):
        """Execute query on both engines and collect results"""
        result = ExecutionResult(query_text, dataset, query_type, query_index)
        
        # Execute on PostgreSQL
        pg_time, pg_error, aqd_features, plan_json = self.execute_postgres_query(query_text, dataset)
        result.postgres_time = pg_time
        result.postgres_error = pg_error
        result.aqd_features = aqd_features or {}
        result.postgres_plan = plan_json
        result.executed_postgres = pg_time is not None
        
        if result.executed_postgres:
            self.stats['postgres_successes'] += 1
        elif pg_error and 'timeout' in pg_error.lower():
            self.stats['postgres_timeouts'] += 1
        
        # Execute on DuckDB
        duck_time, duck_error = self.execute_duckdb_query(query_text, dataset)
        result.duckdb_time = duck_time
        result.duckdb_error = duck_error
        result.executed_duckdb = duck_time is not None
        
        if result.executed_duckdb:
            self.stats['duckdb_successes'] += 1
        elif duck_error and 'timeout' in duck_error.lower():
            self.stats['duckdb_timeouts'] += 1
        
        if result.executed_postgres and result.executed_duckdb:
            self.stats['both_successful'] += 1
        
        self.stats['queries_processed'] += 1
        
        return result
    
    def load_queries_from_file(self, filepath):
        """Load queries from SQL file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Split queries by semicolon and clean up
            queries = [q.strip() for q in content.split(';') if q.strip()]
            return queries
        except Exception as e:
            logger.error(f"Failed to load queries from {filepath}: {e}")
            return []
    
    def collect_dataset_execution_data(self, dataset_name, max_queries_per_type=1000):
        """Collect execution data for a single dataset"""
        logger.info(f"Collecting execution data for dataset: {dataset_name}")
        
        dataset_dir = os.path.join(QUERIES_BASE_DIR, dataset_name)
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return []
        
        results = []
        
        # Process AP queries
        ap_file = os.path.join(dataset_dir, 'workload_10k_ap_queries.sql')
        if os.path.exists(ap_file):
            logger.info(f"Processing AP queries from {ap_file}")
            ap_queries = self.load_queries_from_file(ap_file)
            
            # Limit number of queries to prevent excessive runtime
            ap_queries = ap_queries[:max_queries_per_type]
            
            for i, query in enumerate(tqdm(ap_queries, desc="Executing AP queries")):
                try:
                    result = self.execute_dual_query(query, dataset_name, 'AP', i)
                    results.append(result)
                    
                    # Log progress every 100 queries
                    if (i + 1) % 100 == 0:
                        success_rate = self.stats['both_successful'] / self.stats['queries_processed'] * 100
                        logger.info(f"Progress: {i+1}/{len(ap_queries)} AP queries, {success_rate:.1f}% dual success rate")
                        
                except Exception as e:
                    logger.error(f"Error processing AP query {i}: {e}")
                    traceback.print_exc()
        
        # Process TP queries
        tp_file = os.path.join(dataset_dir, 'workload_10k_tp_queries.sql')
        if os.path.exists(tp_file):
            logger.info(f"Processing TP queries from {tp_file}")
            tp_queries = self.load_queries_from_file(tp_file)
            
            # Limit number of queries
            tp_queries = tp_queries[:max_queries_per_type]
            
            for i, query in enumerate(tqdm(tp_queries, desc="Executing TP queries")):
                try:
                    result = self.execute_dual_query(query, dataset_name, 'TP', i)
                    results.append(result)
                    
                    # Log progress every 100 queries
                    if (i + 1) % 100 == 0:
                        success_rate = self.stats['both_successful'] / self.stats['queries_processed'] * 100
                        logger.info(f"Progress: {i+1}/{len(tp_queries)} TP queries, {success_rate:.1f}% dual success rate")
                        
                except Exception as e:
                    logger.error(f"Error processing TP query {i}: {e}")
                    traceback.print_exc()
        
        logger.info(f"Completed dataset {dataset_name}: {len(results)} query executions")
        return results
    
    def save_execution_data(self, results, dataset_name):
        """Save execution results to a single unified JSON file"""
        if not results:
            logger.warning(f"No results to save for {dataset_name}")
            return
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create unified training data for both LightGBM and GNN
        unified_data = []
        
        for result in results:
            if result.executed_postgres and result.executed_duckdb:
                # Create a single unified record with all information
                record = {
                    # Basic info
                    'dataset': result.dataset,
                    'query_type': result.query_type,
                    'query_index': result.query_index,
                    'query_text': result.query_text,
                    
                    # Execution times
                    'postgres_time': result.postgres_time,
                    'duckdb_time': result.duckdb_time,
                    'log_time_difference': np.log(result.postgres_time / result.duckdb_time) if result.duckdb_time > 0 else None,
                    
                    # Features for LightGBM (flattened)
                    'features': result.aqd_features,
                    
                    # Plan JSON for GNN (parse the JSONL string if available)
                    'postgres_plan_json': None,
                    
                    # Timestamp
                    'timestamp': datetime.now().isoformat()
                }
                
                # Parse the plan JSON if available
                if result.postgres_plan:
                    try:
                        # postgres_plan might be a JSON string or JSONL string
                        plan_data = json.loads(result.postgres_plan)
                        
                        # The plan_data should be a list (from EXPLAIN JSON format)
                        # or a dict with 'plan' key (from JSONL log)
                        if isinstance(plan_data, list):
                            # Direct EXPLAIN output - already in correct format
                            record['postgres_plan_json'] = plan_data
                        elif isinstance(plan_data, dict) and 'plan' in plan_data:
                            # From JSONL log with wrapper
                            record['postgres_plan_json'] = plan_data['plan']
                        else:
                            # Unknown format, store as is
                            record['postgres_plan_json'] = plan_data
                    except json.JSONDecodeError as e:
                        # If parsing fails, log warning and skip this record's plan
                        logger.debug(f"Failed to parse plan JSON: {e}")
                        record['postgres_plan_json'] = None
                
                # Only add if we have valid time difference for training
                if record['log_time_difference'] is not None:
                    unified_data.append(record)
        
        # Save unified data to a single JSON file
        unified_file = os.path.join(OUTPUT_DIR, f'{dataset_name}_unified_training_data.json')
        with open(unified_file, 'w') as f:
            json.dump(unified_data, f, indent=2)
        
        logger.info(f"Saved {len(unified_data)} training records to {unified_file}")
        
        # Also append to a global unified file for all datasets
        global_file = os.path.join(OUTPUT_DIR, 'all_datasets_unified_training_data.json')
        if os.path.exists(global_file):
            with open(global_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        existing_data.extend(unified_data)
        with open(global_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Appended to global file: {global_file} (total: {len(existing_data)} records)")
    
    def run_collection(self, datasets=None, max_queries_per_type=10000):
        """Run complete execution data collection"""
        logger.info("Starting dual execution data collection")
        
        if not self.connect_databases():
            logger.error("Failed to connect to databases")
            return False
        
        # Get list of datasets
        if datasets is None:
            datasets = []
            if os.path.exists(QUERIES_BASE_DIR):
                datasets = [d for d in os.listdir(QUERIES_BASE_DIR) 
                           if os.path.isdir(os.path.join(QUERIES_BASE_DIR, d))]
        
        if not datasets:
            logger.error("No datasets found for processing")
            return False
        
        logger.info(f"Processing {len(datasets)} datasets: {datasets}")
        
        all_results = []
        
        # Process each dataset
        for dataset in datasets:
            try:
                results = self.collect_dataset_execution_data(dataset, max_queries_per_type)
                self.save_execution_data(results, dataset)
                all_results.extend(results)
                
                # Print progress statistics
                duration = datetime.now() - self.stats['start_time']
                logger.info(f"Dataset {dataset} completed. Total stats:")
                logger.info(f"  Queries processed: {self.stats['queries_processed']}")
                logger.info(f"  PostgreSQL successes: {self.stats['postgres_successes']}")
                logger.info(f"  DuckDB successes: {self.stats['duckdb_successes']}")
                logger.info(f"  Both successful: {self.stats['both_successful']}")
                logger.info(f"  Duration: {duration}")
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {e}")
                traceback.print_exc()
        
        # Save summary statistics
        summary = {
            'datasets_processed': len(datasets),
            'total_queries': len(all_results),
            'successful_dual_executions': self.stats['both_successful'],
            'postgres_success_rate': self.stats['postgres_successes'] / self.stats['queries_processed'] if self.stats['queries_processed'] > 0 else 0,
            'duckdb_success_rate': self.stats['duckdb_successes'] / self.stats['queries_processed'] if self.stats['queries_processed'] > 0 else 0,
            'dual_success_rate': self.stats['both_successful'] / self.stats['queries_processed'] if self.stats['queries_processed'] > 0 else 0,
            'total_duration_seconds': (datetime.now() - self.stats['start_time']).total_seconds(),
            'datasets': datasets,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(OUTPUT_DIR, 'collection_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Collection complete! Summary saved to {summary_file}")
        logger.info(f"Total dual executions: {self.stats['both_successful']}")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect dual execution data for AQD training')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to process (default: all available)')
    parser.add_argument('--max_queries', type=int, default=10000,
                       help='Maximum queries per type (AP/TP) per dataset (default: 10000)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Query timeout in seconds')
    
    args = parser.parse_args()
    
    # Create collector
    collector = DualExecutionCollector(timeout_seconds=args.timeout)
    
    # Run collection
    success = collector.run_collection(datasets=args.datasets, 
                                      max_queries_per_type=args.max_queries)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
