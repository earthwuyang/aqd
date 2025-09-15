#!/usr/bin/env python3
"""
LightGBM Data Collection Script v2 - Dual-Engine Measurement
Collects features and execution times from both PostgreSQL and DuckDB engines
"""

import os
import sys
import csv
import json
import time
import psycopg2
import numpy as np
import argparse
import logging
from datetime import datetime
from contextlib import contextmanager

# Feature names - must match kernel
FEATURE_NAMES = [
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type"
]

class DualEngineCollector:
    def __init__(self, dbname="test", user="postgres", host="localhost", port=5432, 
                 output_dir="lightgbm_training_data"):
        self.dbname = dbname
        self.user = user
        self.host = host
        self.port = port
        self.output_dir = output_dir
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Persistent connections for each worker
        self.connections = []
        
    @contextmanager
    def get_connection(self, connection_id=0):
        """Get or create a persistent connection"""
        while len(self.connections) <= connection_id:
            self.connections.append(None)
        
        if self.connections[connection_id] is None:
            self.connections[connection_id] = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                host=self.host,
                port=self.port
            )
            self.connections[connection_id].autocommit = False
            self.logger.info(f"Created connection {connection_id}")
        
        yield self.connections[connection_id]
    
    def warmup_connection(self, conn):
        """Warm up a connection with dummy queries"""
        with conn.cursor() as cur:
            for _ in range(5):
                cur.execute("SELECT 1")
                cur.fetchone()
        conn.commit()
    
    def collect_features(self, conn, query):
        """Collect pre-optimization features using observability GUCs"""
        features = {}
        
        with conn.cursor() as cur:
            # Run query with feature extraction enabled but not executing
            try:
                # First, do an EXPLAIN to trigger feature extraction without execution
                cur.execute(f"EXPLAIN {query}")
                
                # Try to read features from observability GUC if available
                try:
                    cur.execute("SHOW lgbm.last_features_json")
                    features_json = cur.fetchone()[0]
                    
                    if features_json and features_json != '{}':
                        features = json.loads(features_json)
                    else:
                        # If no features available, extract manually (fallback)
                        features = self.extract_features_manually(query)
                except:
                    # GUC not available, extract manually
                    features = self.extract_features_manually(query)
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract features: {e}")
                features = self.extract_features_manually(query)
        
        conn.rollback()  # Don't commit EXPLAIN
        return features
    
    def extract_features_manually(self, query):
        """Fallback manual feature extraction from query text"""
        query_lower = query.lower()
        
        features = {
            'num_tables': query_lower.count(' from ') + query_lower.count(' join '),
            'num_joins': query_lower.count(' join '),
            'query_depth': 1,  # Simple heuristic
            'complexity_score': len(query) // 100,
            'has_aggregates': int(any(agg in query_lower for agg in ['count(', 'sum(', 'avg(', 'max(', 'min('])),
            'has_group_by': int(' group by ' in query_lower),
            'has_order_by': int(' order by ' in query_lower),
            'has_limit': int(' limit ' in query_lower),
            'has_distinct': int('distinct ' in query_lower),
            'has_window_functions': int(' over ' in query_lower),
            'has_outer_joins': int(any(join in query_lower for join in [' left join', ' right join', ' full join'])),
            'estimated_join_complexity': query_lower.count(' join ') * 2,
            'has_subqueries': int('(select ' in query_lower),
            'has_correlated_subqueries': 0,  # Hard to detect from text
            'has_large_tables': 0,  # Need statistics
            'all_tables_small': 0,  # Need statistics
            'has_complex_expressions': int(' case ' in query_lower),
            'has_user_functions': 0,  # Hard to detect
            'has_text_operations': int(any(op in query_lower for op in [' like ', ' ilike ', ' ~'])),
            'has_numeric_heavy_ops': int(any(op in query_lower for op in ['sqrt(', 'pow(', 'exp('])),
            'num_aggregate_funcs': sum(query_lower.count(f"{agg}(") for agg in ['count', 'sum', 'avg', 'max', 'min']),
            'analytical_pattern': int('group by' in query_lower or 'sum(' in query_lower),
            'transactional_pattern': int(query_lower.startswith('select') and 'where' in query_lower and 'join' not in query_lower),
            'etl_pattern': int('insert' in query_lower or 'update' in query_lower),
            'command_type': 0 if query_lower.startswith('select') else 1
        }
        
        return features
    
    def measure_execution_time(self, conn, query, engine, timeout_ms=60000):
        """Measure query execution time for specified engine"""
        start_time = None
        end_time = None
        success = False
        
        with conn.cursor() as cur:
            try:
                # Set the execution engine
                if engine == 'postgres':
                    cur.execute("SET LOCAL duckdb.force_execution = false")
                else:  # duckdb
                    cur.execute("SET LOCAL duckdb.force_execution = true")
                
                # Set statement timeout
                cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                
                # Execute and measure
                start_time = time.perf_counter()
                cur.execute(query)
                
                # Fetch all results to ensure complete execution
                results = cur.fetchall()
                end_time = time.perf_counter()
                
                success = True
                
                # Verify which engine was actually used (if GUC is available)
                try:
                    cur.execute("SHOW lgbm.last_routed_engine")
                    actual_engine = cur.fetchone()
                    if actual_engine:
                        actual_engine = actual_engine[0]
                        if actual_engine != engine and actual_engine != 'none':
                            self.logger.warning(f"Engine mismatch: requested {engine}, got {actual_engine}")
                except:
                    pass  # GUC not available, skip verification
                
            except Exception as e:
                end_time = time.perf_counter()
                self.logger.warning(f"Query failed on {engine}: {str(e)[:100]}")
        
        conn.rollback()  # Don't keep results
        
        if success and start_time and end_time:
            return (end_time - start_time) * 1000  # Convert to milliseconds
        else:
            return -1  # Failed execution
    
    def collect_query_data(self, query, query_id=None, connection_id=0):
        """Collect features and dual-engine execution times for a query"""
        with self.get_connection(connection_id) as conn:
            # Collect features
            features = self.collect_features(conn, query)
            
            # Measure PostgreSQL execution time
            pg_time = self.measure_execution_time(conn, query, 'postgres')
            
            # Small delay to clear caches
            time.sleep(0.1)
            
            # Measure DuckDB execution time
            duck_time = self.measure_execution_time(conn, query, 'duckdb')
            
            # Combine results
            result = {
                'query_id': query_id or hash(query),
                'query_length': len(query),
                **{name: features.get(name, 0) for name in FEATURE_NAMES},
                'pg_time_ms': pg_time,
                'duck_time_ms': duck_time,
                'best_engine': 'postgres' if 0 < pg_time < duck_time else 'duckdb' if 0 < duck_time < pg_time else 'unknown'
            }
            
            return result
    
    def collect_from_file(self, query_file, output_file="training_data.csv", 
                          sample_rate=1.0, interleave=True, max_queries=None):
        """Collect data from a file of queries"""
        if not os.path.exists(query_file):
            self.logger.error(f"Query file not found: {query_file}")
            return []
        
        # Read queries
        with open(query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith('--')]
        
        # Limit queries if requested
        if max_queries and len(queries) > max_queries:
            queries = queries[:max_queries]
        
        # Sample if requested
        if sample_rate < 1.0:
            n_samples = int(len(queries) * sample_rate)
            indices = np.random.choice(len(queries), n_samples, replace=False)
            queries = [queries[i] for i in indices]
        
        self.logger.info(f"Processing {len(queries)} queries from {query_file}")
        
        # Prepare output file
        output_path = os.path.join(self.output_dir, output_file)
        file_exists = os.path.exists(output_path)
        
        results = []
        with open(output_path, 'a', newline='') as csvfile:
            fieldnames = ['query_id', 'query_length'] + FEATURE_NAMES + \
                        ['pg_time_ms', 'duck_time_ms', 'best_engine']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Warm up connection
            with self.get_connection(0) as conn:
                self.warmup_connection(conn)
            
            # Process queries
            for i, query in enumerate(queries):
                if i % 10 == 0:
                    self.logger.info(f"Progress: {i}/{len(queries)}")
                
                try:
                    # Collect data with interleaving to reduce cache bias
                    if interleave and i > 0:
                        time.sleep(0.5)  # Small delay between queries
                    
                    result = self.collect_query_data(query, query_id=i)
                    
                    # Only save valid results (both engines succeeded)
                    if result['pg_time_ms'] > 0 or result['duck_time_ms'] > 0:
                        writer.writerow(result)
                        results.append(result)
                    
                    # Log interesting cases
                    if result['pg_time_ms'] > 0 and result['duck_time_ms'] > 0:
                        speedup = result['pg_time_ms'] / result['duck_time_ms']
                        if speedup > 10:
                            self.logger.info(f"Query {i}: DuckDB {speedup:.1f}x faster")
                        elif speedup < 0.1:
                            self.logger.info(f"Query {i}: PostgreSQL {1/speedup:.1f}x faster")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process query {i}: {e}")
        
        self.logger.info(f"Data collection complete. Results saved to {output_path}")
        return results
    
    def collect_from_benchmark_dir(self, benchmark_dir="benchmark_queries", database=None,
                                   workload_type="both", output_file="training_data.csv",
                                   sample_rate=1.0, interleave=True, max_queries_per_file=None):
        """Collect data from benchmark_queries directory structure"""
        if not os.path.exists(benchmark_dir):
            self.logger.error(f"Benchmark directory not found: {benchmark_dir}")
            return
        
        # Determine which databases to process
        if database:
            databases = [database]
        else:
            # Find all database directories
            databases = [d for d in os.listdir(benchmark_dir) 
                        if os.path.isdir(os.path.join(benchmark_dir, d)) 
                        and not d.startswith('.')]
            databases.sort()
        
        self.logger.info(f"Found {len(databases)} databases to process: {', '.join(databases)}")
        
        # Determine which workload files to process
        workload_files = []
        if workload_type in ['ap', 'both']:
            workload_files.append('workload_ap_queries.sql')
        if workload_type in ['tp', 'both']:
            workload_files.append('workload_tp_queries.sql')
        
        all_results = []
        total_queries = 0
        
        # Save original dbname
        original_dbname = self.dbname
        
        # Process each database
        for db_name in databases:
            db_path = os.path.join(benchmark_dir, db_name)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing database: {db_name}")
            self.logger.info(f"{'='*60}")
            
            # Update database name for this dataset
            self.dbname = db_name
            
            # Close existing connections since we're changing databases
            self.close_connections()
            self.connections = []
            
            # Process each workload file
            for workload_file in workload_files:
                query_file = os.path.join(db_path, workload_file)
                
                if not os.path.exists(query_file):
                    self.logger.warning(f"Query file not found: {query_file}")
                    continue
                
                workload_name = 'AP' if 'ap' in workload_file else 'TP'
                self.logger.info(f"\nProcessing {workload_name} workload from {db_name}")
                
                # Create a unique output file for this database and workload
                if len(databases) > 1 or len(workload_files) > 1:
                    # Add database and workload to output filename
                    base_name = output_file.replace('.csv', '')
                    specific_output = f"{base_name}_{db_name}_{workload_name.lower()}.csv"
                else:
                    specific_output = output_file
                
                # Collect data from this file
                results = self.collect_from_file(
                    query_file=query_file,
                    output_file=specific_output,
                    sample_rate=sample_rate,
                    interleave=interleave,
                    max_queries=max_queries_per_file
                )
                
                all_results.extend(results)
                total_queries += len(results)
                
                self.logger.info(f"Completed {workload_name} workload: {len(results)} queries processed")
        
        # Restore original database name
        self.dbname = original_dbname
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"COLLECTION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total databases processed: {len(databases)}")
        self.logger.info(f"Total queries collected: {total_queries}")
        
        # Analyze results
        if all_results:
            successful = sum(1 for r in all_results if r['pg_time_ms'] > 0 and r['duck_time_ms'] > 0)
            pg_wins = sum(1 for r in all_results if r['best_engine'] == 'postgres')
            duck_wins = sum(1 for r in all_results if r['best_engine'] == 'duckdb')
            
            self.logger.info(f"Successful measurements: {successful}/{total_queries}")
            self.logger.info(f"PostgreSQL faster: {pg_wins}")
            self.logger.info(f"DuckDB faster: {duck_wins}")
            
            # Save combined results if processing multiple databases
            if len(databases) > 1:
                combined_output = os.path.join(self.output_dir, output_file)
                self.logger.info(f"\nSaving combined results to: {combined_output}")
                
                with open(combined_output, 'w', newline='') as csvfile:
                    fieldnames = ['query_id', 'query_length'] + FEATURE_NAMES + \
                                ['pg_time_ms', 'duck_time_ms', 'best_engine']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in all_results:
                        writer.writerow(result)
    
    def close_connections(self):
        """Close all persistent connections"""
        for conn in self.connections:
            if conn:
                conn.close()

def main():
    parser = argparse.ArgumentParser(description='Collect LightGBM training data with dual-engine measurement')
    parser.add_argument('--query-file', help='File containing queries (optional if using --benchmark-dir)')
    parser.add_argument('--benchmark-dir', default='benchmark_queries', 
                       help='Directory containing benchmark query files (default: benchmark_queries)')
    parser.add_argument('--database', help='Specific database to process from benchmark_queries')
    parser.add_argument('--workload-type', choices=['ap', 'tp', 'both'], default='both',
                       help='Type of workload queries to collect (ap, tp, or both)')
    parser.add_argument('--output-file', default='training_data.csv', help='Output CSV file')
    parser.add_argument('--output-dir', default='lightgbm_training_data', help='Output directory')
    parser.add_argument('--dbname', default='test', help='Database name')
    parser.add_argument('--user', default='wuy', help='Database user')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Fraction of queries to sample')
    parser.add_argument('--no-interleave', action='store_true', help='Disable query interleaving')
    parser.add_argument('--max-queries-per-file', type=int, help='Maximum queries to process per file')
    
    args = parser.parse_args()
    
    # Create collector
    collector = DualEngineCollector(
        dbname=args.dbname,
        user=args.user,
        host=args.host,
        port=args.port,
        output_dir=args.output_dir
    )
    
    try:
        if args.query_file:
            # Single file mode (backward compatibility)
            collector.collect_from_file(
                query_file=args.query_file,
                output_file=args.output_file,
                sample_rate=args.sample_rate,
                interleave=not args.no_interleave
            )
        else:
            # Benchmark directory mode
            collector.collect_from_benchmark_dir(
                benchmark_dir=args.benchmark_dir,
                database=args.database,
                workload_type=args.workload_type,
                output_file=args.output_file,
                sample_rate=args.sample_rate,
                interleave=not args.no_interleave,
                max_queries_per_file=args.max_queries_per_file
            )
    finally:
        collector.close_connections()

if __name__ == "__main__":
    main()