#!/usr/bin/env python3
"""
Comprehensive Performance Test Script for LightGBM Query Routing
Tests PostgreSQL, DuckDB, and LightGBM routing using benchmark queries
"""

import os
import sys
import time
import psycopg2
import logging
import argparse
import random
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class BenchmarkQueryLoader:
    """Loads and manages benchmark queries from various datasets"""
    
    def __init__(self, benchmark_dir='advanced_benchmark_queries'):
        self.benchmark_dir = Path(benchmark_dir)
        self.logger = setup_logging()
        
    def get_available_datasets(self):
        """Get list of available benchmark datasets"""
        if not self.benchmark_dir.exists():
            return []
        
        datasets = []
        for item in self.benchmark_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                datasets.append(item.name)
        return sorted(datasets)
    
    def load_queries_from_dataset(self, dataset_name, query_type='both', max_queries=None):
        """Load queries from a specific dataset
        
        Args:
            dataset_name: Name of the dataset (e.g., 'tpch_sf1')
            query_type: 'ap', 'tp', or 'both'
            max_queries: Maximum number of queries to load (None for all)
        """
        dataset_path = self.benchmark_dir / dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found")
        
        queries = []
        
        # Load AP queries
        if query_type in ['ap', 'both']:
            ap_file = dataset_path / 'advanced_ap_queries.sql'
            if ap_file.exists():
                with open(ap_file, 'r') as f:
                    content = f.read()
                    # Split by semicolon and newline to get complete SQL statements
                    raw_queries = content.split(';\n')
                    for i, query in enumerate(raw_queries):
                        query = query.strip()
                        if query and not query.startswith('--'):
                            queries.append({
                                'id': f"{dataset_name}_ap_{i+1}",
                                'query': query,  # Don't remove semicolon, already split by it
                                'type': 'ap',
                                'dataset': dataset_name
                            })
        
        # Load TP queries
        if query_type in ['tp', 'both']:
            tp_file = dataset_path / 'advanced_tp_queries.sql'
            if tp_file.exists():
                with open(tp_file, 'r') as f:
                    content = f.read()
                    # Split by semicolon and newline to get complete SQL statements
                    raw_queries = content.split(';\n')
                    for i, query in enumerate(raw_queries):
                        query = query.strip()
                        if query and not query.startswith('--'):
                            queries.append({
                                'id': f"{dataset_name}_tp_{i+1}",
                                'query': query,  # Don't remove semicolon, already split by it
                                'type': 'tp',
                                'dataset': dataset_name
                            })
        
        # Shuffle and limit if requested
        if max_queries and len(queries) > max_queries:
            random.shuffle(queries)
            queries = queries[:max_queries]
        
        self.logger.info(f"Loaded {len(queries)} queries from {dataset_name}")
        return queries

class PerformanceTestRunner:
    """Runs performance tests comparing PostgreSQL, DuckDB, and LightGBM routing"""

    def __init__(self, db_name='tpch_sf1', user='wuy', host='localhost', port=5432):
        self.db_name = db_name
        self.user = user
        self.host = host
        self.port = port
        self.logger = setup_logging()
        self._lock = threading.Lock()

    def _execute_single_query(self, query_data, execution_mode, query_timeout=30):
        """Execute a single query with specified mode

        Args:
            query_data: Dictionary with query information
            execution_mode: 'postgres', 'duckdb', 'lightgbm', or 'threshold_<value>'
            query_timeout: Query timeout in seconds

        Returns:
            Dictionary with execution results
        """
        query_id = query_data['id']
        query = query_data['query']
        threshold_value = None

        result = {
            'query_id': query_id,
            'query': query,
            'type': query_data['type'],
            'dataset': query_data['dataset'],
            'execution_mode': execution_mode
        }

        # Create new connection for this thread
        # Connect to the appropriate database based on the dataset name
        dataset_name = query_data['dataset']
        try:
            conn = psycopg2.connect(
                dbname=dataset_name,  # Use dataset name as database name
                user=self.user,
                host=self.host,
                port=self.port
            )
            conn.autocommit = False

            with conn.cursor() as cur:
                # Configure execution mode
                if execution_mode == 'postgres':
                    cur.execute("SET LOCAL duckdb.force_execution = false")
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                elif execution_mode == 'duckdb':
                    cur.execute("SET LOCAL duckdb.force_execution = true")
                    cur.execute("SET LOCAL lightgbm.enabled = false")
                elif execution_mode.startswith('threshold'):
                    try:
                        threshold_value = float(execution_mode.split('_', 1)[1])
                    except (IndexError, ValueError):
                        threshold_value = None
                        self.logger.warning(f"Could not parse threshold from mode {execution_mode}, using existing cost threshold")
                    cur.execute("SET lightgbm.enabled = true")
                    cur.execute("SET lightgbm.routing_strategy = 'threshold'")
                    if threshold_value is not None:
                        cur.execute(f"SET lightgbm.cost_threshold = {threshold_value}")
                    result['threshold_value'] = threshold_value
                    cur.execute("SET duckdb.force_execution = false")
                elif execution_mode == 'lightgbm':
                    cur.execute("SET lightgbm.enabled = true")  # Use SET instead of SET LOCAL
                    cur.execute("SET lightgbm.routing_strategy = 'lightgbm'")
                    cur.execute("SET duckdb.force_execution = false")  # Ensure DuckDB isn't forced
                    # Let LightGBM decide the routing

                cur.execute(f"SET LOCAL statement_timeout = {query_timeout * 1000}")

                # Execute query and measure time
                start_time = time.perf_counter()
                cur.execute(query)
                rows = cur.fetchall()
                end_time = time.perf_counter()

                execution_time = (end_time - start_time) * 1000
                result['time_ms'] = execution_time
                result['rows'] = len(rows)
                result['success'] = True

                # For adaptive routing modes, capture the routing decision
                if execution_mode == 'lightgbm' or execution_mode.startswith('threshold'):
                    try:
                        cur.execute("SHOW lightgbm.enabled")
                        enabled = cur.fetchone()

                        cur.execute("SHOW lightgbm.last_decision")
                        decision = cur.fetchone()
                        decision_value = decision[0] if decision else None

                        cur.execute("SHOW lightgbm.last_features_json")
                        features_row = cur.fetchone()
                        features_json = features_row[0] if features_row else None
                        if features_json:
                            result['routing_details'] = features_json
                            try:
                                details_obj = json.loads(features_json)
                                if isinstance(details_obj, dict):
                                    if 'plan_cost' in details_obj:
                                        result['plan_cost'] = details_obj['plan_cost']
                                    if 'threshold' in details_obj and result.get('threshold_value') is None:
                                        result['threshold_value'] = details_obj['threshold']
                            except Exception:
                                pass

                        if execution_mode.startswith('threshold') and threshold_value is not None:
                            result['threshold_value'] = threshold_value

                        with self._lock:
                            if query_id.endswith(("_1", "_2", "_tp_1", "_ap_1")):
                                debug_details = features_json if features_json else '{}'
                                print(f"DEBUG [{query_id}]: mode={execution_mode}, enabled={enabled[0] if enabled else 'N/A'}, last_decision='{decision_value}', details={debug_details}")

                        if decision_value and decision_value.lower() in ['postgres', 'postgresql']:
                            result['routed_to'] = 'postgres'
                        elif decision_value and decision_value.lower() in ['duckdb']:
                            result['routed_to'] = 'duckdb'
                        elif decision_value == 'none' or not decision_value:
                            result['routed_to'] = 'unknown'
                        else:
                            result['routed_to'] = 'unknown'
                            print(f"DEBUG: Unexpected routing decision value: '{decision_value}' for query {query_id}")
                    except Exception as e:
                        result['routed_to'] = 'unknown'
                        result['routing_details'] = f'error: {e}'
                        print(f"ERROR: Could not get routing decision for {query_id}: {e}")

        except Exception as e:
            result['time_ms'] = -1
            result['rows'] = -1
            result['success'] = False
            result['error'] = str(e)
            if execution_mode == 'lightgbm':
                result['routed_to'] = 'error'

        finally:
            try:
                conn.close()
            except:
                pass

        return result

    def run_concurrent_performance_test(self, queries, lightgbm_model_path=None, thresholds=None, query_timeout=30, max_workers=None):
        """Run concurrent performance comparison test

        Args:
            queries: List of query dictionaries
            lightgbm_model_path: Path to LightGBM model file
            thresholds: Iterable of cost thresholds for threshold-based routing
            query_timeout: Query timeout in seconds
            max_workers: Maximum number of concurrent threads (default: min(32, len(queries)))

        Returns:
            Dictionary with results for each execution mode
        """
        if max_workers is None:
            max_workers = min(32, len(queries))

        threshold_values = []
        if thresholds:
            if isinstance(thresholds, (int, float)):
                threshold_values = [float(thresholds)]
            else:
                for value in thresholds:
                    try:
                        threshold_values.append(float(value))
                    except (TypeError, ValueError):
                        self.logger.warning(f"Skipping invalid threshold value: {value}")
        threshold_values = sorted({t for t in threshold_values if t >= 0})

        # Set up LightGBM model if provided
        # We need to set up the model in all databases that will be used
        if lightgbm_model_path and os.path.exists(lightgbm_model_path):
            # Get unique dataset names from queries
            unique_datasets = list(set(query['dataset'] for query in queries))
            self.logger.info(f"Setting up LightGBM model in databases: {', '.join(unique_datasets)}")

            for dataset in unique_datasets:
                try:
                    conn = psycopg2.connect(
                        dbname=dataset,
                        user=self.user,
                        host=self.host,
                        port=self.port
                    )
                    try:
                        with conn.cursor() as cur:
                            cur.execute(f"SET lightgbm.model_path = '{lightgbm_model_path}'")
                        conn.commit()
                        self.logger.info(f"Set LightGBM model path in {dataset} database")
                    except Exception as e:
                        self.logger.warning(f"Could not set LightGBM model in {dataset}: {e}")
                    finally:
                        conn.close()
                except Exception as e:
                    self.logger.warning(f"Could not connect to database {dataset}: {e}")

        results = {}
        threshold_mode_map = {}

        execution_modes = ['postgres', 'duckdb']
        for value in threshold_values:
            if value is None:
                continue
            label = f"threshold_{int(value)}" if float(value).is_integer() else f"threshold_{value}"
            threshold_mode_map[label] = value
            execution_modes.append(label)

        if lightgbm_model_path and os.path.exists(lightgbm_model_path):
            execution_modes.append('lightgbm')

        for mode in execution_modes:
            display_name = mode.upper()
            if mode in threshold_mode_map:
                display_name = f"THRESHOLD({threshold_mode_map[mode]:g})"

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"RUNNING ALL QUERIES CONCURRENTLY WITH {display_name}")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Executing {len(queries)} queries with {max_workers} concurrent workers...")

            # Record start time for the entire batch
            batch_start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all queries for this execution mode
                future_to_query = {
                    executor.submit(self._execute_single_query, query, mode, query_timeout): query
                    for query in queries
                }

                # Collect results as they complete
                mode_results = []
                completed_count = 0
                failed_count = 0

                for future in as_completed(future_to_query):
                    result = future.result()
                    mode_results.append(result)

                    if result['success']:
                        completed_count += 1
                    else:
                        failed_count += 1
                        with self._lock:
                            error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown error'
                            self.logger.warning(f"Query {result['query_id']} failed: {error_msg}")

                    # Progress update every 10 completed queries
                    if (completed_count + failed_count) % 10 == 0:
                        with self._lock:
                            self.logger.info(f"Progress: {completed_count + failed_count}/{len(queries)} queries completed")

            batch_end_time = time.perf_counter()
            batch_total_time = (batch_end_time - batch_start_time) * 1000

            # Calculate statistics for this mode
            successful_results = [r for r in mode_results if r['success']]
            total_query_time = sum(r['time_ms'] for r in successful_results)
            avg_query_time = total_query_time / len(successful_results) if successful_results else 0
            total_rows = sum(r['rows'] for r in successful_results)

            with self._lock:
                self.logger.info(f"\n{mode.upper()} BATCH RESULTS:")
                self.logger.info(f"  Total batch wall-clock time: {batch_total_time:.2f}ms")
                self.logger.info(f"  Successful queries: {completed_count}/{len(queries)}")
                self.logger.info(f"  Failed queries: {failed_count}")
                self.logger.info(f"  Total query execution time: {total_query_time:.2f}ms")
                self.logger.info(f"  Average query time: {avg_query_time:.2f}ms")
                self.logger.info(f"  Total rows returned: {total_rows}")

                if mode == 'lightgbm' and successful_results:
                    # Analyze routing decisions
                    postgres_routes = sum(1 for r in successful_results if isinstance(r, dict) and r.get('routed_to') == 'postgres')
                    duckdb_routes = sum(1 for r in successful_results if isinstance(r, dict) and r.get('routed_to') == 'duckdb')
                    unknown_routes = len(successful_results) - postgres_routes - duckdb_routes

                    self.logger.info(f"  Routing decisions:")
                    self.logger.info(f"    -> PostgreSQL: {postgres_routes}")
                    self.logger.info(f"    -> DuckDB: {duckdb_routes}")
                    self.logger.info(f"    -> Unknown: {unknown_routes}")

            results[mode] = {
                'queries': mode_results,
                'batch_wall_time_ms': batch_total_time,
                'successful_count': completed_count,
                'failed_count': failed_count,
                'total_query_time_ms': total_query_time,
                'avg_query_time_ms': avg_query_time,
                'total_rows': total_rows
            }
            if mode in threshold_mode_map:
                results[mode]['threshold'] = threshold_mode_map[mode]

        return results

def analyze_concurrent_results(results, output_file=None):
    """Analyze and print concurrent performance test results"""
    logger = setup_logging()

    logger.info("\\n" + "="*80)
    logger.info("CONCURRENT PERFORMANCE COMPARISON RESULTS")
    logger.info("="*80)

    # Check if results is a dictionary
    if not isinstance(results, dict):
        logger.error(f"Expected results to be a dictionary, got {type(results)}")
        return

    # Extract execution modes that were actually run
    threshold_modes = sorted([mode for mode in results.keys() if isinstance(mode, str) and mode.startswith('threshold')])

    execution_modes = []
    for base in ['postgres', 'duckdb']:
        if base in results and results[base]:
            execution_modes.append(base)
    execution_modes.extend([mode for mode in threshold_modes if mode in results and results[mode]])
    if 'lightgbm' in results and results.get('lightgbm'):
        execution_modes.append('lightgbm')

    if not execution_modes:
        logger.error("No valid results to analyze!")
        return

    # Get query count from first available mode
    first_mode = execution_modes[0]
    total_queries = len(results[first_mode]['queries'])
    logger.info(f"Total queries executed: {total_queries}")
    logger.info("")

    # Compare batch performance
    logger.info("BATCH PERFORMANCE COMPARISON (Wall-Clock Time):")
    for mode in execution_modes:
        batch_time = results[mode]['batch_wall_time_ms']
        successful = results[mode]['successful_count']
        failed = results[mode]['failed_count']
        label = mode.upper()
        if mode in threshold_modes:
            threshold_value = results[mode].get('threshold')
            if threshold_value is not None:
                label = f"THRESHOLD({threshold_value:g})"
        logger.info(f"  {label:<16}: {batch_time:>8.2f}ms ({successful} success, {failed} failed)")

    logger.info("")

    # Calculate speedups if we have multiple modes
    if len(execution_modes) >= 2:
        logger.info("SPEEDUP ANALYSIS:")
        postgres_data = results.get('postgres', {})
        postgres_time = postgres_data.get('batch_wall_time_ms', 0) if isinstance(postgres_data, dict) else 0

        duckdb_data = results.get('duckdb', {})
        duckdb_time = duckdb_data.get('batch_wall_time_ms', 0) if isinstance(duckdb_data, dict) else 0

        lightgbm_data = results.get('lightgbm', {})
        lightgbm_time = lightgbm_data.get('batch_wall_time_ms', 0) if isinstance(lightgbm_data, dict) else 0

        if postgres_time > 0 and duckdb_time > 0:
            if postgres_time < duckdb_time:
                speedup = duckdb_time / postgres_time
                logger.info(f"  PostgreSQL vs DuckDB: {speedup:.2f}x faster")
            else:
                speedup = postgres_time / duckdb_time
                logger.info(f"  DuckDB vs PostgreSQL: {speedup:.2f}x faster")

        if lightgbm_time > 0:
            if postgres_time > 0:
                if lightgbm_time < postgres_time:
                    speedup = postgres_time / lightgbm_time
                    logger.info(f"  LightGBM vs PostgreSQL: {speedup:.2f}x faster")
                else:
                    speedup = lightgbm_time / postgres_time
                    logger.info(f"  PostgreSQL vs LightGBM: {speedup:.2f}x faster")

            if duckdb_time > 0:
                if lightgbm_time < duckdb_time:
                    speedup = duckdb_time / lightgbm_time
                    logger.info(f"  LightGBM vs DuckDB: {speedup:.2f}x faster")
                else:
                    speedup = lightgbm_time / duckdb_time
                    logger.info(f"  DuckDB vs LightGBM: {speedup:.2f}x faster")

        for mode in threshold_modes:
            mode_data = results.get(mode) or {}
            threshold_time = mode_data.get('batch_wall_time_ms', 0)
            if threshold_time <= 0:
                continue
            threshold_value = mode_data.get('threshold')
            label = f"Threshold({threshold_value:g})" if threshold_value is not None else mode
            if postgres_time > 0:
                logger.info(f"  {label} vs PostgreSQL: {postgres_time / threshold_time:.2f}x speedup")
            if duckdb_time > 0:
                logger.info(f"  {label} vs DuckDB: {duckdb_time / threshold_time:.2f}x speedup")
            if lightgbm_time > 0:
                logger.info(f"  {label} vs LightGBM: {lightgbm_time / threshold_time:.2f}x speedup")

        logger.info("")

    # Show detailed statistics for each mode
    logger.info("DETAILED STATISTICS:")
    for mode in execution_modes:
        mode_data = results[mode]
        label = mode.upper()
        if mode in threshold_modes:
            threshold_value = mode_data.get('threshold')
            if threshold_value is not None:
                label = f"THRESHOLD({threshold_value:g})"
        logger.info(f"  {label}:")
        logger.info(f"    Total execution time: {mode_data['total_query_time_ms']:.2f}ms")
        logger.info(f"    Average query time:   {mode_data['avg_query_time_ms']:.2f}ms")
        logger.info(f"    Total rows returned:  {mode_data['total_rows']}")
        logger.info(f"    Success rate:         {mode_data['successful_count']}/{total_queries} ({mode_data['successful_count']/total_queries*100:.1f}%)")

    # LightGBM routing analysis
    if 'lightgbm' in results and results['lightgbm']:
        lgbm_queries = results['lightgbm']['queries']
        successful_lgbm = [q for q in lgbm_queries if q['success']]

        if successful_lgbm:
            postgres_routes = sum(1 for q in successful_lgbm if isinstance(q, dict) and q.get('routed_to') == 'postgres')
            duckdb_routes = sum(1 for q in successful_lgbm if isinstance(q, dict) and q.get('routed_to') == 'duckdb')
            unknown_routes = len(successful_lgbm) - postgres_routes - duckdb_routes

            logger.info("")
            logger.info("LIGHTGBM ROUTING ANALYSIS:")
            logger.info(f"  Total successful queries: {len(successful_lgbm)}")
            logger.info(f"  Routed to PostgreSQL:    {postgres_routes} ({postgres_routes/len(successful_lgbm)*100:.1f}%)")
            logger.info(f"  Routed to DuckDB:        {duckdb_routes} ({duckdb_routes/len(successful_lgbm)*100:.1f}%)")
            logger.info(f"  Unknown routing:         {unknown_routes} ({unknown_routes/len(successful_lgbm)*100:.1f}%)")

            # Analyze routing by query type
            ap_queries = [q for q in successful_lgbm if q['type'] == 'ap']
            tp_queries = [q for q in successful_lgbm if q['type'] == 'tp']

            if ap_queries:
                ap_postgres = sum(1 for q in ap_queries if isinstance(q, dict) and q.get('routed_to') == 'postgres')
                ap_duckdb = sum(1 for q in ap_queries if isinstance(q, dict) and q.get('routed_to') == 'duckdb')
                logger.info(f"  AP queries -> PostgreSQL: {ap_postgres}/{len(ap_queries)} ({ap_postgres/len(ap_queries)*100:.1f}%)")
                logger.info(f"  AP queries -> DuckDB:     {ap_duckdb}/{len(ap_queries)} ({ap_duckdb/len(ap_queries)*100:.1f}%)")

            if tp_queries:
                tp_postgres = sum(1 for q in tp_queries if isinstance(q, dict) and q.get('routed_to') == 'postgres')
                tp_duckdb = sum(1 for q in tp_queries if isinstance(q, dict) and q.get('routed_to') == 'duckdb')
                logger.info(f"  TP queries -> PostgreSQL: {tp_postgres}/{len(tp_queries)} ({tp_postgres/len(tp_queries)*100:.1f}%)")
                logger.info(f"  TP queries -> DuckDB:     {tp_duckdb}/{len(tp_queries)} ({tp_duckdb/len(tp_queries)*100:.1f}%)")

    if threshold_modes:
        logger.info("")
        logger.info("THRESHOLD ROUTING ANALYSIS:")
        for mode in threshold_modes:
            mode_data = results.get(mode)
            if not mode_data:
                continue
            successful_threshold = [q for q in mode_data['queries'] if isinstance(q, dict) and q.get('success')]
            if not successful_threshold:
                continue
            threshold_value = mode_data.get('threshold')
            label = f"Threshold <= {threshold_value:g}" if threshold_value is not None else mode
            pg_routes = sum(1 for q in successful_threshold if q.get('routed_to') == 'postgres')
            duck_routes = sum(1 for q in successful_threshold if q.get('routed_to') == 'duckdb')
            unknown_routes = len(successful_threshold) - pg_routes - duck_routes
            logger.info(f"  {label}: {len(successful_threshold)} successful queries")
            if successful_threshold:
                logger.info(f"    -> PostgreSQL: {pg_routes} ({(pg_routes/len(successful_threshold))*100:.1f}%)")
                logger.info(f"    -> DuckDB:     {duck_routes} ({(duck_routes/len(successful_threshold))*100:.1f}%)")
                logger.info(f"    -> Unknown:    {unknown_routes} ({(unknown_routes/len(successful_threshold))*100:.1f}%)")
            plan_costs = [q.get('plan_cost') for q in successful_threshold if isinstance(q.get('plan_cost'), (int, float))]
            if plan_costs:
                logger.info(f"    Plan cost mean/median: {statistics.mean(plan_costs):.2f} / {statistics.median(plan_costs):.2f}")
    logger.info("="*80)

    # Save results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {output_file}")
    def run_performance_test(self, queries, lightgbm_model_path=None, query_timeout=30):
        """Run performance comparison test
        
        Args:
            queries: List of query dictionaries
            lightgbm_model_path: Path to LightGBM model file
            query_timeout: Query timeout in seconds
        """
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=self.db_name,
            user=self.user,
            host=self.host,
            port=self.port
        )
        conn.autocommit = False
        
        results = []
        
        try:
            # Load the LightGBM model if provided
            if lightgbm_model_path and os.path.exists(lightgbm_model_path):
                with conn.cursor() as cur:
                    self.logger.info("Loading LightGBM model into PostgreSQL...")
                    try:
                        cur.execute(f"SET lightgbm.model_path = '{lightgbm_model_path}'")
                        self.logger.info(f"Set model path to: {lightgbm_model_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not set model path: {e}")
                    
                    # Check current settings
                    try:
                        cur.execute("SHOW lightgbm.enabled")
                        enabled = cur.fetchone()[0]
                        self.logger.info(f"LightGBM enabled: {enabled}")
                        
                        cur.execute("SHOW duckdb.force_execution") 
                        force_duck = cur.fetchone()[0]
                        self.logger.info(f"DuckDB force execution: {force_duck}")
                    except Exception as e:
                        self.logger.warning(f"Could not check settings: {e}")
                
                conn.commit()
            
            self.logger.info(f"Starting performance test on {len(queries)} queries...")
            
            for i, query_data in enumerate(queries):
                query_id = query_data['id']
                query = query_data['query']
                query_type = query_data['type']
                dataset = query_data['dataset']
                
                self.logger.info(f"\\nTesting Query {i+1}/{len(queries)}: {query_id}")
                self.logger.info(f"Dataset: {dataset}, Type: {query_type}")
                
                result = {
                    'query_id': query_id,
                    'query': query,
                    'type': query_type,
                    'dataset': dataset
                }
                
                # Test 1: Always PostgreSQL
                self.logger.info("  Running with PostgreSQL...")
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL duckdb.force_execution = false")
                        cur.execute("SET LOCAL lightgbm.enabled = false")
                        cur.execute(f"SET LOCAL statement_timeout = {query_timeout * 1000}")
                        
                        start_time = time.perf_counter()
                        cur.execute(query)
                        rows = cur.fetchall()
                        end_time = time.perf_counter()
                        
                        pg_time = (end_time - start_time) * 1000
                        result['postgres_time_ms'] = pg_time
                        result['postgres_rows'] = len(rows)
                        self.logger.info(f"    PostgreSQL: {pg_time:.2f}ms, {len(rows)} rows")
                        
                except Exception as e:
                    self.logger.warning(f"    PostgreSQL failed: {e}")
                    result['postgres_time_ms'] = -1
                    result['postgres_rows'] = -1
                    result['postgres_error'] = str(e)
                
                conn.rollback()
                time.sleep(0.1)
                
                # Test 2: Always DuckDB
                self.logger.info("  Running with DuckDB...")
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL duckdb.force_execution = true")
                        cur.execute("SET LOCAL lightgbm.enabled = false")
                        cur.execute(f"SET LOCAL statement_timeout = {query_timeout * 1000}")
                        
                        start_time = time.perf_counter()
                        cur.execute(query)
                        rows = cur.fetchall()
                        end_time = time.perf_counter()
                        
                        duck_time = (end_time - start_time) * 1000
                        result['duckdb_time_ms'] = duck_time
                        result['duckdb_rows'] = len(rows)
                        self.logger.info(f"    DuckDB: {duck_time:.2f}ms, {len(rows)} rows")
                        
                except Exception as e:
                    self.logger.warning(f"    DuckDB failed: {e}")
                    result['duckdb_time_ms'] = -1
                    result['duckdb_rows'] = -1
                    result['duckdb_error'] = str(e)
                
                conn.rollback()
                time.sleep(0.1)
                
                # Test 3: LightGBM Routing (if model is available)
                if lightgbm_model_path and os.path.exists(lightgbm_model_path):
                    self.logger.info("  Running with LightGBM routing...")
                    try:
                        with conn.cursor() as cur:
                            # Reset to allow LightGBM routing
                            cur.execute("SET LOCAL lightgbm.enabled = true")
                            cur.execute(f"SET LOCAL statement_timeout = {query_timeout * 1000}")
                            
                            start_time = time.perf_counter()
                            cur.execute(query)
                            rows = cur.fetchall()
                            end_time = time.perf_counter()
                            
                            lgbm_time = (end_time - start_time) * 1000
                            result['lightgbm_time_ms'] = lgbm_time
                            result['lightgbm_rows'] = len(rows)
                            
                            # Try to get the routing decision
                            routing_decision = 'unknown'
                            try:
                                cur.execute("SHOW lightgbm.last_decision")
                                decision = cur.fetchone()
                                routing_decision = decision[0] if decision else 'unknown'
                            except:
                                pass
                            
                            result['lightgbm_decision'] = routing_decision
                            self.logger.info(f"    LightGBM: {lgbm_time:.2f}ms, {len(rows)} rows, routed to: {routing_decision}")
                            
                    except Exception as e:
                        self.logger.warning(f"    LightGBM routing failed: {e}")
                        result['lightgbm_time_ms'] = -1
                        result['lightgbm_rows'] = -1
                        result['lightgbm_decision'] = 'error'
                        result['lightgbm_error'] = str(e)
                    
                    conn.rollback()
                    time.sleep(0.1)
                else:
                    result['lightgbm_time_ms'] = -1
                    result['lightgbm_rows'] = -1
                    result['lightgbm_decision'] = 'no_model'
                
                # Determine best performance
                valid_times = []
                if result['postgres_time_ms'] > 0:
                    valid_times.append(('postgres', result['postgres_time_ms']))
                if result['duckdb_time_ms'] > 0:
                    valid_times.append(('duckdb', result['duckdb_time_ms']))
                if result['lightgbm_time_ms'] > 0:
                    valid_times.append(('lightgbm', result['lightgbm_time_ms']))
                
                if valid_times:
                    best_engine, best_time = min(valid_times, key=lambda x: x[1])
                    result['best_engine'] = best_engine
                    result['best_time_ms'] = best_time
                    self.logger.info(f"    BEST: {best_engine} ({best_time:.2f}ms)")
                else:
                    result['best_engine'] = 'none'
                    result['best_time_ms'] = -1
                
                results.append(result)
        
        finally:
            conn.close()
        
        return results

def analyze_and_print_results(results, output_file=None):
    """Analyze and print performance test results"""
    logger = setup_logging()
    
    # Separate results by dataset and type
    dataset_results = {}
    type_results = {'ap': [], 'tp': []}
    
    for r in results:
        dataset = r['dataset']
        query_type = r['type']
        
        if dataset not in dataset_results:
            dataset_results[dataset] = []
        dataset_results[dataset].append(r)
        
        if query_type in type_results:
            type_results[query_type].append(r)
    
    logger.info("\\n" + "="*80)
    logger.info("COMPREHENSIVE PERFORMANCE COMPARISON RESULTS")
    logger.info("="*80)
    
    # Overall statistics
    total_queries = len(results)
    valid_results = [r for r in results if r['postgres_time_ms'] > 0 and r['duckdb_time_ms'] > 0]
    valid_with_lgbm = [r for r in valid_results if r['lightgbm_time_ms'] > 0]
    
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"Valid queries (both PG & Duck): {len(valid_results)}")
    logger.info(f"Valid queries (all three): {len(valid_with_lgbm)}")
    logger.info("")
    
    if valid_results:
        # Calculate totals
        total_pg_time = sum(r['postgres_time_ms'] for r in valid_results)
        total_duck_time = sum(r['duckdb_time_ms'] for r in valid_results)
        
        logger.info("OVERALL PERFORMANCE:")
        logger.info(f"  Always PostgreSQL: {total_pg_time:.2f}ms")
        logger.info(f"  Always DuckDB:     {total_duck_time:.2f}ms")
        
        if valid_with_lgbm:
            total_lgbm_time = sum(r['lightgbm_time_ms'] for r in valid_with_lgbm)
            logger.info(f"  LightGBM Routing:  {total_lgbm_time:.2f}ms")
            
            pg_vs_lgbm_speedup = total_pg_time / total_lgbm_time if total_lgbm_time > 0 else 1
            duck_vs_lgbm_speedup = total_duck_time / total_lgbm_time if total_lgbm_time > 0 else 1
            
            logger.info(f"  LightGBM vs PostgreSQL: {pg_vs_lgbm_speedup:.2f}x speedup")
            logger.info(f"  LightGBM vs DuckDB:     {duck_vs_lgbm_speedup:.2f}x speedup")
            
            # Routing decision analysis
            postgres_routes = sum(1 for r in valid_with_lgbm if 'postgres' in str(r.get('lightgbm_decision', '')).lower())
            duckdb_routes = sum(1 for r in valid_with_lgbm if 'duckdb' in str(r.get('lightgbm_decision', '')).lower())
            unknown_routes = len(valid_with_lgbm) - postgres_routes - duckdb_routes
            
            logger.info("")
            logger.info("ROUTING DECISIONS:")
            logger.info(f"  Routed to PostgreSQL: {postgres_routes}")
            logger.info(f"  Routed to DuckDB:     {duckdb_routes}")
            logger.info(f"  Unknown routing:      {unknown_routes}")
        
        logger.info("")
        
    # Per-dataset analysis
    logger.info("PER-DATASET BREAKDOWN:")
    for dataset_name, dataset_queries in dataset_results.items():
        valid_dataset = [r for r in dataset_queries if r['postgres_time_ms'] > 0 and r['duckdb_time_ms'] > 0]
        if valid_dataset:
            dataset_pg_time = sum(r['postgres_time_ms'] for r in valid_dataset)
            dataset_duck_time = sum(r['duckdb_time_ms'] for r in valid_dataset)
            logger.info(f"  {dataset_name}: {len(valid_dataset)} queries, PG={dataset_pg_time:.1f}ms, Duck={dataset_duck_time:.1f}ms")
    
    # Per-type analysis
    logger.info("")
    logger.info("PER-TYPE BREAKDOWN:")
    for query_type, type_queries in type_results.items():
        valid_type = [r for r in type_queries if r['postgres_time_ms'] > 0 and r['duckdb_time_ms'] > 0]
        if valid_type:
            type_pg_time = sum(r['postgres_time_ms'] for r in valid_type)
            type_duck_time = sum(r['duckdb_time_ms'] for r in valid_type)
            logger.info(f"  {query_type.upper()}: {len(valid_type)} queries, PG={type_pg_time:.1f}ms, Duck={type_duck_time:.1f}ms")
    
    # Error analysis
    pg_errors = sum(1 for r in results if r['postgres_time_ms'] == -1)
    duck_errors = sum(1 for r in results if r['duckdb_time_ms'] == -1)
    lgbm_errors = sum(1 for r in results if r['lightgbm_time_ms'] == -1 and 'no_model' not in str(r.get('lightgbm_decision', '')))
    
    if pg_errors > 0 or duck_errors > 0 or lgbm_errors > 0:
        logger.info("")
        logger.info("ERROR SUMMARY:")
        if pg_errors > 0:
            logger.info(f"  PostgreSQL errors: {pg_errors}")
        if duck_errors > 0:
            logger.info(f"  DuckDB errors: {duck_errors}")
        if lgbm_errors > 0:
            logger.info(f"  LightGBM errors: {lgbm_errors}")
    
    logger.info("="*80)
    
    # Save results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Performance Test for LightGBM Query Routing')
    parser.add_argument('--benchmark-dir', default='advanced_benchmark_queries', 
                       help='Directory containing benchmark queries')
    parser.add_argument('--datasets', nargs='+', 
                       help='Datasets to test (default: all available)')
    parser.add_argument('--query-type', choices=['ap', 'tp', 'both'], default='both',
                       help='Type of queries to test')
    parser.add_argument('--max-queries', type=int, default=5,
                       help='Maximum number of queries per dataset (default: 50)')
    parser.add_argument('--query-timeout', type=int, default=60,
                       help='Query timeout in seconds (default: 30)')
    parser.add_argument('--lightgbm-model',
                       default='lightgbm_models/lightgbm_model.txt',
                       help='Path to LightGBM model file')
    parser.add_argument('--thresholds', default='10000,50000',
                       help='Comma separated list of cost thresholds for threshold-based routing (set to none to skip)')
    parser.add_argument('--db-name', default='tpch_sf1',
                       help='Database name to connect to')
    parser.add_argument('--user', default='wuy',
                       help='Database user')
    parser.add_argument('--host', default='localhost',
                       help='Database host')
    parser.add_argument('--port', type=int, default=5432,
                       help='Database port')
    parser.add_argument('--output', 
                       help='Output file for detailed results (JSON format)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for query selection')
    parser.add_argument('--concurrent', action='store_true', default=True,
                       help='Run queries concurrently in batches (default: sequential)')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum concurrent workers (default: min(32, num_queries))')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    logger = setup_logging()
    logger.info("Starting Comprehensive Performance Test")
    logger.info(f"Benchmark directory: {args.benchmark_dir}")
    logger.info(f"Max queries per dataset: {args.max_queries}")
    logger.info(f"Query timeout: {args.query_timeout}s")
    
    # Load benchmark queries
    loader = BenchmarkQueryLoader(args.benchmark_dir)
    available_datasets = loader.get_available_datasets()
    
    if not available_datasets:
        logger.error(f"No benchmark datasets found in {args.benchmark_dir}")
        return 1
    
    datasets_to_test = args.datasets if args.datasets else available_datasets
    logger.info(f"Available datasets: {', '.join(available_datasets)}")
    logger.info(f"Testing datasets: {', '.join(datasets_to_test)}")
    
    all_queries = []
    for dataset in datasets_to_test:
        if dataset not in available_datasets:
            logger.warning(f"Dataset {dataset} not found, skipping...")
            continue
        
        try:
            queries = loader.load_queries_from_dataset(
                dataset, args.query_type, args.max_queries
            )
            all_queries.extend(queries)
        except Exception as e:
            logger.error(f"Failed to load queries from {dataset}: {e}")
    
    if not all_queries:
        logger.error("No queries loaded!")
        return 1
    
    thresholds = []
    if getattr(args, 'thresholds', None):
        if isinstance(args.thresholds, str) and args.thresholds.lower() not in ('', 'none'):
            thresholds = [item.strip() for item in args.thresholds.split(',') if item.strip()]
        elif isinstance(args.thresholds, (list, tuple)):
            thresholds = list(args.thresholds)

    if thresholds:
        logger.info(f"Threshold routing candidates: {thresholds}")
    else:
        logger.info("Threshold routing disabled or not requested")

    # Run performance test
    runner = PerformanceTestRunner(
        db_name=args.db_name,
        user=args.user, 
        host=args.host,
        port=args.port
    )
    
    try:
        if args.concurrent:
            logger.info("\\nRunning CONCURRENT performance test...")
            results = runner.run_concurrent_performance_test(
                all_queries,
                lightgbm_model_path=args.lightgbm_model,
                thresholds=thresholds,
                query_timeout=args.query_timeout,
                max_workers=args.max_workers
            )

            # Analyze and print concurrent results
            analyze_concurrent_results(results, args.output)
        else:
            logger.info("\\nRunning SEQUENTIAL performance test...")
            results = runner.run_performance_test(
                all_queries,
                args.lightgbm_model,
                args.query_timeout
            )

            # Analyze and print results
            analyze_and_print_results(results, args.output)

        logger.info("\\nPerformance test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())