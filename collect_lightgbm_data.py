#!/usr/bin/env python3
"""
LightGBM Data Collection Script - Advanced Benchmark Queries
Processes datasets from advanced_benchmark_queries directory
Uses expanded feature set (50 features) for improved routing accuracy
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
import glob
import random
from pathlib import Path

# Expanded feature names - must match kernel exactly (v2.0.0 schema)
FEATURE_NAMES = [
    # Original 25 features
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type",

    # Phase 1 expansion - 25 new features
    "join_type_inner", "join_type_left", "join_type_right", "join_type_full", "join_type_cross",
    "predicate_simple_eq", "predicate_range", "predicate_like", "predicate_in", "predicate_exists",
    "has_parameters", "num_cte", "max_subquery_depth", "has_recursive_cte", "has_lateral_join",
    "selectivity_high", "selectivity_medium", "selectivity_low", "cardinality_large", "cardinality_medium",
    "index_usage_likely", "partition_pruning_likely", "parallel_safe", "has_volatile_funcs", "cost_estimate_high"
]

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def extract_features_manually(query):
    """Fallback manual feature extraction from query text"""
    query_lower = query.lower()

    # Original 25 features
    features = {
        'num_tables': query_lower.count(' from ') + query_lower.count(' join '),
        'num_joins': query_lower.count(' join '),
        'query_depth': min(query_lower.count('('), 5),
        'complexity_score': len(query) // 100,
        'has_aggregates': 1 if any(agg in query_lower for agg in ['sum(', 'avg(', 'count(', 'max(', 'min(']) else 0,
        'has_group_by': 1 if ' group by ' in query_lower else 0,
        'has_order_by': 1 if ' order by ' in query_lower else 0,
        'has_limit': 1 if ' limit ' in query_lower else 0,
        'has_distinct': 1 if ' distinct ' in query_lower else 0,
        'has_window_functions': 1 if ' over(' in query_lower or ' over (' in query_lower else 0,
        'has_outer_joins': 1 if any(join in query_lower for join in [' left join', ' right join', ' full join']) else 0,
        'estimated_join_complexity': query_lower.count(' join ') * 2,
        'has_subqueries': 1 if 'select' in query_lower[10:] else 0,
        'has_correlated_subqueries': 0,
        'has_large_tables': 0,
        'all_tables_small': 0,
        'has_complex_expressions': 1 if 'case when' in query_lower else 0,
        'has_user_functions': 0,
        'has_text_operations': 1 if any(op in query_lower for op in ['like ', 'ilike ', '|| ']) else 0,
        'has_numeric_heavy_ops': 1 if any(op in query_lower for op in ['sum(', 'avg(', 'stddev(', 'variance(']) else 0,
        'num_aggregate_funcs': sum(1 for agg in ['sum(', 'avg(', 'count(', 'max(', 'min('] if agg in query_lower),
        'analytical_pattern': 1 if ('group by' in query_lower and 'sum(' in query_lower) else 0,
        'transactional_pattern': 1 if 'where' in query_lower and 'limit 1' in query_lower else 0,
        'etl_pattern': 0,
        'command_type': 0
    }

    # Phase 1 expansion features (heuristic approximations)
    # Join type analysis
    features.update({
        'join_type_inner': query_lower.count(' inner join ') + query_lower.count(' join ') - query_lower.count(' left join ') - query_lower.count(' right join ') - query_lower.count(' full join '),
        'join_type_left': query_lower.count(' left join '),
        'join_type_right': query_lower.count(' right join '),
        'join_type_full': query_lower.count(' full join '),
        'join_type_cross': query_lower.count(' cross join '),

        # Predicate type analysis
        'predicate_simple_eq': 1 if ' = ' in query_lower else 0,
        'predicate_range': 1 if any(op in query_lower for op in [' > ', ' < ', ' >= ', ' <= ', ' between ']) else 0,
        'predicate_like': 1 if any(op in query_lower for op in [' like ', ' ilike ']) else 0,
        'predicate_in': 1 if ' in (' in query_lower else 0,
        'predicate_exists': 1 if ' exists(' in query_lower else 0,

        # Parameter and CTE analysis
        'has_parameters': 1 if '$' in query else 0,
        'num_cte': query_lower.count(' with '),
        'max_subquery_depth': min(query_lower.count('select') - 1, 3) if query_lower.count('select') > 1 else 0,
        'has_recursive_cte': 1 if 'recursive' in query_lower else 0,
        'has_lateral_join': 1 if 'lateral' in query_lower else 0,

        # Selectivity and cardinality estimates (heuristic)
        'selectivity_high': 1 if (' = ' in query_lower and query_lower.count(' and ') >= 2) else 0,
        'selectivity_medium': 1 if (' = ' in query_lower or any(op in query_lower for op in [' > ', ' < '])) else 0,
        'selectivity_low': 1 if ('select *' in query_lower and ' where ' not in query_lower) else 0,
        'cardinality_large': 1 if (query_lower.count(' join ') > 2 and ' group by ' not in query_lower) else 0,
        'cardinality_medium': 1 if (query_lower.count(' join ') > 0 or query_lower.count(' from ') > 1) else 0,

        # Optimization hints (heuristic)
        'index_usage_likely': 1 if (' = ' in query_lower and ' where ' in query_lower) else 0,
        'partition_pruning_likely': 1 if any(op in query_lower for op in [' > ', ' < ', ' between ']) else 0,
        'parallel_safe': 1 if ('random()' not in query_lower and 'now()' not in query_lower) else 0,
        'has_volatile_funcs': 1 if any(func in query_lower for func in ['random()', 'now()', 'current_timestamp']) else 0,
        'cost_estimate_high': 1 if (query_lower.count(' join ') > 3 or query_lower.count(' from ') > 4) else 0
    })

    # Ensure all values are valid
    for key in features:
        if features[key] is None:
            features[key] = 0
        elif key.startswith('join_type_inner') and features[key] < 0:
            features[key] = 0

    return features

def load_advanced_benchmark_queries(benchmark_dir, dataset_name, max_ap_queries=10000, max_tp_queries=10000):
    """Load queries from advanced_benchmark_queries directory structure"""
    dataset_path = Path(benchmark_dir) / dataset_name
    ap_queries = []
    tp_queries = []

    # Load AP queries
    ap_file = dataset_path / 'advanced_ap_queries.sql'
    if ap_file.exists():
        with open(ap_file, 'r') as f:
            content = f.read()
            # Split by semicolon and newline to get complete SQL statements
            raw_queries = content.split(';\n')
            for i, query in enumerate(raw_queries):
                query = query.strip()
                if query and not query.startswith('--'):
                    ap_queries.append((f"{dataset_name}_ap_{i+1}", query, 'ap'))
                    if len(ap_queries) >= max_ap_queries:
                        break

    # Load TP queries
    tp_file = dataset_path / 'advanced_tp_queries.sql'
    if tp_file.exists():
        with open(tp_file, 'r') as f:
            content = f.read()
            # Split by semicolon and newline to get complete SQL statements
            raw_queries = content.split(';\n')
            for i, query in enumerate(raw_queries):
                query = query.strip()
                if query and not query.startswith('--'):
                    tp_queries.append((f"{dataset_name}_tp_{i+1}", query, 'tp'))
                    if len(tp_queries) >= max_tp_queries:
                        break

    # If we don't have enough unique queries, repeat them with different IDs
    while len(ap_queries) < max_ap_queries and ap_queries:
        base_queries = ap_queries[:]
        for i, (_, query, query_type) in enumerate(base_queries):
            if len(ap_queries) >= max_ap_queries:
                break
            cycle_num = len(ap_queries) // len(base_queries) + 1
            new_id = f"{dataset_name}_ap_{len(ap_queries) + 1}_cycle{cycle_num}"
            ap_queries.append((new_id, query, query_type))

    while len(tp_queries) < max_tp_queries and tp_queries:
        base_queries = tp_queries[:]
        for i, (_, query, query_type) in enumerate(base_queries):
            if len(tp_queries) >= max_tp_queries:
                break
            cycle_num = len(tp_queries) // len(base_queries) + 1
            new_id = f"{dataset_name}_tp_{len(tp_queries) + 1}_cycle{cycle_num}"
            tp_queries.append((new_id, query, query_type))

    combined_queries = ap_queries + tp_queries

    # Mix AP and TP queries so execution order alternates between workloads
    if combined_queries:
        random.shuffle(combined_queries)

    return combined_queries

def process_dataset(dataset_name, db_name, queries, user, host, port, output_dir):
    """Process all queries for a single dataset sequentially, writing results immediately"""

    logger = setup_logging()
    logger.info(f"Processing {len(queries)} queries for dataset {dataset_name}")

    # Open CSV file for writing immediately
    output_file = os.path.join(output_dir, f"training_data_{dataset_name}.csv")
    csv_file = open(output_file, 'w', newline='')
    writer = csv.writer(csv_file)

    # Write header
    header = ['query_id', 'query_type', 'query_length'] + FEATURE_NAMES + \
            ['pg_time_ms', 'duck_time_ms', 'optimal_engine']
    writer.writerow(header)
    csv_file.flush()  # Ensure header is written immediately

    results_count = 0

    # Create a single connection for this dataset
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            host=host,
            port=port
        )
        conn.autocommit = False

        # Warm up connection
        with conn.cursor() as cur:
            for _ in range(3):
                cur.execute("SELECT 1")
                cur.fetchone()
        conn.commit()

        # Process each query
        for i, (query_id, query, query_type) in enumerate(queries):
            if i % 50 == 0:
                logger.info(f"[{dataset_name}] Progress: {i}/{len(queries)}")

            try:
                # Extract features using kernel v2.0.0 features
                features = {}
                with conn.cursor() as cur:
                    try:
                        cur.execute(f"EXPLAIN {query}")
                        # Try to get features from GUC if available
                        try:
                            cur.execute("SHOW lightgbm.last_features_json")
                            features_json = cur.fetchone()[0]
                            if features_json and features_json != '{}':
                                features = json.loads(features_json)
                                # Validate we have the v2.0.0 feature set
                                if 'join_type_inner' not in features:
                                    # Fallback to manual extraction for old schema
                                    features = extract_features_manually(query)
                            else:
                                features = extract_features_manually(query)
                        except:
                            features = extract_features_manually(query)
                    except:
                        features = extract_features_manually(query)
                conn.rollback()

                # Measure PostgreSQL performance
                pg_time = -1
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL duckdb.force_execution = false")
                        try:
                            cur.execute("SET LOCAL lightgbm.enabled = false")
                        except:
                            pass
                        cur.execute("SET LOCAL statement_timeout = 60000")

                        start_time = time.perf_counter()
                        cur.execute(query)
                        results_pg = cur.fetchall()
                        end_time = time.perf_counter()
                        pg_time = (end_time - start_time) * 1000
                except:
                    pass
                conn.rollback()

                # Small delay
                time.sleep(0.05)

                # Measure DuckDB performance
                duck_time = -1
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL duckdb.force_execution = true")
                        try:
                            cur.execute("SET LOCAL lightgbm.enabled = false")
                        except:
                            pass
                        cur.execute("SET LOCAL statement_timeout = 60000")

                        start_time = time.perf_counter()
                        cur.execute(query)
                        results_duck = cur.fetchall()
                        end_time = time.perf_counter()
                        duck_time = (end_time - start_time) * 1000
                except:
                    pass
                conn.rollback()

                # Determine optimal engine
                if pg_time > 0 and duck_time > 0:
                    optimal_engine = 'duckdb' if duck_time < pg_time else 'postgres'
                else:
                    optimal_engine = 'unknown'

                # Create result row with all 50 features
                row = [query_id, query_type, len(query)]
                for feature_name in FEATURE_NAMES:
                    row.append(features.get(feature_name, 0))
                row.extend([pg_time, duck_time, optimal_engine])

                # Write row immediately to CSV
                writer.writerow(row)
                csv_file.flush()  # Ensure data is written to disk
                results_count += 1

                # Log progress every 10 queries
                if results_count % 10 == 0:
                    logger.info(f"[{dataset_name}] Saved {results_count} samples so far...")

            except Exception as e:
                logger.error(f"Error processing query {query_id}: {e}")
                continue

        logger.info(f"Completed {dataset_name}: collected {results_count} samples")

    except Exception as e:
        logger.error(f"Failed to process dataset {dataset_name}: {e}")
    finally:
        # Close CSV file
        if 'csv_file' in locals():
            csv_file.close()

        # Close database connection
        if conn:
            try:
                conn.close()
            except:
                pass

    return dataset_name, results_count

def write_results_to_file(output_file, results, mode='w'):
    """Write results to CSV file"""
    with open(output_file, mode, newline='') as f:
        writer = csv.writer(f)

        if mode == 'w':
            # Write header with all 50 features
            header = ['query_id', 'query_type', 'query_length'] + FEATURE_NAMES + \
                    ['pg_time_ms', 'duck_time_ms', 'optimal_engine']
            writer.writerow(header)

        # Write data rows
        writer.writerows(results)

def combine_csv_files(output_dir, combined_file, dataset_names):
    """Combine individual dataset CSV files into a single combined file"""
    logger = setup_logging()

    # Increase CSV field size limit for large queries
    csv.field_size_limit(sys.maxsize)

    with open(combined_file, 'w', newline='') as combined_f:
        writer = csv.writer(combined_f)

        # Write header (same as individual files)
        header = ['query_id', 'query_type', 'query_length'] + FEATURE_NAMES + \
                ['pg_time_ms', 'duck_time_ms', 'optimal_engine']
        writer.writerow(header)

        # Combine data from all individual CSV files
        total_rows = 0
        for dataset_name in dataset_names:
            individual_file = os.path.join(output_dir, f"training_data_{dataset_name}.csv")
            if os.path.exists(individual_file):
                with open(individual_file, 'r') as dataset_f:
                    reader = csv.reader(dataset_f)
                    next(reader)  # Skip header
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1

        logger.info(f"Combined {total_rows} total rows from {len(dataset_names)} datasets")

def check_existing_data(output_dir, dataset_name, target_ap=10000, target_tp=10000):
    """Check if dataset already has enough data collected"""
    output_file = os.path.join(output_dir, f"training_data_{dataset_name}.csv")
    if not os.path.exists(output_file):
        return 0, 0, False

    # Increase CSV field size limit for large queries
    csv.field_size_limit(sys.maxsize)

    ap_count = 0
    tp_count = 0

    try:
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 1:
                    query_type = row[1]  # query_type is second column
                    if query_type == 'ap':
                        ap_count += 1
                    elif query_type == 'tp':
                        tp_count += 1
    except Exception as e:
        return 0, 0, False

    # Check if we have enough data
    has_enough = ap_count >= target_ap and tp_count >= target_tp
    return ap_count, tp_count, has_enough

def collect_from_advanced_benchmark_queries(benchmark_dir="advanced_benchmark_queries",
                                           output_dir="lightgbm_training_data",
                                           max_ap_per_dataset=10000,
                                           max_tp_per_dataset=10000,
                                           user="wuy",
                                           host="localhost",
                                           port=5432):
    """Collect data from advanced_benchmark_queries directory"""

    logger = setup_logging()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting LightGBM data collection from advanced benchmark queries")
    logger.info(f"Using expanded feature set: {len(FEATURE_NAMES)} features")

    # Find all dataset directories
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        logger.error(f"Benchmark directory {benchmark_dir} does not exist")
        return

    dataset_dirs = [d.name for d in benchmark_path.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]

    logger.info(f"Found {len(dataset_dirs)} datasets to process: {', '.join(dataset_dirs)}")

    # Process datasets sequentially
    all_results = {}

    for dataset_name in dataset_dirs:
        logger.info(f"Processing dataset: {dataset_name}")

        # Check if we already have enough data for this dataset
        ap_count, tp_count, has_enough = check_existing_data(output_dir, dataset_name, max_ap_per_dataset, max_tp_per_dataset)

        if has_enough:
            logger.info(f"Dataset {dataset_name} already has sufficient data: {ap_count} AP queries, {tp_count} TP queries. Skipping.")
            all_results[dataset_name] = ap_count + tp_count
            continue

        logger.info(f"Dataset {dataset_name} needs more data: has {ap_count}/{max_ap_per_dataset} AP, {tp_count}/{max_tp_per_dataset} TP")

        # Load queries from advanced benchmark structure
        all_queries = load_advanced_benchmark_queries(benchmark_dir, dataset_name, max_ap_per_dataset, max_tp_per_dataset)

        if not all_queries:
            logger.warning(f"No queries found for dataset {dataset_name}")
            continue

        logger.info(f"Loaded {len(all_queries)} queries for {dataset_name} (target: {max_ap_per_dataset} AP + {max_tp_per_dataset} TP = {max_ap_per_dataset + max_tp_per_dataset} total)")

        # Determine database name (dataset name should match database name)
        db_name = dataset_name

        # Process dataset
        try:
            dataset_name_result, results_count = process_dataset(
                dataset_name, db_name, all_queries, user, host, port, output_dir
            )
            all_results[dataset_name] = results_count

            # Results are already written to individual CSV files
            logger.info(f"Dataset {dataset_name} completed with {results_count} samples")

        except Exception as e:
            logger.error(f"Dataset {dataset_name} failed: {e}")
            continue

    # Summary
    total_samples = sum(count for count in all_results.values())
    logger.info(f"\nCollection complete!")
    logger.info(f"Total datasets processed: {len(all_results)}")
    logger.info(f"Total samples collected: {total_samples}")
    logger.info(f"Feature vector size: {len(FEATURE_NAMES)} features")

    # Combine all individual CSV files into single file
    combined_file = os.path.join(output_dir, "training_data_combined.csv")
    combine_csv_files(output_dir, combined_file, all_results.keys())
    logger.info(f"Combined data saved to {combined_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect LightGBM training data from advanced benchmark queries')
    parser.add_argument('--benchmark-dir', default='advanced_benchmark_queries',
                       help='Directory containing advanced benchmark queries')
    parser.add_argument('--output-dir', default='lightgbm_training_data',
                       help='Output directory for training data')
    parser.add_argument('--max-ap-queries', type=int, default=10000,
                       help='Maximum AP queries per dataset')
    parser.add_argument('--max-tp-queries', type=int, default=10000,
                       help='Maximum TP queries per dataset')
    parser.add_argument('--user', default='wuy',
                       help='Database user')
    parser.add_argument('--host', default='localhost',
                       help='Database host')
    parser.add_argument('--port', type=int, default=5432,
                       help='Database port')

    args = parser.parse_args()

    collect_from_advanced_benchmark_queries(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        max_ap_per_dataset=args.max_ap_queries,
        max_tp_per_dataset=args.max_tp_queries,
        user=args.user,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()
