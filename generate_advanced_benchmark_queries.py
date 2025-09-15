#!/usr/bin/env python3
"""
Advanced Benchmark Query Generator - Simplified Version
Generates complex AP (Analytical) and TP (Transactional) queries without external dependencies

Features:
- Complex predicates with AND/OR combinations
- Multiple joins (up to 10 tables)
- GROUP BY with multiple columns and HAVING clauses
- Aggregations with COUNT, SUM, AVG, MIN, MAX
- ORDER BY and LIMIT clauses
- Data-driven query generation using actual database schema
"""

import os
import sys
import time
import json
import random
import argparse
import psycopg2
import numpy as np
from pathlib import Path
from datetime import datetime
from enum import Enum
from collections import defaultdict

# Database configuration
POSTGRESQL_CONFIG = {
    'host': '/tmp',  # Unix socket
    'port': 5432,
    'user': os.environ.get('USER', 'postgres')
}

# Base directory
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'advanced_benchmark_queries'

# Available datasets from our PostgreSQL instance
CTU_DATASETS = [
    'Airline', 'Credit', 'Carcinogenesis', 'employee', 
    'financial', 'geneea', 'Hepatitis_std'
]

ALL_DATASETS = CTU_DATASETS + ['tpch_sf1', 'tpcds_sf1']

class Operator(Enum):
    EQ = '='
    NEQ = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NULL = 'IS NULL'
    IS_NOT_NULL = 'IS NOT NULL'
    IN = 'IN'
    BETWEEN = 'BETWEEN'

class AggregateFunction(Enum):
    COUNT = 'COUNT'
    SUM = 'SUM'
    AVG = 'AVG'
    MIN = 'MIN'
    MAX = 'MAX'

def get_database_connection(database_name):
    """Get connection to specified database"""
    conn = psycopg2.connect(
        host=POSTGRESQL_CONFIG['host'],
        port=POSTGRESQL_CONFIG['port'],
        user=POSTGRESQL_CONFIG['user'],
        database=database_name
    )
    return conn

def compute_column_statistics(conn, database_name):
    """Compute comprehensive column statistics for all tables"""
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    column_stats = {}
    
    for table in tables:
        print(f"  Computing statistics for table: {table}")
        
        # Get column info
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        
        columns = cursor.fetchall()
        table_stats = {}
        
        for col_name, data_type, is_nullable in columns:
            try:
                # Get row count for the table
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                row_count = cursor.fetchone()[0]
                
                if row_count == 0:
                    continue
                
                # Basic statistics
                cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}" WHERE "{col_name}" IS NOT NULL')
                num_unique = cursor.fetchone()[0]
                
                cursor.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL')
                null_count = cursor.fetchone()[0]
                
                # Determine datatype category
                if data_type in ['integer', 'bigint', 'smallint', 'serial', 'bigserial']:
                    datatype = 'int'
                elif data_type in ['real', 'double precision', 'numeric', 'decimal']:
                    datatype = 'float'
                elif data_type in ['character varying', 'text', 'character', 'varchar', 'char']:
                    if num_unique <= 100:  # Categorical threshold
                        datatype = 'categorical'
                    else:
                        datatype = 'string'
                else:
                    datatype = 'misc'
                
                # Compute percentiles for numeric columns
                percentiles = []
                if datatype in ['int', 'float'] and num_unique > 1:
                    try:
                        cursor.execute(f"""
                            SELECT 
                                MIN("{col_name}") as min_val,
                                MAX("{col_name}") as max_val,
                                percentile_cont(0.25) WITHIN GROUP (ORDER BY "{col_name}") as p25,
                                percentile_cont(0.5) WITHIN GROUP (ORDER BY "{col_name}") as p50,
                                percentile_cont(0.75) WITHIN GROUP (ORDER BY "{col_name}") as p75
                            FROM "{table}" 
                            WHERE "{col_name}" IS NOT NULL
                        """)
                        min_val, max_val, p25, p50, p75 = cursor.fetchone()
                        percentiles = [float(min_val), float(p25), float(p50), float(p75), float(max_val)]
                    except:
                        percentiles = []
                
                # Get unique values for categorical columns
                unique_vals = []
                if datatype == 'categorical' and num_unique <= 100:
                    try:
                        cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 100')
                        unique_vals = [str(row[0]) for row in cursor.fetchall()]
                    except:
                        unique_vals = []
                
                # Sample some values for string columns
                sample_values = []
                if datatype in ['string', 'categorical']:
                    try:
                        cursor.execute(f'SELECT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL ORDER BY RANDOM() LIMIT 50')
                        sample_values = [str(row[0]) for row in cursor.fetchall()]
                    except:
                        sample_values = []
                
                table_stats[col_name] = {
                    'datatype': datatype,
                    'num_unique': num_unique,
                    'null_count': null_count,
                    'row_count': row_count,
                    'null_frac': null_count / row_count if row_count > 0 else 0,
                    'percentiles': percentiles,
                    'unique_vals': unique_vals,
                    'sample_values': sample_values,
                    'is_nullable': is_nullable == 'YES',
                    'postgres_type': data_type
                }
                
            except Exception as e:
                print(f"    Warning: Failed to compute stats for {table}.{col_name}: {e}")
                continue
        
        column_stats[table] = table_stats
    
    cursor.close()
    return column_stats

def compute_string_statistics(conn, column_stats):
    """Compute string-specific statistics"""
    cursor = conn.cursor()
    string_stats = {}
    
    for table, table_stats in column_stats.items():
        table_string_stats = {}
        
        for col_name, col_stats in table_stats.items():
            if col_stats['datatype'] in ['string', 'categorical']:
                try:
                    # Get string length statistics
                    cursor.execute(f"""
                        SELECT 
                            MIN(LENGTH("{col_name}")) as min_len,
                            MAX(LENGTH("{col_name}")) as max_len,
                            AVG(LENGTH("{col_name}")) as avg_len
                        FROM "{table}" 
                        WHERE "{col_name}" IS NOT NULL
                    """)
                    min_len, max_len, avg_len = cursor.fetchone()
                    
                    # Get character patterns
                    cursor.execute(f"""
                        SELECT "{col_name}", COUNT(*) 
                        FROM "{table}" 
                        WHERE "{col_name}" IS NOT NULL 
                        GROUP BY "{col_name}" 
                        ORDER BY COUNT(*) DESC 
                        LIMIT 20
                    """)
                    top_values = [(str(val), count) for val, count in cursor.fetchall()]
                    
                    table_string_stats[col_name] = {
                        'min_length': int(min_len) if min_len else 0,
                        'max_length': int(max_len) if max_len else 0,
                        'avg_length': float(avg_len) if avg_len else 0,
                        'top_values': top_values,
                        'total_unique': col_stats['num_unique']
                    }
                    
                except Exception as e:
                    print(f"    Warning: Failed to compute string stats for {table}.{col_name}: {e}")
                    continue
        
        if table_string_stats:
            string_stats[table] = table_string_stats
    
    cursor.close()
    return string_stats

def get_table_info(conn, database_name):
    """Get table and column information from database"""
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    # Get column info for each table
    table_columns = {}
    table_relationships = []
    
    for table in tables:
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        
        columns = []
        for col_name, data_type, is_nullable in cursor.fetchall():
            columns.append({
                'name': col_name,
                'type': data_type,
                'nullable': is_nullable == 'YES'
            })
        table_columns[table] = columns
        
        # Get foreign key relationships
        cursor.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
            AND tc.table_name = %s
        """, (table,))
        
        for col, foreign_table, foreign_col in cursor.fetchall():
            table_relationships.append((table, col, foreign_table, foreign_col))
    
    cursor.close()
    return {
        'tables': tables,
        'columns': table_columns,
        'relationships': table_relationships
    }

def sample_values_from_table(conn, table, column, limit=100):
    """Sample actual values from a table column"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            SELECT DISTINCT "{column}" 
            FROM "{table}" 
            WHERE "{column}" IS NOT NULL 
            ORDER BY RANDOM() 
            LIMIT %s
        """, (limit,))
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []
    finally:
        cursor.close()

def generate_literal_value_from_stats(column_stats, string_stats=None):
    """Generate a realistic literal value using computed statistics"""
    datatype = column_stats['datatype']
    
    # For categorical columns, use actual unique values
    if datatype == 'categorical' and column_stats['unique_vals']:
        value = random.choice(column_stats['unique_vals'])
        return f"'{value}'" if isinstance(value, str) else str(value)
    
    # For numeric columns, use percentile-based sampling
    if datatype in ['int', 'float'] and column_stats['percentiles']:
        percentiles = column_stats['percentiles']
        if len(percentiles) >= 2:
            # Sample between random percentiles to get realistic distribution
            idx = random.randint(0, len(percentiles) - 2)
            low, high = percentiles[idx], percentiles[idx + 1]
            
            # Avoid "low >= high" error
            if low >= high:
                # Use a single percentile value or generate around it
                value = low
                if datatype == 'int':
                    value = int(value) + random.randint(-10, 10)
                else:
                    value = float(value) * (1 + random.uniform(-0.1, 0.1))
            else:
                if datatype == 'int':
                    value = random.randint(int(low), int(high))
                else:
                    value = random.uniform(float(low), float(high))
            
            return str(value)
    
    # For string columns, use sample values or generate based on patterns
    if datatype == 'string':
        if column_stats['sample_values']:
            value = random.choice(column_stats['sample_values'])
            return f"'{value}'"
        elif string_stats and 'top_values' in string_stats:
            if string_stats['top_values']:
                value, _ = random.choice(string_stats['top_values'])
                return f"'{value}'"
        
        # Generate based on average length
        avg_len = 8
        if string_stats and 'avg_length' in string_stats:
            avg_len = max(1, int(string_stats['avg_length']))
        
        # Generate a random string of appropriate length
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        value = ''.join(random.choice(chars) for _ in range(random.randint(1, avg_len)))
        return f"'{value}'"
    
    # Fallback based on PostgreSQL data type
    postgres_type = column_stats.get('postgres_type', 'text')
    if postgres_type in ['integer', 'bigint', 'smallint']:
        return str(random.randint(1, 10000))
    elif postgres_type in ['real', 'double precision', 'numeric']:
        return str(round(random.uniform(0.1, 1000.0), 2))
    elif postgres_type in ['date']:
        return f"'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'"
    elif postgres_type in ['timestamp', 'timestamp without time zone']:
        return f"'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00'"
    else:
        return "'default'"

def generate_predicate_from_stats(table, column_stats, string_stats, available_tables):
    """Generate a single predicate using computed statistics"""
    if table not in available_tables or table not in column_stats:
        return None
    
    table_stats = column_stats[table]
    if not table_stats:
        return None
    
    col_name = random.choice(list(table_stats.keys()))
    col_stats = table_stats[col_name]
    
    # Choose operator based on data type
    datatype = col_stats['datatype']
    if datatype in ['int', 'float']:
        operators = [Operator.EQ, Operator.NEQ, Operator.LT, Operator.LE, Operator.GT, Operator.GE]
        if col_stats['is_nullable'] and col_stats['null_frac'] > 0.1:
            operators.extend([Operator.IS_NULL, Operator.IS_NOT_NULL])
    else:
        operators = [Operator.EQ, Operator.NEQ]
        if datatype == 'string':
            operators.append(Operator.LIKE)
        if col_stats['is_nullable'] and col_stats['null_frac'] > 0.1:
            operators.extend([Operator.IS_NULL, Operator.IS_NOT_NULL])
    
    operator = random.choice(operators)
    
    if operator in [Operator.IS_NULL, Operator.IS_NOT_NULL]:
        return f'"{table}"."{col_name}" {operator.value}'
    else:
        # Get string stats for this column if available
        col_string_stats = None
        if string_stats and table in string_stats and col_name in string_stats[table]:
            col_string_stats = string_stats[table][col_name]
            
        literal = generate_literal_value_from_stats(col_stats, col_string_stats)
        
        if operator == Operator.LIKE:
            # Remove quotes for LIKE pattern
            if literal.startswith("'") and literal.endswith("'"):
                literal_val = literal[1:-1]
                return f'"{table}"."{col_name}" {operator.value} \'%{literal_val}%\''
            else:
                return f'"{table}"."{col_name}" {operator.value} \'%{literal}%\''
        else:
            return f'"{table}"."{col_name}" {operator.value} {literal}'

def generate_join_clause(table_info, start_table, max_joins):
    """Generate JOIN clauses"""
    joins = []
    joined_tables = {start_table}
    available_rels = table_info['relationships']
    
    for _ in range(max_joins):
        # Find possible joins from current tables
        possible_joins = []
        for table1, col1, table2, col2 in available_rels:
            if table1 in joined_tables and table2 not in joined_tables:
                possible_joins.append((table1, col1, table2, col2))
            elif table2 in joined_tables and table1 not in joined_tables:
                possible_joins.append((table2, col2, table1, col1))
        
        if not possible_joins:
            break
            
        table1, col1, table2, col2 = random.choice(possible_joins)
        join_type = random.choice(['JOIN', 'LEFT JOIN']) 
        joins.append(f'{join_type} "{table2}" ON "{table1}"."{col1}" = "{table2}"."{col2}"')
        joined_tables.add(table2)
    
    return joins, list(joined_tables)

def generate_select_clause(table_info, available_tables, include_aggregates=False):
    """Generate SELECT clause"""
    select_items = []
    
    # Add regular columns
    for _ in range(random.randint(2, 5)):
        table = random.choice(available_tables)
        columns = table_info['columns'][table]
        if columns:
            column = random.choice(columns)
            select_items.append(f'"{table}"."{column["name"]}"')
    
    # Add aggregates for AP queries
    if include_aggregates:
        for _ in range(random.randint(1, 3)):
            table = random.choice(available_tables)
            columns = [c for c in table_info['columns'][table] 
                      if c['type'] in ['integer', 'bigint', 'smallint', 'real', 'double precision', 'numeric']]
            if columns:
                column = random.choice(columns)
                agg_func = random.choice(list(AggregateFunction))
                select_items.append(f'{agg_func.value}("{table}"."{column["name"]}")')
    
    return select_items

def generate_ap_query_from_stats(table_info, column_stats, string_stats, database_name):
    """Generate Analytical Processing query using computed statistics"""
    tables = table_info['tables']
    if not tables:
        return None
        
    # Start with random table
    start_table = random.choice(tables)
    
    # Generate joins
    max_joins = random.randint(1, min(3, len(tables)-1))
    joins, available_tables = generate_join_clause(table_info, start_table, max_joins)
    
    # Generate SELECT with aggregates
    select_items = generate_select_clause(table_info, available_tables, include_aggregates=True)
    
    # Generate WHERE clause using statistics
    predicates = []
    for _ in range(random.randint(1, 5)):
        table = random.choice(available_tables)
        pred = generate_predicate_from_stats(table, column_stats, string_stats, available_tables)
        if pred:
            predicates.append(pred)
    
    where_clause = ""
    if predicates:
        # Mix AND/OR operators
        if len(predicates) > 1 and random.random() < 0.3:
            logical_op = random.choice([' AND ', ' OR '])
            where_clause = f" WHERE {logical_op.join(predicates)}"
        else:
            where_clause = f" WHERE {' AND '.join(predicates)}"
    
    # Generate GROUP BY for aggregates
    group_by_clause = ""
    having_clause = ""
    regular_columns = [item for item in select_items if not any(agg.value in item for agg in AggregateFunction)]
    
    if regular_columns and any(agg.value in item for item in select_items for agg in AggregateFunction):
        group_by_clause = f" GROUP BY {', '.join(regular_columns)}"
        
        # Add HAVING clause sometimes
        if random.random() < 0.3:
            agg_columns = [item for item in select_items if any(agg.value in item for agg in AggregateFunction)]
            if agg_columns:
                having_pred = f"{random.choice(agg_columns)} > {random.randint(1, 100)}"
                having_clause = f" HAVING {having_pred}"
    
    # Generate ORDER BY
    order_by_clause = ""
    if random.random() < 0.7:
        order_cols = random.sample(select_items, min(2, len(select_items)))
        order_items = [f"{col} {random.choice(['ASC', 'DESC'])}" for col in order_cols]
        order_by_clause = f" ORDER BY {', '.join(order_items)}"
    
    # Generate LIMIT
    limit_clause = ""
    if random.random() < 0.5:
        limit_clause = f" LIMIT {random.randint(10, 1000)}"
    
    # Build query
    join_clause = f' {" ".join(joins)}' if joins else ""
    
    query = f"""SELECT {', '.join(select_items)}
FROM "{start_table}"{join_clause}{where_clause}{group_by_clause}{having_clause}{order_by_clause}{limit_clause};"""
    
    return query

def generate_tp_query_from_stats(table_info, column_stats, string_stats, database_name):
    """Generate Transactional Processing query using computed statistics"""
    tables = table_info['tables']
    if not tables:
        return None
        
    # Start with random table
    start_table = random.choice(tables)
    
    # Generate fewer joins for TP
    max_joins = random.randint(0, min(2, len(tables)-1))
    joins, available_tables = generate_join_clause(table_info, start_table, max_joins)
    
    # Generate SELECT without aggregates (point queries)
    select_items = generate_select_clause(table_info, available_tables, include_aggregates=False)
    
    # Generate focused WHERE clause (equality predicates)
    predicates = []
    for _ in range(random.randint(1, 3)):
        table = random.choice(available_tables)
        pred = generate_predicate_from_stats(table, column_stats, string_stats, available_tables)
        if pred and ('=' in pred or 'IS NOT NULL' in pred):  # Focus on equality/existence
            predicates.append(pred)
    
    where_clause = ""
    if predicates:
        where_clause = f" WHERE {' AND '.join(predicates)}"
    
    # Generate LIMIT for point queries
    limit_clause = f" LIMIT {random.randint(1, 100)}"
    
    # Build query
    join_clause = f' {" ".join(joins)}' if joins else ""
    
    query = f"""SELECT {', '.join(select_items)}
FROM "{start_table}"{join_clause}{where_clause}{limit_clause};"""
    
    return query

def generate_queries_for_dataset(database_name, output_dir, num_ap=10000, num_tp=10000):
    """Generate queries for a specific dataset"""
    print(f"\n{'='*80}")
    print(f"Generating Advanced queries for: {database_name}")
    print(f"{'='*80}")
    
    # Create output directory
    dataset_output_dir = output_dir / database_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Connect to database
        conn = get_database_connection(database_name)
        
        # Get table information
        print("Analyzing database schema...")
        table_info = get_table_info(conn, database_name)
        print(f"  Found {len(table_info['tables'])} tables")
        print(f"  Found {len(table_info['relationships'])} relationships")
        
        # Compute column statistics
        print("Computing column statistics...")
        column_stats = compute_column_statistics(conn, database_name)
        print(f"  Computed statistics for {len(column_stats)} tables")
        
        # Compute string statistics
        print("Computing string statistics...")
        string_stats = compute_string_statistics(conn, column_stats)
        print(f"  Computed string statistics for {len(string_stats)} tables")
        
        # Save statistics for debugging/reference
        stats_file = dataset_output_dir / 'column_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(column_stats, f, indent=2)
        
        string_stats_file = dataset_output_dir / 'string_statistics.json'
        with open(string_stats_file, 'w') as f:
            json.dump(string_stats, f, indent=2)
        
        # Generate AP queries
        print(f"\nGenerating {num_ap:,} AP (Analytical) queries...")
        ap_start_time = time.time()
        
        ap_file = dataset_output_dir / 'advanced_ap_queries.sql'
        with open(ap_file, 'w') as f:
            successful_queries = 0
            for i in range(num_ap):
                try:
                    query = generate_ap_query_from_stats(table_info, column_stats, string_stats, database_name)
                    if query:
                        f.write(query + '\n\n')
                        successful_queries += 1
                except Exception as e:
                    print(f"  Warning: Failed to generate AP query {i+1}: {e}")
                
                if (i + 1) % 1000 == 0:
                    print(f"  Progress: {i+1:,}/{num_ap:,} AP queries")
        
        ap_time = time.time() - ap_start_time
        print(f"  ✓ Generated {successful_queries:,} AP queries in {ap_time:.1f}s")
        
        # Generate TP queries
        print(f"\nGenerating {num_tp:,} TP (Transactional) queries...")
        tp_start_time = time.time()
        
        tp_file = dataset_output_dir / 'advanced_tp_queries.sql'
        with open(tp_file, 'w') as f:
            successful_queries = 0
            for i in range(num_tp):
                try:
                    query = generate_tp_query_from_stats(table_info, column_stats, string_stats, database_name)
                    if query:
                        f.write(query + '\n\n')
                        successful_queries += 1
                except Exception as e:
                    print(f"  Warning: Failed to generate TP query {i+1}: {e}")
                
                if (i + 1) % 1000 == 0:
                    print(f"  Progress: {i+1:,}/{num_tp:,} TP queries")
        
        tp_time = time.time() - tp_start_time
        print(f"  ✓ Generated {successful_queries:,} TP queries in {tp_time:.1f}s")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to generate queries for {database_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate advanced benchmark queries')
    parser.add_argument('--databases', nargs='+', default=ALL_DATASETS,
                       help='Databases to process')
    parser.add_argument('--num-ap', type=int, default=10000,
                       help='Number of AP queries per database')
    parser.add_argument('--num-tp', type=int, default=10000,
                       help='Number of TP queries per database')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Advanced Benchmark Query Generator")
    print("Generates complex AP and TP queries with sophisticated patterns")
    print("="*80)
    
    print(f"\nDatabases to process: {args.databases}")
    print(f"Queries per database: {args.num_ap:,} AP + {args.num_tp:,} TP = {args.num_ap + args.num_tp:,} total")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    successful_databases = 0
    total_queries = 0
    
    for database_name in args.databases:
        if generate_queries_for_dataset(database_name, args.output_dir, args.num_ap, args.num_tp):
            successful_databases += 1
            total_queries += args.num_ap + args.num_tp
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("ADVANCED QUERY GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Databases processed: {successful_databases}/{len(args.databases)}")
    print(f"Total queries generated: {total_queries:,}")
    print(f"Time taken: {total_time:.1f} seconds")
    if total_time > 0:
        print(f"Queries per second: {total_queries/total_time:.1f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Advanced Benchmark Generation',
        'databases_processed': successful_databases,
        'databases_attempted': len(args.databases),
        'databases': args.databases,
        'ap_queries_per_db': args.num_ap,
        'tp_queries_per_db': args.num_tp,
        'total_queries': total_queries,
        'generation_time_seconds': total_time,
        'output_directory': str(args.output_dir),
        'features': {
            'complex_predicates_with_and_or': True,
            'max_joins_per_query': 3,
            'data_driven_literal_sampling': True,
            'having_clauses': True,
            'order_by_clauses': True,
            'limit_clauses': True,
            'aggregate_functions': True,
            'realistic_workloads': True
        }
    }
    
    with open(args.output_dir / 'advanced_generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {args.output_dir / 'advanced_generation_summary.json'}")

if __name__ == '__main__':
    main()