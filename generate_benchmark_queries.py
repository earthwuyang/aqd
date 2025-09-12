#!/usr/bin/env python3
"""
Benchmark Query Generator for PostgreSQL with pg_duckdb Extension
Generates AP (Analytical Processing) and TP (Transactional Processing) queries for CTU datasets

Based on:
- TiDB AQD (Automatic Query Diversification) methodology from ZSCE
- Cross-database benchmark tools for query routing optimization
"""

import os
import sys
import json
import time
import random
import argparse
import psycopg2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from enum import Enum
import collections

# Database configuration
POSTGRESQL_CONFIG = {
    'host': '/tmp',  # Unix socket
    'port': 5432,
    'user': os.environ.get('USER', 'postgres')
}

# Base directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'ctu_data'
OUTPUT_DIR = BASE_DIR / 'benchmark_queries'

# CTU Datasets
CTU_DATASETS = [
    'Airline',
    'Credit', 
    'Carcinogenesis',
    'employee',
    'financial',
    'geneea',
    'Hepatitis_std'
]

# TPC Benchmark Datasets
TPC_DATASETS = [
    'tpch_sf1',
    'tpcds_sf1'
]

# All datasets
ALL_DATASETS = CTU_DATASETS + TPC_DATASETS


class Datatype(Enum):
    INT = 'INT'
    FLOAT = 'FLOAT'
    CATEGORICAL = 'CATEGORICAL'
    TEXT = 'TEXT'
    BOOLEAN = 'BOOLEAN'
    DATE = 'DATE'
    TIMESTAMP = 'TIMESTAMP'
    MISC = 'MISC'


class Operator(Enum):
    EQ = '='
    NEQ = '!='
    LT = '<'
    LEQ = '<='
    GT = '>'
    GEQ = '>='
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NULL = 'IS NULL'
    IS_NOT_NULL = 'IS NOT NULL'
    IN = 'IN'
    NOT_IN = 'NOT IN'
    BETWEEN = 'BETWEEN'
    
    def __str__(self):
        return self.value


class Aggregator(Enum):
    COUNT = 'COUNT'
    SUM = 'SUM'
    AVG = 'AVG'
    MIN = 'MIN'
    MAX = 'MAX'
    STDDEV = 'STDDEV'
    VARIANCE = 'VARIANCE'
    
    def __str__(self):
        return self.value


class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'
    
    def __str__(self):
        return self.value


class DatabaseIntrospector:
    """Extract schema and statistics from PostgreSQL databases"""
    
    def __init__(self, database, use_cache=True):
        self.database = database
        self.conn = None
        self.schema_info = {}
        self.column_stats = {}
        self.use_cache = use_cache
        self.cache_dir = OUTPUT_DIR / self.database / '.cache'
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            config = POSTGRESQL_CONFIG.copy()
            config['database'] = self.database
            self.conn = psycopg2.connect(**config)
            return True
        except Exception as e:
            print(f"Failed to connect to {self.database}: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_tables(self):
        """Get all tables in the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    
    def get_columns(self, table):
        """Get column information for a table"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        columns = cursor.fetchall()
        cursor.close()
        return columns
    
    def get_primary_keys(self, table):
        """Get primary key columns for a table"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_schema = 'public'
            AND tc.table_name = %s
            AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
        """, (table,))
        pks = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return pks
    
    def get_foreign_keys(self):
        """Get all foreign key relationships in the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                kcu.table_name AS child_table,
                kcu.column_name AS child_column,
                ccu.table_name AS parent_table,
                ccu.column_name AS parent_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.table_schema = 'public'
            AND tc.constraint_type = 'FOREIGN KEY'
        """)
        fks = cursor.fetchall()
        cursor.close()
        return fks
    
    def get_column_statistics(self, table, column, data_type, quick_mode=False):
        """Get statistics for a column
        
        Args:
            table: Table name
            column: Column name
            data_type: PostgreSQL data type
            quick_mode: If True, skip expensive statistics calculations (for large tables)
        """
        cursor = self.conn.cursor()
        stats = {}
        
        # Map PostgreSQL types to our Datatype enum
        if data_type in ['integer', 'bigint', 'smallint', 'serial', 'bigserial']:
            dtype = Datatype.INT
        elif data_type in ['numeric', 'decimal', 'real', 'double precision', 'float']:
            dtype = Datatype.FLOAT
        elif data_type in ['boolean']:
            dtype = Datatype.BOOLEAN
        elif data_type in ['date']:
            dtype = Datatype.DATE
        elif data_type in ['timestamp', 'timestamp without time zone', 'timestamp with time zone']:
            dtype = Datatype.TIMESTAMP
        elif data_type in ['text', 'character varying', 'varchar', 'character', 'char']:
            dtype = Datatype.CATEGORICAL
        else:
            dtype = Datatype.MISC
        
        stats['datatype'] = dtype
        
        # For TPC datasets or quick mode, use simplified statistics
        if quick_mode or self.database in TPC_DATASETS:
            # Just get basic counts - much faster
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_count
                FROM "{table}"
                LIMIT 1
            """)
            result = cursor.fetchone()
            total_count = result[0] if result[0] else 1000000  # Assume large for TPC
            
            stats['total_count'] = total_count
            stats['non_null_count'] = int(total_count * 0.95)  # Assume 95% non-null
            stats['unique_count'] = min(total_count // 2, 10000)  # Reasonable estimate
            stats['nan_ratio'] = 0.05
            
            if dtype in [Datatype.INT, Datatype.FLOAT]:
                # Use simplified stats for numeric columns
                cursor.execute(f"""
                    SELECT 
                        MIN("{column}")::float as min_val,
                        MAX("{column}")::float as max_val
                    FROM "{table}"
                    WHERE "{column}" IS NOT NULL
                    LIMIT 1
                """)
                result = cursor.fetchone()
                
                if result:
                    min_val = float(result[0]) if result[0] is not None else 0
                    max_val = float(result[1]) if result[1] is not None else 100
                    stats['min'] = min_val
                    stats['max'] = max_val
                    stats['mean'] = (min_val + max_val) / 2
                    stats['std'] = (max_val - min_val) / 4
                    stats['percentiles'] = [min_val, min_val, (min_val + max_val) / 2, max_val, max_val]
            
            elif dtype == Datatype.CATEGORICAL:
                # Get just a few sample values
                cursor.execute(f"""
                    SELECT DISTINCT "{column}"
                    FROM "{table}"
                    WHERE "{column}" IS NOT NULL
                    LIMIT 10
                """)
                unique_vals = [row[0] for row in cursor.fetchall()]
                stats['unique_vals'] = unique_vals
        else:
            # Full statistics for smaller CTU datasets
            # Get basic statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT("{column}") as non_null_count,
                    COUNT(DISTINCT "{column}") as unique_count
                FROM "{table}"
            """)
            result = cursor.fetchone()
            
            total_count = result[0] if result[0] else 0
            non_null_count = result[1] if result[1] else 0
            unique_count = result[2] if result[2] else 0
            
            stats['total_count'] = total_count
            stats['non_null_count'] = non_null_count
            stats['unique_count'] = unique_count
            stats['nan_ratio'] = 1.0 - (non_null_count / total_count) if total_count > 0 else 1.0
            
            if dtype in [Datatype.INT, Datatype.FLOAT]:
                # Get numeric statistics with percentiles
                cursor.execute(f"""
                    SELECT 
                        MIN("{column}")::float as min_val,
                        MAX("{column}")::float as max_val,
                        AVG("{column}")::float as mean_val,
                        STDDEV("{column}")::float as std_val,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{column}") as q1,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{column}") as median,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{column}") as q3
                    FROM "{table}"
                    WHERE "{column}" IS NOT NULL
                """)
                result = cursor.fetchone()
                
                if result:
                    stats['min'] = float(result[0]) if result[0] is not None else 0
                    stats['max'] = float(result[1]) if result[1] is not None else 0
                    stats['mean'] = float(result[2]) if result[2] is not None else 0
                    stats['std'] = float(result[3]) if result[3] is not None else 0
                    stats['percentiles'] = [
                        stats['min'],
                        float(result[4]) if result[4] is not None else 0,
                        float(result[5]) if result[5] is not None else 0,
                        float(result[6]) if result[6] is not None else 0,
                        stats['max']
                    ]
            
            elif dtype == Datatype.CATEGORICAL:
                # Get sample unique values for categorical columns
                cursor.execute(f"""
                    SELECT DISTINCT "{column}"
                    FROM "{table}"
                    WHERE "{column}" IS NOT NULL
                    LIMIT 100
                """)
                unique_vals = [row[0] for row in cursor.fetchall()]
                stats['unique_vals'] = unique_vals
        
        cursor.close()
        return stats
    
    def load_cache(self):
        """Load cached analysis results if available"""
        if not self.use_cache:
            return False
            
        schema_cache_file = self.cache_dir / 'schema_info.json'
        stats_cache_file = self.cache_dir / 'column_statistics.json'
        
        if schema_cache_file.exists() and stats_cache_file.exists():
            try:
                # Load schema info
                with open(schema_cache_file, 'r') as f:
                    self.schema_info = json.load(f)
                
                # Load column statistics and convert datatype strings back to enums
                with open(stats_cache_file, 'r') as f:
                    stats_data = json.load(f)
                    self.column_stats = {}
                    for table, cols in stats_data.items():
                        self.column_stats[table] = {}
                        for col, stats in cols.items():
                            if 'datatype' in stats and isinstance(stats['datatype'], str):
                                # Convert string back to Datatype enum
                                stats['datatype'] = Datatype[stats['datatype']]
                            self.column_stats[table][col] = stats
                
                print(f"  Loaded cached analysis from {self.cache_dir}")
                return True
            except Exception as e:
                print(f"  Warning: Failed to load cache: {e}")
                return False
        return False
    
    def save_cache(self):
        """Save analysis results to cache"""
        if not self.use_cache:
            return
            
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save schema info
            schema_cache_file = self.cache_dir / 'schema_info.json'
            with open(schema_cache_file, 'w') as f:
                # Convert foreign keys to serializable format
                schema_copy = self.schema_info.copy()
                if 'relationships' in schema_copy:
                    schema_copy['relationships'] = [
                        {
                            'child_table': r[0] if isinstance(r, tuple) else r.get('child_table'),
                            'child_column': r[1] if isinstance(r, tuple) else r.get('child_column'),
                            'parent_table': r[2] if isinstance(r, tuple) else r.get('parent_table'),
                            'parent_column': r[3] if isinstance(r, tuple) else r.get('parent_column')
                        }
                        for r in self.schema_info['relationships']
                    ]
                json.dump(schema_copy, f, indent=2, default=str)
            
            # Save column statistics
            stats_cache_file = self.cache_dir / 'column_statistics.json'
            with open(stats_cache_file, 'w') as f:
                # Convert Datatype enums to strings
                stats_copy = {}
                for table, cols in self.column_stats.items():
                    stats_copy[table] = {}
                    for col, stats in cols.items():
                        stats_dict = stats.copy()
                        if 'datatype' in stats_dict and isinstance(stats_dict['datatype'], Datatype):
                            stats_dict['datatype'] = stats_dict['datatype'].name
                        stats_copy[table][col] = stats_dict
                json.dump(stats_copy, f, indent=2, default=str)
            
            print(f"  Saved analysis cache to {self.cache_dir}")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")
    
    def analyze_database(self):
        """Analyze complete database schema and statistics"""
        print(f"Analyzing database: {self.database}")
        
        # Try to load from cache first
        if self.load_cache():
            print(f"  Using cached analysis for {self.database}")
            return True
        
        if not self.connect():
            return False
        
        # Get tables
        tables = self.get_tables()
        print(f"  Found {len(tables)} tables")
        
        # Get foreign keys
        foreign_keys = self.get_foreign_keys()
        
        # Build schema info
        self.schema_info = {
            'database': self.database,
            'tables': {},
            'relationships': foreign_keys,
            'primary_keys': {}
        }
        
        # Analyze each table
        for table in tqdm(tables, desc="Analyzing tables"):
            # Get columns
            columns = self.get_columns(table)
            self.schema_info['tables'][table] = columns
            
            # Get primary keys
            pks = self.get_primary_keys(table)
            self.schema_info['primary_keys'][table] = pks
            
            # Get column statistics
            self.column_stats[table] = {}
            # Use quick mode for TPC datasets to avoid slow percentile calculations
            quick_mode = self.database in TPC_DATASETS
            for col_name, data_type, is_nullable, default in columns:
                try:
                    stats = self.get_column_statistics(table, col_name, data_type, quick_mode=quick_mode)
                    self.column_stats[table][col_name] = stats
                except Exception as e:
                    print(f"    Warning: Failed to get stats for {table}.{col_name}: {e}")
        
        self.close()
        
        # Save cache for future use
        self.save_cache()
        
        return True
    
    def save_metadata(self):
        """Save schema and statistics to files"""
        output_dir = OUTPUT_DIR / self.database
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save schema info
        schema_path = output_dir / 'schema_info.json'
        with open(schema_path, 'w') as f:
            # Convert foreign keys to serializable format
            schema_copy = self.schema_info.copy()
            if 'relationships' in schema_copy:
                schema_copy['relationships'] = [
                    {
                        'child_table': r[0] if isinstance(r, tuple) else r.get('child_table'),
                        'child_column': r[1] if isinstance(r, tuple) else r.get('child_column'),
                        'parent_table': r[2] if isinstance(r, tuple) else r.get('parent_table'),
                        'parent_column': r[3] if isinstance(r, tuple) else r.get('parent_column')
                    }
                    for r in self.schema_info['relationships']
                ]
            json.dump(schema_copy, f, indent=2, default=str)
        
        # Save column statistics
        stats_path = output_dir / 'column_statistics.json'
        with open(stats_path, 'w') as f:
            # Convert Datatype enums to strings
            stats_copy = {}
            for table, cols in self.column_stats.items():
                stats_copy[table] = {}
                for col, stats in cols.items():
                    stats_dict = stats.copy()
                    if 'datatype' in stats_dict:
                        stats_dict['datatype'] = stats_dict['datatype'].value
                    stats_copy[table][col] = stats_dict
            json.dump(stats_copy, f, indent=2, default=str)
        
        print(f"  Saved metadata to {output_dir}")


class QueryGenerator:
    """Base class for query generation"""
    
    def __init__(self, schema_info, column_stats):
        self.schema_info = schema_info
        self.column_stats = column_stats
        self.tables = list(schema_info['tables'].keys())
        self.relationships = self._build_relationship_index()
    
    def _build_relationship_index(self):
        """Build index of relationships for efficient lookup"""
        rel_index = collections.defaultdict(list)
        for rel in self.schema_info['relationships']:
            child_table = rel[0] if isinstance(rel, tuple) else rel['child_table']
            child_col = rel[1] if isinstance(rel, tuple) else rel['child_column']
            parent_table = rel[2] if isinstance(rel, tuple) else rel['parent_table']
            parent_col = rel[3] if isinstance(rel, tuple) else rel['parent_column']
            
            # Add bidirectional relationships
            rel_index[child_table].append({
                'table': parent_table,
                'join_col': child_col,
                'ref_col': parent_col
            })
            rel_index[parent_table].append({
                'table': child_table,
                'join_col': parent_col,
                'ref_col': child_col
            })
        
        return rel_index
    
    def rand_choice(self, items, n=None):
        """Random choice from list"""
        if n is None:
            return random.choice(items) if items else None
        else:
            return random.sample(items, min(n, len(items)))
    
    def sample_literal(self, stats, operator=None):
        """Sample a literal value based on column statistics"""
        if not stats:
            return None
        
        dtype = stats.get('datatype')
        if isinstance(dtype, str):
            dtype = Datatype[dtype]
        
        if dtype in [Datatype.INT, Datatype.FLOAT]:
            if 'percentiles' in stats and stats['percentiles']:
                # Sample from percentile range
                percentiles = stats['percentiles']
                idx = random.randint(0, len(percentiles) - 2)
                val = random.uniform(percentiles[idx], percentiles[idx + 1])
                if dtype == Datatype.INT:
                    val = int(val)
                return val
            elif 'min' in stats and 'max' in stats:
                val = random.uniform(stats['min'], stats['max'])
                if dtype == Datatype.INT:
                    val = int(val)
                return val
        
        elif dtype == Datatype.CATEGORICAL:
            if 'unique_vals' in stats and stats['unique_vals']:
                return f"'{random.choice(stats['unique_vals'])}'"
        
        elif dtype == Datatype.BOOLEAN:
            return random.choice(['TRUE', 'FALSE'])
        
        return None


class APQueryGenerator(QueryGenerator):
    """Generate Analytical Processing (OLAP) queries"""
    
    def generate_query(self, seed=None):
        """Generate a single AP query"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Select tables to join (1-5 tables)
        num_tables = random.randint(1, min(5, len(self.tables)))
        selected_tables = self.rand_choice(self.tables, num_tables)
        
        # Build FROM clause with JOINs
        from_clause = self._build_from_clause(selected_tables)
        
        # Build SELECT clause with aggregations
        select_clause = self._build_select_clause(selected_tables)
        
        # Build WHERE clause
        where_clause = self._build_where_clause(selected_tables)
        
        # Build GROUP BY clause
        group_by_clause = self._build_group_by_clause(selected_tables)
        
        # Build HAVING clause (if GROUP BY exists)
        having_clause = ""
        if group_by_clause and random.random() < 0.3:
            having_clause = self._build_having_clause()
        
        # Build ORDER BY clause
        order_by_clause = ""
        if random.random() < 0.5:
            order_by_clause = self._build_order_by_clause(group_by_clause)
        
        # Build LIMIT clause
        limit_clause = ""
        if random.random() < 0.3:
            limit_clause = f" LIMIT {random.choice([10, 100, 1000])}"
        
        # Construct query
        query = f"SELECT {select_clause}"
        query += f" FROM {from_clause}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if group_by_clause:
            query += f" GROUP BY {group_by_clause}"
        if having_clause:
            query += f" HAVING {having_clause}"
        if order_by_clause:
            query += f" ORDER BY {order_by_clause}"
        query += limit_clause + ";"
        
        return query
    
    def _build_from_clause(self, tables):
        """Build FROM clause with JOINs"""
        if len(tables) == 1:
            return f'"{tables[0]}"'
        
        # Start with first table
        from_clause = f'"{tables[0]}"'
        joined_tables = {tables[0]}
        
        # Try to join remaining tables
        for table in tables[1:]:
            # Find join path
            join_found = False
            for joined_table in joined_tables:
                if table in [r['table'] for r in self.relationships.get(joined_table, [])]:
                    # Found direct relationship
                    rel = next(r for r in self.relationships[joined_table] if r['table'] == table)
                    join_type = random.choice(['JOIN', 'LEFT JOIN', 'INNER JOIN'])
                    from_clause += f' {join_type} "{table}" ON "{joined_table}"."{rel["join_col"]}" = "{table}"."{rel["ref_col"]}"'
                    joined_tables.add(table)
                    join_found = True
                    break
            
            if not join_found:
                # Cross join if no relationship found
                from_clause += f' CROSS JOIN "{table}"'
                joined_tables.add(table)
        
        return from_clause
    
    def _build_select_clause(self, tables):
        """Build SELECT clause with aggregations"""
        select_items = []
        
        # Add aggregations
        num_aggs = random.randint(1, 5)
        for _ in range(num_aggs):
            agg = random.choice(list(Aggregator))
            
            if agg == Aggregator.COUNT:
                select_items.append("COUNT(*)")
            else:
                # Find numeric columns
                numeric_cols = []
                for table in tables:
                    for col, stats in self.column_stats.get(table, {}).items():
                        dtype = stats.get('datatype')
                        if isinstance(dtype, str):
                            dtype = Datatype[dtype]
                        if dtype in [Datatype.INT, Datatype.FLOAT]:
                            numeric_cols.append(f'"{table}"."{col}"')
                
                if numeric_cols:
                    col = random.choice(numeric_cols)
                    select_items.append(f"{agg}({col})")
        
        # Add some regular columns (for GROUP BY)
        if random.random() < 0.5:
            num_cols = random.randint(1, 3)
            for _ in range(num_cols):
                table = random.choice(tables)
                cols = list(self.column_stats.get(table, {}).keys())
                if cols:
                    col = random.choice(cols)
                    select_items.append(f'"{table}"."{col}"')
        
        return ", ".join(select_items) if select_items else "COUNT(*)"
    
    def _build_where_clause(self, tables):
        """Build WHERE clause"""
        predicates = []
        num_predicates = random.randint(0, 5)
        
        for _ in range(num_predicates):
            table = random.choice(tables)
            cols = list(self.column_stats.get(table, {}).keys())
            if not cols:
                continue
            
            col = random.choice(cols)
            stats = self.column_stats[table][col]
            dtype = stats.get('datatype')
            if isinstance(dtype, str):
                dtype = Datatype[dtype]
            
            # Select appropriate operator
            if dtype in [Datatype.INT, Datatype.FLOAT]:
                op = random.choice([Operator.EQ, Operator.NEQ, Operator.LT, Operator.LEQ, Operator.GT, Operator.GEQ])
                literal = self.sample_literal(stats)
                if literal is not None:
                    predicates.append(f'"{table}"."{col}" {op} {literal}')
            
            elif dtype == Datatype.CATEGORICAL:
                if random.random() < 0.5:
                    op = random.choice([Operator.EQ, Operator.NEQ])
                    literal = self.sample_literal(stats)
                    if literal:
                        predicates.append(f'"{table}"."{col}" {op} {literal}')
                else:
                    # LIKE operator
                    predicates.append(f'"{table}"."{col}" LIKE \'%{random.choice(["a", "e", "i", "o", "u"])}%\'')
            
            elif dtype == Datatype.BOOLEAN:
                predicates.append(f'"{table}"."{col}" = {random.choice(["TRUE", "FALSE"])}')
            
            # Add NULL checks occasionally
            if random.random() < 0.2:
                op = random.choice([Operator.IS_NULL, Operator.IS_NOT_NULL])
                predicates.append(f'"{table}"."{col}" {op}')
        
        if predicates:
            # Combine with AND/OR
            if len(predicates) > 1 and random.random() < 0.3:
                # Mix AND and OR
                result = predicates[0]
                for pred in predicates[1:]:
                    op = random.choice([' AND ', ' OR '])
                    result += op + pred
                return result
            else:
                # All AND
                return ' AND '.join(predicates)
        
        return ""
    
    def _build_group_by_clause(self, tables):
        """Build GROUP BY clause"""
        if random.random() < 0.5:
            return ""
        
        group_cols = []
        num_cols = random.randint(1, 3)
        
        for _ in range(num_cols):
            table = random.choice(tables)
            # Prefer categorical columns for grouping
            best_cols = []
            for col, stats in self.column_stats.get(table, {}).items():
                dtype = stats.get('datatype')
                if isinstance(dtype, str):
                    dtype = Datatype[dtype]
                if dtype == Datatype.CATEGORICAL or (dtype == Datatype.INT and stats.get('unique_count', 0) < 1000):
                    best_cols.append(col)
            
            if best_cols:
                col = random.choice(best_cols)
                group_cols.append(f'"{table}"."{col}"')
        
        return ", ".join(group_cols)
    
    def _build_having_clause(self):
        """Build HAVING clause"""
        agg = random.choice(['COUNT(*)', 'SUM(1)', 'AVG(1)'])
        op = random.choice(['>', '>=', '<', '<=', '='])
        val = random.randint(1, 100)
        return f"{agg} {op} {val}"
    
    def _build_order_by_clause(self, group_by_clause):
        """Build ORDER BY clause"""
        if group_by_clause:
            # Order by grouped columns
            if random.random() < 0.5:
                return group_by_clause + random.choice([' ASC', ' DESC', ''])
        
        # Order by aggregate
        return f"COUNT(*) {random.choice(['ASC', 'DESC'])}"


class TPQueryGenerator(QueryGenerator):
    """Generate Transactional Processing (OLTP) queries"""
    
    def generate_query(self, seed=None):
        """Generate a single TP query"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Select tables (1-3 for TP queries)
        num_tables = random.randint(1, min(3, len(self.tables)))
        selected_tables = self.rand_choice(self.tables, num_tables)
        
        # Build FROM clause
        from_clause = self._build_from_clause(selected_tables)
        
        # Build SELECT clause (specific columns)
        select_clause = self._build_select_clause(selected_tables)
        
        # Build WHERE clause (point queries)
        where_clause = self._build_where_clause(selected_tables)
        
        # Build ORDER BY clause
        order_by_clause = ""
        if random.random() < 0.3:
            order_by_clause = self._build_order_by_clause(selected_tables)
        
        # Build LIMIT clause (common in TP)
        limit_clause = f" LIMIT {random.choice([1, 10, 100])}"
        
        # Construct query
        query = f"SELECT {select_clause}"
        query += f" FROM {from_clause}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if order_by_clause:
            query += f" ORDER BY {order_by_clause}"
        query += limit_clause + ";"
        
        return query
    
    def _build_from_clause(self, tables):
        """Build FROM clause with minimal JOINs"""
        if len(tables) == 1:
            return f'"{tables[0]}"'
        
        # Similar to AP but prefer INNER JOIN
        from_clause = f'"{tables[0]}"'
        joined_tables = {tables[0]}
        
        for table in tables[1:]:
            join_found = False
            for joined_table in joined_tables:
                if table in [r['table'] for r in self.relationships.get(joined_table, [])]:
                    rel = next(r for r in self.relationships[joined_table] if r['table'] == table)
                    from_clause += f' INNER JOIN "{table}" ON "{joined_table}"."{rel["join_col"]}" = "{table}"."{rel["ref_col"]}"'
                    joined_tables.add(table)
                    join_found = True
                    break
            
            if not join_found:
                # Skip table if no join found (keep query simple)
                continue
        
        return from_clause
    
    def _build_select_clause(self, tables):
        """Build SELECT clause with specific columns"""
        select_items = []
        
        # Select specific columns (not aggregations)
        for table in tables:
            cols = list(self.column_stats.get(table, {}).keys())
            if cols:
                # Select 1-3 columns per table
                num_cols = random.randint(1, min(3, len(cols)))
                selected_cols = random.sample(cols, num_cols)
                for col in selected_cols:
                    select_items.append(f'"{table}"."{col}"')
        
        return ", ".join(select_items) if select_items else "*"
    
    def _build_where_clause(self, tables):
        """Build WHERE clause with point query predicates"""
        predicates = []
        
        # Focus on high-selectivity predicates (equality on unique columns)
        for table in tables:
            # Look for primary key or high-cardinality columns
            pks = self.schema_info.get('primary_keys', {}).get(table, [])
            
            if pks and random.random() < 0.7:
                # Use primary key
                for pk in pks:
                    stats = self.column_stats.get(table, {}).get(pk, {})
                    literal = self.sample_literal(stats)
                    if literal is not None:
                        predicates.append(f'"{table}"."{pk}" = {literal}')
                        break
            else:
                # Use high-cardinality column
                high_card_cols = []
                for col, stats in self.column_stats.get(table, {}).items():
                    if stats.get('unique_count', 0) > 100:
                        high_card_cols.append((col, stats))
                
                if high_card_cols:
                    col, stats = random.choice(high_card_cols)
                    literal = self.sample_literal(stats)
                    if literal is not None:
                        predicates.append(f'"{table}"."{col}" = {literal}')
        
        # Add additional filters
        if random.random() < 0.3:
            table = random.choice(tables)
            cols = list(self.column_stats.get(table, {}).keys())
            if cols:
                col = random.choice(cols)
                stats = self.column_stats[table][col]
                dtype = stats.get('datatype')
                if isinstance(dtype, str):
                    dtype = Datatype[dtype]
                
                if dtype in [Datatype.INT, Datatype.FLOAT]:
                    op = random.choice([Operator.LT, Operator.LEQ, Operator.GT, Operator.GEQ])
                    literal = self.sample_literal(stats)
                    if literal is not None:
                        predicates.append(f'"{table}"."{col}" {op} {literal}')
        
        return ' AND '.join(predicates)
    
    def _build_order_by_clause(self, tables):
        """Build ORDER BY clause"""
        table = random.choice(tables)
        cols = list(self.column_stats.get(table, {}).keys())
        if cols:
            col = random.choice(cols)
            direction = random.choice(['ASC', 'DESC'])
            return f'"{table}"."{col}" {direction}'
        return ""


def generate_queries_for_database(database, num_ap=1000, num_tp=1000, use_cache=True):
    """Generate queries for a single database"""
    print(f"\n{'='*60}")
    print(f"Generating queries for: {database}")
    print(f"{'='*60}")
    
    # Analyze database
    introspector = DatabaseIntrospector(database, use_cache=use_cache)
    if not introspector.analyze_database():
        print(f"Failed to analyze {database}")
        return False
    
    # Save metadata
    introspector.save_metadata()
    
    # Generate AP queries
    print(f"\nGenerating {num_ap} AP queries...")
    ap_generator = APQueryGenerator(introspector.schema_info, introspector.column_stats)
    ap_queries = []
    
    for i in tqdm(range(num_ap), desc="AP queries"):
        try:
            query = ap_generator.generate_query(seed=i)
            ap_queries.append(query)
        except Exception as e:
            if i < 10:  # Only show first few errors
                print(f"  Error generating AP query {i}: {e}")
    
    # Generate TP queries
    print(f"\nGenerating {num_tp} TP queries...")
    tp_generator = TPQueryGenerator(introspector.schema_info, introspector.column_stats)
    tp_queries = []
    
    for i in tqdm(range(num_tp), desc="TP queries"):
        try:
            query = tp_generator.generate_query(seed=i + num_ap)
            tp_queries.append(query)
        except Exception as e:
            if i < 10:  # Only show first few errors
                print(f"  Error generating TP query {i}: {e}")
    
    # Save queries
    output_dir = OUTPUT_DIR / database
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ap_file = output_dir / 'workload_ap_queries.sql'
    with open(ap_file, 'w') as f:
        f.write('\n'.join(ap_queries))
    print(f"  Saved {len(ap_queries)} AP queries to {ap_file}")
    
    tp_file = output_dir / 'workload_tp_queries.sql'
    with open(tp_file, 'w') as f:
        f.write('\n'.join(tp_queries))
    print(f"  Saved {len(tp_queries)} TP queries to {tp_file}")
    
    # Save generation metadata
    metadata = {
        'database': database,
        'timestamp': datetime.now().isoformat(),
        'ap_queries': len(ap_queries),
        'tp_queries': len(tp_queries),
        'total_queries': len(ap_queries) + len(tp_queries),
        'tables': len(introspector.schema_info['tables']),
        'relationships': len(introspector.schema_info['relationships']),
        'ap_file': str(ap_file),
        'tp_file': str(tp_file)
    }
    
    metadata_file = output_dir / 'generation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Total queries generated: {len(ap_queries) + len(tp_queries)}")
    return True


def main():
    global OUTPUT_DIR
    
    parser = argparse.ArgumentParser(
        description='Generate benchmark queries for PostgreSQL with pg_duckdb'
    )
    parser.add_argument(
        '--databases', 
        nargs='+',
        default=ALL_DATASETS,
        help='Databases to generate queries for'
    )
    parser.add_argument(
        '--num-ap',
        type=int,
        default=1000,
        help='Number of AP queries per database'
    )
    parser.add_argument(
        '--num-tp',
        type=int,
        default=1000,
        help='Number of TP queries per database'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for queries'
    )
    parser.add_argument(
        '--tpc-only',
        action='store_true',
        help='Generate queries only for TPC-H and TPC-DS databases'
    )
    parser.add_argument(
        '--ctu-only',
        action='store_true',
        help='Generate queries only for CTU databases'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached analysis results (force re-analysis)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached analysis results before generating queries'
    )
    
    args = parser.parse_args()
    
    # Determine which databases to use based on flags
    if args.tpc_only:
        args.databases = TPC_DATASETS
    elif args.ctu_only:
        args.databases = CTU_DATASETS
    # Otherwise use the provided databases or default (ALL_DATASETS)
    
    # Update output directory
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle cache clearing if requested
    if args.clear_cache:
        print("Clearing all cached analysis results...")
        import shutil
        for database in ALL_DATASETS:
            cache_dir = OUTPUT_DIR / database / '.cache'
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"  Cleared cache for {database}")
    
    print("="*60)
    print("Benchmark Query Generator for PostgreSQL + pg_duckdb")
    print("Based on TiDB AQD methodology")
    print("Supports CTU datasets and TPC-H/TPC-DS benchmarks")
    print("="*60)
    
    # Check which databases exist
    available_dbs = []
    for db in args.databases:
        config = POSTGRESQL_CONFIG.copy()
        config['database'] = db
        try:
            conn = psycopg2.connect(**config)
            conn.close()
            available_dbs.append(db)
        except:
            print(f"Warning: Database {db} not accessible")
    
    if not available_dbs:
        print("No accessible databases found")
        return 1
    
    print(f"\nDatabases to process: {available_dbs}")
    print(f"Queries per database: {args.num_ap} AP, {args.num_tp} TP")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Generate queries for each database
    start_time = time.time()
    successful = 0
    
    for db in available_dbs:
        if generate_queries_for_database(db, args.num_ap, args.num_tp, use_cache=not args.no_cache):
            successful += 1
    
    # Summary
    duration = time.time() - start_time
    total_queries = successful * (args.num_ap + args.num_tp)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Databases processed: {successful}/{len(available_dbs)}")
    print(f"Total queries generated: {total_queries:,}")
    print(f"Time taken: {duration:.1f} seconds")
    print(f"Queries per second: {total_queries/duration:.1f}")
    
    # Save overall summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'databases_processed': successful,
        'databases_attempted': len(available_dbs),
        'databases': available_dbs,
        'ap_queries_per_db': args.num_ap,
        'tp_queries_per_db': args.num_tp,
        'total_queries': total_queries,
        'generation_time_seconds': duration,
        'output_directory': str(OUTPUT_DIR)
    }
    
    summary_file = OUTPUT_DIR / 'generation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())