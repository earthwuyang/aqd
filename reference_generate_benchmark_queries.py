#!/usr/bin/env python3
"""
Benchmark Query Generator for PostgreSQL/DuckDB Query Router Training
Based on TiDB AQD (Automatic Query Diversification) methodology from ZSCE
Generates 10K AP (Analytical Processing) + 10K TP (Transactional Processing) queries per dataset

References:
- ~/DB/tidb/aqd/src/zsce/generate_zsce_queries.py (AP query generation)
- ~/DB/tidb/aqd/src/zsce/cross_db_benchmark/benchmark_tools/generate_TP_workload.py (TP query generation)
- ~/DB/tidb/aqd/src/zsce/cross_db_benchmark/benchmark_tools/generate_workload.py (Core AP generation)
"""

import os
import json
import time
import collections
import multiprocessing as mp
from enum import Enum
from functools import partial
import argparse
import numpy as np
import psycopg2
from tqdm import tqdm
import sqlite3

# Import DuckDB for schema introspection
import sys
sys.path.insert(0, '/home/wuy/DB/duckdb/duckdb_src/build/release/tools/pythonpkg')
import duckdb

# Database connection configuration
POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'benchmark_datasets',
    'user': 'postgres',
    'password': 'postgres'
}

DUCKDB_PATH = '/data/wuy/db/benchmark_datasets.db'


class Datatype(Enum):
    INT = 'INT'
    FLOAT = 'FLOAT'
    CATEGORICAL = 'CATEGORICAL'
    MISC = 'MISC'
    VARCHAR = 'VARCHAR'
    TEXT = 'TEXT'


class Operator(Enum):
    NEQ = '!='
    EQ = '='
    LEQ = '<='
    GEQ = '>='
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NOT_NULL = 'IS NOT NULL'
    IS_NULL = 'IS NULL'
    IN = 'IN'
    BETWEEN = 'BETWEEN'

    def __str__(self):
        return self.value


class Aggregator(Enum):
    AVG = 'AVG'
    SUM = 'SUM'
    COUNT = 'COUNT'
    MIN = 'MIN'
    MAX = 'MAX'

    def __str__(self):
        return self.value


class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'

    def __str__(self):
        return self.value


def rand_choice(randstate, l, no_elements=None, replace=False):
    """Helper function to sample from a list with a given random state."""
    if no_elements is None:
        idx = randstate.randint(0, len(l))
        return l[idx]
    else:
        idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
        return [l[i] for i in idxs]


class ColumnPredicate:
    """Represents a single predicate like table.col operator literal"""

    def __init__(self, table, col_name, operator, literal):
        self.table = table
        self.col_name = col_name
        self.operator = operator
        self.literal = literal

    def __str__(self):
        return self.to_sql(top_operator=True)

    def to_sql(self, top_operator=False):
        if self.operator == Operator.IS_NOT_NULL:
            predicates_str = f'{self.table}.{self.col_name} IS NOT NULL'
        elif self.operator == Operator.IS_NULL:
            predicates_str = f'{self.table}.{self.col_name} IS NULL'
        else:
            predicates_str = f'{self.table}.{self.col_name} {str(self.operator)} {self.literal}'

        if top_operator:
            predicates_str = f' WHERE {predicates_str}'

        return predicates_str


class PredicateOperator:
    """Represents a logical AND/OR combination of child predicates."""

    def __init__(self, logical_op, children=None):
        self.logical_op = logical_op
        if children is None:
            children = []
        self.children = children

    def __str__(self):
        return self.to_sql(top_operator=True)

    def to_sql(self, top_operator=False):
        sql = ''
        if len(self.children) > 0:
            predicates_str_list = [c.to_sql() for c in self.children]
            sql = f' {str(self.logical_op)} '.join(predicates_str_list)

            if top_operator:
                sql = f' WHERE {sql}'
            elif len(self.children) > 1:
                sql = f'({sql})'

        return sql


class GenQueryAP:
    """Represents an Analytical Processing (OLAP) query with aggregations, group by, joins, etc."""

    def __init__(self, aggregations, group_bys, joins, predicates, start_t, join_tables, 
                 alias_dict=None, limit=None, having_clause=None):
        if alias_dict is None:
            alias_dict = dict()
        self.aggregations = aggregations
        self.group_bys = group_bys
        self.joins = joins
        self.predicates = predicates
        self.start_t = start_t
        self.join_tables = join_tables
        self.alias_dict = alias_dict
        self.limit = limit
        self.having_clause = having_clause

    def generate_sql_query(self, semicolon=True):
        # Group by clause
        group_by_str = ''
        order_by_str = ''
        group_by_cols = []
        
        if len(self.group_bys) > 0:
            group_by_cols = [f'{table}.{column}' for table, column, _ in self.group_bys]
            group_by_col_str = ', '.join(group_by_cols)
            group_by_str = f' GROUP BY {group_by_col_str}'
            order_by_str = f' ORDER BY {group_by_col_str}'

        # Aggregations
        aggregation_str_list = []
        for i, (aggregator, columns) in enumerate(self.aggregations):
            if aggregator == Aggregator.COUNT:
                aggregation_str_list.append('COUNT(*)')
            else:
                agg_cols = ' + '.join([f'{table}.{col}' for table, col in columns])
                aggregation_str_list.append(f'{str(aggregator)}({agg_cols})')
                
        aggregation_str = ', '.join(group_by_cols + [f'{agg} as agg_{i}' for i, agg in enumerate(aggregation_str_list)])
        if aggregation_str == '':
            aggregation_str = '*'

        # Having clause
        having_str = ""
        if self.having_clause is not None:
            idx, literal, op = self.having_clause
            having_str = f" HAVING {aggregation_str_list[idx]} {str(op)} {literal}"

        # Predicates
        predicate_str = str(self.predicates)

        # Joins
        join_str = self.start_t
        for table_l, column_l, table_r, column_r, left_outer in self.joins:
            join_kw = "JOIN" if not left_outer else "LEFT OUTER JOIN"
            join_str += f' {join_kw} {table_r}'
            join_cond = ' AND '.join([f'{table_l}.{col_l} = {table_r}.{col_r}'
                                     for col_l, col_r in zip(column_l, column_r)])
            join_str += f' ON {join_cond}'

        # Limit
        limit_str = ""
        if self.limit is not None:
            limit_str = f" LIMIT {self.limit}"

        sql_query = f"SELECT {aggregation_str} FROM {join_str} {predicate_str}{group_by_str}{having_str}{order_by_str}{limit_str}".strip()

        if semicolon:
            sql_query += ';'

        return sql_query


class GenQueryTP:
    """Represents a Transactional Processing (OLTP) point query for row-based execution"""

    def __init__(self, select_columns, joins, predicates, start_table, join_tables, 
                 alias_dict=None, limit=None):
        if alias_dict is None:
            alias_dict = dict()
        self.select_columns = select_columns
        self.joins = joins
        self.predicates = predicates
        self.start_table = start_table
        self.join_tables = join_tables
        self.alias_dict = alias_dict
        self.limit = limit

    def generate_sql_query(self, semicolon=True):
        # Build SELECT list
        if len(self.select_columns) == 0:
            select_str = '*'
        else:
            selected = [f'{tbl}.{col}' for (tbl, col) in self.select_columns]
            select_str = ', '.join(selected)

        # Build FROM + JOINs
        join_str = self.start_table
        for (table_l, column_l, table_r, column_r, left_outer) in self.joins:
            join_kw = "LEFT OUTER JOIN" if left_outer else "JOIN"
            join_str += f" {join_kw} {table_r} ON "
            join_cond = []
            for col_l, col_r in zip(column_l, column_r):
                join_cond.append(f'{table_l}.{col_l} = {table_r}.{col_r}')
            join_str += ' AND '.join(join_cond)

        # Build predicates
        predicate_str = str(self.predicates)

        # Build limit
        limit_str = f" LIMIT {self.limit}" if self.limit is not None else ""

        sql_query = f"SELECT {select_str} FROM {join_str}{predicate_str}{limit_str}"
        if semicolon:
            sql_query += ";"
        return sql_query


class DatabaseIntrospector:
    """Extract schema and statistics from PostgreSQL and DuckDB"""
    
    def __init__(self):
        self.pg_conn = None
        self.duck_conn = None
        
    def connect_postgresql(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
            return True
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            return False
    
    def connect_duckdb(self):
        """Connect to DuckDB"""
        try:
            self.duck_conn = duckdb.connect(DUCKDB_PATH)
            return True
        except Exception as e:
            print(f"DuckDB connection failed: {e}")
            return False
    
    def get_schemas(self):
        """Get list of available schemas (datasets)"""
        if not self.connect_duckdb():
            return []
            
        try:
            # Get schemas that are not system schemas
            schemas = self.duck_conn.execute("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('information_schema', 'main', 'temp', 'pg_catalog')
            """).fetchall()
            return [s[0] for s in schemas]
        except Exception as e:
            print(f"Error getting schemas: {e}")
            return []
    
    def get_schema_tables(self, schema_name):
        """Get tables in a schema"""
        if not self.duck_conn:
            return []
            
        try:
            tables = self.duck_conn.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema_name}'
            """).fetchall()
            return [t[0] for t in tables]
        except Exception as e:
            print(f"Error getting tables for {schema_name}: {e}")
            return []
    
    def get_table_columns(self, schema_name, table_name):
        """Get column information for a table"""
        if not self.duck_conn:
            return []
            
        try:
            columns = self.duck_conn.execute(f"""
                SELECT 
                    column_name, 
                    data_type,
                    is_nullable
                FROM information_schema.columns 
                WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()
            return [(c[0], c[1], c[2]) for c in columns]
        except Exception as e:
            print(f"Error getting columns for {schema_name}.{table_name}: {e}")
            return []
    
    def get_column_statistics(self, schema_name, table_name, column_name, data_type):
        """Get statistics for a column"""
        if not self.duck_conn:
            return None
            
        try:
            # Get basic stats
            if data_type.upper() in ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DOUBLE', 'REAL', 'FLOAT', 'DECIMAL', 'NUMERIC']:
                stats = self.duck_conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT({column_name}) as non_null_count,
                        COUNT(DISTINCT {column_name}) as unique_count,
                        MIN({column_name}) as min_val,
                        MAX({column_name}) as max_val,
                        AVG({column_name}) as mean_val
                    FROM {schema_name}.{table_name}
                """).fetchone()
                
                # Get percentiles for numeric columns
                percentiles = self.duck_conn.execute(f"""
                    SELECT 
                        percentile_cont(0.05) WITHIN GROUP (ORDER BY {column_name}) as p5,
                        percentile_cont(0.25) WITHIN GROUP (ORDER BY {column_name}) as p25,
                        percentile_cont(0.5) WITHIN GROUP (ORDER BY {column_name}) as p50,
                        percentile_cont(0.75) WITHIN GROUP (ORDER BY {column_name}) as p75,
                        percentile_cont(0.95) WITHIN GROUP (ORDER BY {column_name}) as p95
                    FROM {schema_name}.{table_name}
                    WHERE {column_name} IS NOT NULL
                """).fetchone()
                
                return {
                    'datatype': 'INT' if 'INT' in data_type.upper() else 'FLOAT',
                    'num_unique': stats[2] if stats[2] else 0,
                    'nan_ratio': 1.0 - (stats[1] / stats[0]) if stats[0] > 0 else 1.0,
                    'mean': float(stats[5]) if stats[5] else 0.0,
                    'min': float(stats[3]) if stats[3] else 0.0,
                    'max': float(stats[4]) if stats[4] else 0.0,
                    'percentiles': [p for p in percentiles] if percentiles else [0, 0, 0, 0, 0]
                }
            else:
                # String/categorical columns
                stats = self.duck_conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT({column_name}) as non_null_count,
                        COUNT(DISTINCT {column_name}) as unique_count
                    FROM {schema_name}.{table_name}
                """).fetchone()
                
                # Get sample unique values
                unique_vals = self.duck_conn.execute(f"""
                    SELECT DISTINCT {column_name}
                    FROM {schema_name}.{table_name}
                    WHERE {column_name} IS NOT NULL
                    LIMIT 20
                """).fetchall()
                
                return {
                    'datatype': 'CATEGORICAL',
                    'num_unique': stats[2] if stats[2] else 0,
                    'nan_ratio': 1.0 - (stats[1] / stats[0]) if stats[0] > 0 else 1.0,
                    'unique_vals': [v[0] for v in unique_vals]
                }
                
        except Exception as e:
            print(f"Error getting stats for {schema_name}.{table_name}.{column_name}: {e}")
            return None
    
    def analyze_dataset(self, schema_name):
        """Analyze a complete dataset schema and statistics"""
        print(f"Analyzing dataset: {schema_name}")
        
        tables = self.get_schema_tables(schema_name)
        if not tables:
            print(f"No tables found in schema {schema_name}")
            return None
        
        schema_info = {
            'tables': tables,
            'columns': {},
            'relationships': []
        }
        
        column_stats = {}
        
        for table in tables:
            print(f"  Analyzing table: {table}")
            columns = self.get_table_columns(schema_name, table)
            schema_info['columns'][table] = columns
            column_stats[table] = {}
            
            for col_name, data_type, is_nullable in columns:
                stats = self.get_column_statistics(schema_name, table, col_name, data_type)
                if stats:
                    column_stats[table][col_name] = stats
        
        # Try to infer relationships based on column names and types
        # Simple heuristic: columns named 'id' are primary keys, columns ending with '_id' are foreign keys
        for table in tables:
            for col_name, data_type, is_nullable in schema_info['columns'][table]:
                if col_name.endswith('_id') and col_name != 'id':
                    # Look for potential referenced table
                    ref_table = col_name[:-3]  # Remove '_id' suffix
                    if ref_table in tables:
                        # Check if referenced table has 'id' column
                        ref_cols = [c[0] for c in schema_info['columns'][ref_table]]
                        if 'id' in ref_cols:
                            schema_info['relationships'].append((table, [col_name], ref_table, ['id']))
        
        return schema_info, column_stats


class APQueryGenerator:
    """Generate Analytical Processing (OLAP) queries - adapted from generate_zsce_queries.py"""
    
    def __init__(self, schema_info, column_stats):
        self.schema_info = schema_info
        self.column_stats = column_stats
        self.relationships_table = collections.defaultdict(list)
        
        # Build relationship index
        for table_l, column_l, table_r, column_r in schema_info['relationships']:
            if not isinstance(column_l, list):
                column_l = [column_l]
            if not isinstance(column_r, list):
                column_r = [column_r]
            self.relationships_table[table_l].append([column_l, table_r, column_r])
            self.relationships_table[table_r].append([column_r, table_l, column_l])
    
    def sample_literal_from_percentiles(self, percentiles, randstate, round_val=False):
        """Sample a literal from column percentiles"""
        if not percentiles or len(percentiles) < 2:
            return None
        start_idx = randstate.randint(0, len(percentiles) - 1)
        try:
            literal = randstate.uniform(percentiles[start_idx], percentiles[start_idx + 1] if start_idx + 1 < len(percentiles) else percentiles[start_idx])
            if round_val:
                literal = int(literal)
            return literal
        except:
            return None
    
    def sample_predicate(self, table, col_name, stats, randstate):
        """Sample a single predicate"""
        if not stats:
            return None
            
        if stats['datatype'] == 'INT':
            reasonable_ops = [Operator.LEQ, Operator.GEQ]
            if stats['num_unique'] < 100:  # int_neq_predicate_threshold
                reasonable_ops += [Operator.EQ, Operator.NEQ]
            literal = self.sample_literal_from_percentiles(stats['percentiles'], randstate, round_val=True)
        elif stats['datatype'] == 'FLOAT':
            reasonable_ops = [Operator.LEQ, Operator.GEQ]
            literal = self.sample_literal_from_percentiles(stats['percentiles'], randstate, round_val=False)
        elif stats['datatype'] == 'CATEGORICAL':
            reasonable_ops = [Operator.EQ, Operator.NEQ]
            if 'unique_vals' in stats and stats['unique_vals']:
                literal = rand_choice(randstate, stats['unique_vals'])
                literal = f"'{literal}'"
            else:
                return None
        else:
            return None
            
        if literal is None:
            return None
            
        operator = rand_choice(randstate, reasonable_ops)
        return ColumnPredicate(table, col_name, operator, literal)
    
    def sample_aggregations(self, join_tables, randstate, max_no_aggregates=3, max_cols_per_agg=2):
        """Sample aggregation functions"""
        # Find numerical columns for aggregation
        numerical_columns = []
        for table in join_tables:
            if table in self.column_stats:
                for col_name, stats in self.column_stats[table].items():
                    if stats['datatype'] in ['INT', 'FLOAT']:
                        numerical_columns.append([table, col_name])
        
        aggregations = []
        no_aggregates = randstate.randint(1, max_no_aggregates + 1)
        
        for i in range(no_aggregates):
            agg = rand_choice(randstate, list(Aggregator))
            if agg == Aggregator.COUNT:
                aggregations.append((agg, []))
            elif numerical_columns:
                no_agg_cols = min(randstate.randint(1, max_cols_per_agg + 1), len(numerical_columns))
                cols = rand_choice(randstate, numerical_columns, no_elements=no_agg_cols, replace=False)
                aggregations.append((agg, cols))
        
        if not aggregations:
            aggregations.append((Aggregator.COUNT, []))
            
        return aggregations
    
    def sample_group_bys(self, join_tables, randstate, max_no_group_by=3):
        """Sample group by columns"""
        # Find categorical/low-cardinality columns for group by
        group_by_columns = []
        for table in join_tables:
            if table in self.column_stats:
                for col_name, stats in self.column_stats[table].items():
                    if stats['datatype'] in ['INT', 'CATEGORICAL'] and stats['num_unique'] < 10000:
                        group_by_columns.append([table, col_name, stats['num_unique']])
        
        no_group_bys = randstate.randint(0, max_no_group_by + 1)
        if no_group_bys == 0 or not group_by_columns:
            return []
        
        no_group_bys = min(no_group_bys, len(group_by_columns))
        return rand_choice(randstate, group_by_columns, no_elements=no_group_bys, replace=False)
    
    def sample_predicates(self, join_tables, randstate, max_no_predicates=5):
        """Sample WHERE predicates"""
        possible_columns = []
        for table in join_tables:
            if table in self.column_stats:
                for col_name, stats in self.column_stats[table].items():
                    if stats['datatype'] in ['INT', 'FLOAT', 'CATEGORICAL']:
                        possible_columns.append((table, col_name))
        
        if not possible_columns:
            return PredicateOperator(LogicalOperator.AND, [])
        
        no_predicates = randstate.randint(1, min(max_no_predicates + 1, len(possible_columns)))
        selected_columns = rand_choice(randstate, possible_columns, no_elements=no_predicates, replace=False)
        
        predicates = []
        for table, col_name in selected_columns:
            stats = self.column_stats[table][col_name]
            pred = self.sample_predicate(table, col_name, stats, randstate)
            if pred:
                predicates.append(pred)
        
        return PredicateOperator(LogicalOperator.AND, predicates)
    
    def sample_acyclic_join(self, no_joins, randstate):
        """Sample acyclic joins"""
        joins = []
        start_t = rand_choice(randstate, self.schema_info['tables'])
        join_tables = {start_t}
        
        for _ in range(no_joins):
            possible_joins = []
            for t in join_tables:
                for column_l, table_r, column_r in self.relationships_table[t]:
                    if table_r not in join_tables:
                        possible_joins.append((t, column_l, table_r, column_r))
            
            if not possible_joins:
                break
                
            t, column_l, table_r, column_r = rand_choice(randstate, possible_joins)
            join_tables.add(table_r)
            joins.append((t, column_l, table_r, column_r, False))  # Inner join
        
        return start_t, joins, join_tables
    
    def generate_single_query(self, seed):
        """Generate a single AP query"""
        randstate = np.random.RandomState(seed)
        
        # Sample query structure
        max_no_joins = 5
        max_no_predicates = 5
        max_no_aggregates = 3
        max_no_group_by = 3
        
        no_joins = randstate.randint(0, max_no_joins + 1)
        start_t, joins, join_tables = self.sample_acyclic_join(no_joins, randstate)
        
        # Sample query components
        aggregations = self.sample_aggregations(join_tables, randstate, max_no_aggregates)
        group_bys = self.sample_group_bys(join_tables, randstate, max_no_group_by)
        predicates = self.sample_predicates(join_tables, randstate, max_no_predicates)
        
        # Optional limit
        limit = None
        if randstate.rand() < 0.2:  # 20% chance of LIMIT
            limit = randstate.choice([10, 100, 1000])
        
        # Optional having clause
        having_clause = None
        if group_bys and aggregations and randstate.rand() < 0.2:  # 20% chance of HAVING
            idx = randstate.randint(0, len(aggregations))
            literal = randstate.randint(1, 100)  # Simple literal for HAVING
            op = rand_choice(randstate, [Operator.LEQ, Operator.GEQ])
            having_clause = (idx, literal, op)
        
        q = GenQueryAP(aggregations, group_bys, joins, predicates, start_t, list(join_tables),
                       limit=limit, having_clause=having_clause)
        return q.generate_sql_query()


class TPQueryGenerator:
    """Generate Transactional Processing (OLTP) queries - adapted from generate_TP_workload.py"""
    
    def __init__(self, schema_info, column_stats):
        self.schema_info = schema_info
        self.column_stats = column_stats
        self.relationships_table = collections.defaultdict(list)
        
        # Build relationship index
        for table_l, column_l, table_r, column_r in schema_info['relationships']:
            if not isinstance(column_l, list):
                column_l = [column_l]
            if not isinstance(column_r, list):
                column_r = [column_r]
            self.relationships_table[table_l].append([column_l, table_r, column_r])
            self.relationships_table[table_r].append([column_r, table_l, column_l])
    
    def analyze_columns_for_point_queries(self, join_tables):
        """Find high-cardinality columns good for point queries"""
        table_point_cols = collections.defaultdict(list)
        for table in join_tables:
            if table in self.column_stats:
                for col_name, stats in self.column_stats[table].items():
                    if stats['datatype'] in ['INT', 'CATEGORICAL', 'FLOAT']:
                        # High cardinality columns are good for point queries
                        if stats['num_unique'] > 10:
                            table_point_cols[table].append((col_name, stats))
        return table_point_cols
    
    def sample_literal_from_percentiles(self, percentiles, randstate, round_val=False):
        """Sample a literal from percentiles"""
        if not percentiles or len(percentiles) < 2:
            return None
        try:
            start_idx = randstate.randint(0, len(percentiles) - 1)
            literal = randstate.uniform(percentiles[start_idx], 
                                      percentiles[start_idx + 1] if start_idx + 1 < len(percentiles) else percentiles[start_idx])
            if round_val:
                literal = int(literal)
            return literal
        except:
            return None
    
    def sample_point_predicates(self, table_point_cols, randstate, max_no_predicates=1):
        """Sample equality predicates for point queries"""
        all_tables = list(table_point_cols.keys())
        if not all_tables:
            return PredicateOperator(LogicalOperator.AND, [])
        
        chosen_tables = rand_choice(randstate, all_tables, 
                                   no_elements=min(len(all_tables), max_no_predicates), replace=False)
        
        predicates = []
        for table in chosen_tables:
            possible_cols = table_point_cols[table]
            if not possible_cols:
                continue
                
            col_name, col_stats = rand_choice(randstate, possible_cols)
            
            # Sample literal
            literal_val = None
            if col_stats['datatype'] == 'INT':
                literal_val = self.sample_literal_from_percentiles(col_stats['percentiles'], randstate, round_val=True)
            elif col_stats['datatype'] == 'FLOAT':
                literal_val = self.sample_literal_from_percentiles(col_stats['percentiles'], randstate, round_val=False)
            elif col_stats['datatype'] == 'CATEGORICAL':
                if 'unique_vals' in col_stats and col_stats['unique_vals']:
                    literal_val = rand_choice(randstate, col_stats['unique_vals'])
                
            if literal_val is None:
                continue
                
            if isinstance(literal_val, str):
                literal_str = f"'{literal_val}'"
            else:
                literal_str = f"{literal_val}"
            
            # Build equality condition
            p = ColumnPredicate(table, col_name, Operator.EQ, literal_str)
            predicates.append(p)
        
        return PredicateOperator(LogicalOperator.AND, predicates)
    
    def build_select_list(self, join_tables, randstate, max_cols_per_table=2):
        """Build SELECT clause for TP query"""
        select_list = []
        for table in join_tables:
            if table in self.column_stats:
                all_cols = list(self.column_stats[table].keys())
                no_cols = min(len(all_cols), max_cols_per_table)
                chosen = rand_choice(randstate, all_cols, no_elements=no_cols, replace=False)
                for col in chosen:
                    select_list.append((table, col))
        return select_list
    
    def sample_acyclic_join(self, no_joins, randstate):
        """Sample joins for TP queries"""
        joins = []
        start_t = rand_choice(randstate, self.schema_info['tables'])
        join_tables = {start_t}
        
        for _ in range(no_joins):
            possible_joins = []
            for t in join_tables:
                for column_l, table_r, column_r in self.relationships_table[t]:
                    if table_r not in join_tables:
                        possible_joins.append((t, column_l, table_r, column_r))
            
            if not possible_joins:
                break
                
            t, column_l, table_r, column_r = rand_choice(randstate, possible_joins)
            join_tables.add(table_r)
            joins.append((t, column_l, table_r, column_r, False))
        
        return start_t, joins, join_tables
    
    def generate_single_query(self, seed):
        """Generate a single TP query"""
        randstate = np.random.RandomState(seed)
        
        # Sample query structure (fewer joins for TP)
        max_no_joins = 3
        max_no_predicates = 5
        
        no_joins = randstate.randint(0, max_no_joins + 1)
        start_table, joins, joined_tables = self.sample_acyclic_join(no_joins, randstate)
        
        # Build select columns
        select_cols = self.build_select_list(joined_tables, randstate, max_cols_per_table=3)
        
        # Build point predicates
        table_point_cols = self.analyze_columns_for_point_queries(joined_tables)
        predicates = self.sample_point_predicates(table_point_cols, randstate, max_no_predicates=max_no_predicates)
        
        # Optional limit (more common in TP)
        limit = None
        if randstate.rand() < 0.7:  # 70% chance of LIMIT for TP
            limit = randstate.choice([1, 10, 100])
        
        q = GenQueryTP(select_cols, joins, predicates, start_table, joined_tables, limit=limit)
        return q.generate_sql_query()


def generate_queries_for_dataset(schema_name, target_dir, num_ap_queries=10000, num_tp_queries=10000):
    """Generate AP and TP queries for a single dataset"""
    print(f"\n=== Generating queries for dataset: {schema_name} ===")
    
    # Analyze dataset
    introspector = DatabaseIntrospector()
    if not introspector.connect_duckdb():
        print(f"Failed to connect to DuckDB for {schema_name}")
        return False
    
    schema_info, column_stats = introspector.analyze_dataset(schema_name)
    if not schema_info:
        print(f"Failed to analyze schema {schema_name}")
        return False
    
    print(f"Found {len(schema_info['tables'])} tables, {len(schema_info['relationships'])} relationships")
    
    # Create output directory
    dataset_dir = os.path.join(target_dir, schema_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate AP queries
    print(f"Generating {num_ap_queries:,} AP queries...")
    ap_generator = APQueryGenerator(schema_info, column_stats)
    
    ap_queries = []
    seeds = [i for i in range(num_ap_queries)]
    for seed in tqdm(seeds, desc="Generating AP queries"):
        try:
            query = ap_generator.generate_single_query(seed)
            if query:
                ap_queries.append(query)
        except Exception as e:
            print(f"Error generating AP query {seed}: {e}")
    
    # Write AP queries
    ap_path = os.path.join(dataset_dir, 'workload_10k_ap_queries.sql')
    with open(ap_path, 'w') as f:
        f.write('\n'.join(ap_queries))
    print(f"Generated {len(ap_queries):,} AP queries -> {ap_path}")
    
    # Generate TP queries
    print(f"Generating {num_tp_queries:,} TP queries...")
    tp_generator = TPQueryGenerator(schema_info, column_stats)
    
    tp_queries = []
    seeds = [i + num_ap_queries for i in range(num_tp_queries)]
    for seed in tqdm(seeds, desc="Generating TP queries"):
        try:
            query = tp_generator.generate_single_query(seed)
            if query:
                tp_queries.append(query)
        except Exception as e:
            print(f"Error generating TP query {seed}: {e}")
    
    # Write TP queries
    tp_path = os.path.join(dataset_dir, 'workload_10k_tp_queries.sql')
    with open(tp_path, 'w') as f:
        f.write('\n'.join(tp_queries))
    print(f"Generated {len(tp_queries):,} TP queries -> {tp_path}")
    
    # Save metadata
    metadata = {
        'dataset': schema_name,
        'ap_queries': len(ap_queries),
        'tp_queries': len(tp_queries),
        'total_queries': len(ap_queries) + len(tp_queries),
        'tables': len(schema_info['tables']),
        'relationships': len(schema_info['relationships']),
        'ap_file': ap_path,
        'tp_file': tp_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(dataset_dir, 'query_generation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Total queries generated for {schema_name}: {len(ap_queries) + len(tp_queries):,}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark queries for PostgreSQL/DuckDB query routing')
    parser.add_argument('--output_dir', type=str, default='/data/wuy/db/benchmark_queries',
                       help='Directory to store generated query workloads')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to generate queries for (default: all available)')
    parser.add_argument('--num_ap_queries', type=int, default=10000,
                       help='Number of AP (analytical) queries per dataset')
    parser.add_argument('--num_tp_queries', type=int, default=10000,
                       help='Number of TP (transactional) queries per dataset')
    
    args = parser.parse_args()
    
    print("=== Benchmark Query Generator for PostgreSQL/DuckDB ===")
    print("Based on TiDB AQD (Automatic Query Diversification) methodology")
    
    # Get available datasets
    introspector = DatabaseIntrospector()
    available_datasets = introspector.get_schemas()
    
    if not available_datasets:
        print("No datasets found in DuckDB")
        return 1
    
    print(f"Available datasets: {available_datasets}")
    
    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = [d for d in args.datasets if d in available_datasets]
        if not datasets_to_process:
            print("No valid datasets specified")
            return 1
    else:
        datasets_to_process = available_datasets
    
    print(f"Processing {len(datasets_to_process)} datasets: {datasets_to_process}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate queries for each dataset
    start_time = time.time()
    successful_datasets = 0
    total_queries = 0
    
    for dataset in datasets_to_process:
        if generate_queries_for_dataset(dataset, args.output_dir, args.num_ap_queries, args.num_tp_queries):
            successful_datasets += 1
            total_queries += args.num_ap_queries + args.num_tp_queries
    
    # Generate summary report
    end_time = time.time()
    duration = end_time - start_time
    
    summary = {
        'total_datasets_processed': successful_datasets,
        'total_queries_generated': total_queries,
        'ap_queries_per_dataset': args.num_ap_queries,
        'tp_queries_per_dataset': args.num_tp_queries,
        'generation_duration_seconds': duration,
        'queries_per_second': total_queries / duration if duration > 0 else 0,
        'datasets': datasets_to_process,
        'output_directory': args.output_dir,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'methodology': 'TiDB AQD (Automatic Query Diversification) adapted for PostgreSQL/DuckDB'
    }
    
    summary_path = os.path.join(args.output_dir, 'generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Generation Complete ===")
    print(f"Datasets processed: {successful_datasets}")
    print(f"Total queries generated: {total_queries:,}")
    print(f"Duration: {duration:.1f} seconds ({total_queries / duration:.1f} queries/sec)")
    print(f"Summary saved to: {summary_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())