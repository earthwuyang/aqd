#!/usr/bin/env python3
"""
ZSCE (Zero-Shot Cross-Engine) Query Generator for PostgreSQL + DuckDB AQD
Adapted from ~/DB/duckdb/generate_benchmark_queries.py
Based on TiDB AQD (Automatic Query Diversification) methodology
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
import logging

# Database connection configuration
# Database connection configuration
POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'postgres',
    'user': 'wuy',
    'password': ''
}

DUCKDB_CONFIG = {
    'database': '/home/wuy/DB/pg_duckdb_postgres/benchmark_datasets.db'
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/zsce_generation.log'),
        logging.StreamHandler()
    ]
)

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
    """Extract schema and statistics from PostgreSQL for our specific schema"""
    
    def __init__(self):
        self.pg_conn = None
        self.schema_info = {
            'tables': ['users', 'orders', 'products'],
            'columns': {
                'users': [
                    ('user_id', 'integer', 'NO'),
                    ('name', 'character varying', 'YES'),
                    ('email', 'character varying', 'YES'),
                    ('created_at', 'timestamp without time zone', 'YES'),
                    ('status', 'character varying', 'YES'),
                    ('age', 'integer', 'YES'),
                    ('city', 'character varying', 'YES')
                ],
                'orders': [
                    ('order_id', 'integer', 'NO'),
                    ('user_id', 'integer', 'YES'),
                    ('product_id', 'integer', 'YES'),
                    ('quantity', 'integer', 'YES'),
                    ('price', 'numeric', 'YES'),
                    ('order_date', 'timestamp without time zone', 'YES'),
                    ('status', 'character varying', 'YES'),
                    ('total_amount', 'numeric', 'YES')
                ],
                'products': [
                    ('product_id', 'integer', 'NO'),
                    ('name', 'character varying', 'YES'),
                    ('category', 'character varying', 'YES'),
                    ('price', 'numeric', 'YES'),
                    ('stock', 'integer', 'YES'),
                    ('created_at', 'timestamp without time zone', 'YES'),
                    ('description', 'text', 'YES')
                ]
            },
            'relationships': [
                ('orders', ['user_id'], 'users', ['user_id']),
                ('orders', ['product_id'], 'products', ['product_id'])
            ]
        }
        
    def connect_postgresql(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
            return True
        except Exception as e:
            logging.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def get_column_statistics(self, table_name, column_name, data_type):
        """Get statistics for a column using predefined values"""
        if not self.pg_conn:
            return None
            
        try:
            cursor = self.pg_conn.cursor()
            
            if 'int' in data_type.lower() or 'numeric' in data_type.lower():
                # Get numeric stats
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT({column_name}) as non_null_count,
                        COUNT(DISTINCT {column_name}) as unique_count,
                        MIN({column_name}) as min_val,
                        MAX({column_name}) as max_val,
                        AVG({column_name}) as mean_val
                    FROM {table_name}
                    WHERE {column_name} IS NOT NULL
                """)
                stats = cursor.fetchone()
                
                if stats and stats[0] > 0:
                    # Get percentiles
                    cursor.execute(f"""
                        SELECT 
                            percentile_cont(0.05) WITHIN GROUP (ORDER BY {column_name}) as p5,
                            percentile_cont(0.25) WITHIN GROUP (ORDER BY {column_name}) as p25,
                            percentile_cont(0.5) WITHIN GROUP (ORDER BY {column_name}) as p50,
                            percentile_cont(0.75) WITHIN GROUP (ORDER BY {column_name}) as p75,
                            percentile_cont(0.95) WITHIN GROUP (ORDER BY {column_name}) as p95
                        FROM {table_name}
                        WHERE {column_name} IS NOT NULL
                    """)
                    percentiles = cursor.fetchone()
                    
                    return {
                        'datatype': 'INT' if 'int' in data_type.lower() else 'FLOAT',
                        'num_unique': stats[2] or 0,
                        'nan_ratio': 1.0 - (stats[1] / stats[0]) if stats[0] > 0 else 1.0,
                        'mean': float(stats[5]) if stats[5] else 0.0,
                        'min': float(stats[3]) if stats[3] else 0.0,
                        'max': float(stats[4]) if stats[4] else 0.0,
                        'percentiles': [float(p) for p in percentiles] if percentiles else [0, 0, 0, 0, 0]
                    }
            else:
                # String/categorical columns
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT({column_name}) as non_null_count,
                        COUNT(DISTINCT {column_name}) as unique_count
                    FROM {table_name}
                """)
                stats = cursor.fetchone()
                
                # Get sample values
                cursor.execute(f"""
                    SELECT DISTINCT {column_name}
                    FROM {table_name}
                    WHERE {column_name} IS NOT NULL
                    LIMIT 20
                """)
                unique_vals = cursor.fetchall()
                
                return {
                    'datatype': 'CATEGORICAL',
                    'num_unique': stats[2] if stats[2] else 0,
                    'nan_ratio': 1.0 - (stats[1] / stats[0]) if stats[0] > 0 else 1.0,
                    'unique_vals': [str(v[0]) for v in unique_vals]
                }
                
        except Exception as e:
            logging.warning(f"Error getting stats for {table_name}.{column_name}: {e}")
            # Return default stats based on data type
            if 'int' in data_type.lower():
                return {'datatype': 'INT', 'num_unique': 1000, 'nan_ratio': 0.0, 'percentiles': [1, 100, 500, 1000, 5000]}
            elif 'numeric' in data_type.lower():
                return {'datatype': 'FLOAT', 'num_unique': 1000, 'nan_ratio': 0.0, 'percentiles': [1.0, 10.0, 50.0, 100.0, 500.0]}
            else:
                return {'datatype': 'CATEGORICAL', 'num_unique': 10, 'nan_ratio': 0.1, 'unique_vals': ['active', 'inactive', 'pending']}
    
    def analyze_dataset(self):
        """Analyze the complete dataset"""
        logging.info("Analyzing dataset schema")
        
        if not self.connect_postgresql():
            return None, None
            
        column_stats = {}
        
        for table_name in self.schema_info['tables']:
            logging.info(f"Analyzing table: {table_name}")
            column_stats[table_name] = {}
            
            for col_name, data_type, is_nullable in self.schema_info['columns'][table_name]:
                stats = self.get_column_statistics(table_name, col_name, data_type)
                if stats:
                    column_stats[table_name][col_name] = stats
        
        return self.schema_info, column_stats

class APQueryGenerator:
    """Generate Analytical Processing (OLAP) queries"""
    
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
            if stats['num_unique'] < 100:
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
        
        max_no_joins = 3  # Limit joins for our 3-table schema
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
        if randstate.rand() < 0.2:
            limit = randstate.choice([10, 100, 1000])
        
        # Optional having clause
        having_clause = None
        if group_bys and aggregations and randstate.rand() < 0.2:
            idx = randstate.randint(0, len(aggregations))
            literal = randstate.randint(1, 100)
            op = rand_choice(randstate, [Operator.LEQ, Operator.GEQ])
            having_clause = (idx, literal, op)
        
        q = GenQueryAP(aggregations, group_bys, joins, predicates, start_t, list(join_tables),
                       limit=limit, having_clause=having_clause)
        return q.generate_sql_query()

class TPQueryGenerator:
    """Generate Transactional Processing (OLTP) queries"""
    
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
        
        max_no_joins = 2  # Fewer joins for TP
        max_no_predicates = 3
        
        no_joins = randstate.randint(0, max_no_joins + 1)
        start_table, joins, joined_tables = self.sample_acyclic_join(no_joins, randstate)
        
        # Build select columns
        select_cols = self.build_select_list(joined_tables, randstate, max_cols_per_table=3)
        
        # Build point predicates
        table_point_cols = self.analyze_columns_for_point_queries(joined_tables)
        predicates = self.sample_point_predicates(table_point_cols, randstate, max_no_predicates=max_no_predicates)
        
        # Optional limit (more common in TP)
        limit = None
        if randstate.rand() < 0.7:
            limit = randstate.choice([1, 10, 100])
        
        q = GenQueryTP(select_cols, joins, predicates, start_table, joined_tables, limit=limit)
        return q.generate_sql_query()

def generate_queries_for_dataset(dataset_name, target_dir, num_ap_queries=10000, num_tp_queries=10000):
    """Generate AP and TP queries for our PostgreSQL + DuckDB setup"""
    logging.info(f"\n=== Generating ZSCE queries for dataset: {dataset_name} ===")
    
    # Analyze dataset
    introspector = DatabaseIntrospector()
    schema_info, column_stats = introspector.analyze_dataset()
    if not schema_info:
        logging.error(f"Failed to analyze schema")
        return False
    
    logging.info(f"Found {len(schema_info['tables'])} tables, {len(schema_info['relationships'])} relationships")
    
    # Create output directory
    dataset_dir = os.path.join(target_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate AP queries
    logging.info(f"Generating {num_ap_queries:,} AP queries...")
    ap_generator = APQueryGenerator(schema_info, column_stats)
    
    ap_queries = []
    for seed in tqdm(range(num_ap_queries), desc="Generating AP queries"):
        try:
            query = ap_generator.generate_single_query(seed)
            if query:
                ap_queries.append(query)
        except Exception as e:
            logging.warning(f"Error generating AP query {seed}: {e}")
    
    # Write AP queries
    ap_path = os.path.join(dataset_dir, 'workload_10k_ap_queries.sql')
    with open(ap_path, 'w') as f:
        f.write('\n'.join(ap_queries))
    logging.info(f"Generated {len(ap_queries):,} AP queries -> {ap_path}")
    
    # Generate TP queries
    logging.info(f"Generating {num_tp_queries:,} TP queries...")
    tp_generator = TPQueryGenerator(schema_info, column_stats)
    
    tp_queries = []
    for seed in tqdm(range(num_ap_queries, num_ap_queries + num_tp_queries), desc="Generating TP queries"):
        try:
            query = tp_generator.generate_single_query(seed)
            if query:
                tp_queries.append(query)
        except Exception as e:
            logging.warning(f"Error generating TP query {seed}: {e}")
    
    # Write TP queries
    tp_path = os.path.join(dataset_dir, 'workload_10k_tp_queries.sql')
    with open(tp_path, 'w') as f:
        f.write('\n'.join(tp_queries))
    logging.info(f"Generated {len(tp_queries):,} TP queries -> {tp_path}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'methodology': 'ZSCE (Zero-Shot Cross-Engine) adapted from TiDB AQD',
        'ap_queries': len(ap_queries),
        'tp_queries': len(tp_queries),
        'total_queries': len(ap_queries) + len(tp_queries),
        'tables': len(schema_info['tables']),
        'relationships': len(schema_info['relationships']),
        'ap_file': ap_path,
        'tp_file': tp_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(dataset_dir, 'zsce_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Total ZSCE queries generated for {dataset_name}: {len(ap_queries) + len(tp_queries):,}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate ZSCE queries for PostgreSQL + DuckDB AQD')
    parser.add_argument('--output_dir', type=str, default='data/zsce_queries',
                       help='Directory to store generated query workloads')
    parser.add_argument('--num_ap_queries', type=int, default=10000,
                       help='Number of AP (analytical) queries to generate')
    parser.add_argument('--num_tp_queries', type=int, default=10000,
                       help='Number of TP (transactional) queries to generate')
    
    args = parser.parse_args()
    
    logging.info("=== ZSCE Query Generator for PostgreSQL + DuckDB AQD ===")
    logging.info("Based on TiDB AQD (Automatic Query Diversification) methodology")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get available imported datasets from PostgreSQL
    start_time = time.time()
    
    try:
        conn = psycopg2.connect(**POSTGRESQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast', 'pg_temp_1', 'pg_toast_temp_1', 'public');")
        available_datasets = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        logging.info(f"Found {len(available_datasets)} imported datasets: {', '.join(available_datasets)}")
    except Exception as e:
        logging.error(f"Failed to get available datasets: {e}")
        available_datasets = []
    
    if not available_datasets:
        logging.error("No imported datasets found! Please run dataset import first.")
        return 1
    
    # Generate queries for each dataset (use first available for now)
    dataset_name = available_datasets[0]  # Start with first dataset
    success = generate_queries_for_dataset(dataset_name, args.output_dir, args.num_ap_queries, args.num_tp_queries)
    
    end_time = time.time()
    duration = end_time - start_time
    total_queries = args.num_ap_queries + args.num_tp_queries if success else 0
    
    # Generate summary report
    summary = {
        'total_queries_generated': total_queries,
        'ap_queries': args.num_ap_queries,
        'tp_queries': args.num_tp_queries,
        'generation_duration_seconds': duration,
        'queries_per_second': total_queries / duration if duration > 0 else 0,
        'dataset': dataset_name,
        'output_directory': args.output_dir,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'methodology': 'ZSCE (Zero-Shot Cross-Engine) adapted from TiDB AQD',
        'success': success
    }
    
    summary_path = os.path.join(args.output_dir, 'zsce_generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\n=== ZSCE Generation Complete ===")
    logging.info(f"Success: {success}")
    if success:
        logging.info(f"Total queries generated: {total_queries:,}")
        logging.info(f"Duration: {duration:.1f} seconds ({total_queries / duration:.1f} queries/sec)")
    logging.info(f"Summary saved to: {summary_path}")
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())