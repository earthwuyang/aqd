import collections
import os
from enum import Enum

import numpy as np

from column_types import Datatype
from utils import load_schema_json, load_column_statistics, load_string_statistics, load_json


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


class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'

    def __str__(self):
        return self.value


class ColumnPredicate:
    """
    Represents a single predicate like:
    table.col operator literal
    e.g., "myTable.id = 12345"
    """

    def __init__(self, table, col_name, operator, literal):
        self.table = table
        self.col_name = col_name
        self.operator = operator
        self.literal = literal

    def __str__(self):
        return self.to_sql(top_operator=True)

    def to_sql(self, top_operator=False):
        """
        Convert a single predicate to SQL. top_operator=True
        means we might prefix with 'WHERE' if this predicate
        stands alone. For combination, we often omit that.
        """
        if self.operator == Operator.IS_NOT_NULL:
            predicates_str = f'"{self.table}"."{self.col_name}" IS NOT NULL'
        elif self.operator == Operator.IS_NULL:
            predicates_str = f'"{self.table}"."{self.col_name}" IS NULL'
        else:
            predicates_str = f'"{self.table}"."{self.col_name}" {str(self.operator)} {self.literal}'

        if top_operator:
            predicates_str = f' WHERE {predicates_str}'

        return predicates_str


class PredicateOperator:
    """
    Represents a logical AND/OR combination of child predicates.
    For point queries here, we typically just do AND among them.
    """

    def __init__(self, logical_op, children=None):
        self.logical_op = logical_op
        if children is None:
            children = []
        self.children = children

    def __str__(self):
        return self.to_sql(top_operator=True)

    def to_sql(self, top_operator=False):
        if len(self.children) == 0:
            return ""
        # Combine child predicates with the logical operator
        predicates_str_list = [c.to_sql() for c in self.children]
        sql = f' {str(self.logical_op)} '.join(predicates_str_list)
        # Possibly wrap with WHERE if top_operator
        if top_operator and sql.strip():
            sql = f' WHERE {sql}'
        elif len(self.children) > 1:
            sql = f'({sql})'
        return sql


def rand_choice(randstate, l, no_elements=None, replace=False):
    """
    Helper function to sample from a list with a given random state.
    """
    if no_elements is None:
        idx = randstate.randint(0, len(l))
        return l[idx]
    else:
        idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
        return [l[i] for i in idxs]


def sample_literal_from_percentiles(percentiles, randstate, round_val=False):
    """
    Choose a random numeric value from a table's distribution percentiles.
    E.g. if percentiles = [1, 10, 100], we pick a uniform value
    between two consecutive percentiles.
    """
    if len(percentiles) < 2 or np.all(np.isnan(percentiles)):
        return None
    start_idx = randstate.randint(0, len(percentiles) - 1)
    low = percentiles[start_idx]
    high = percentiles[start_idx + 1]

    if np.isnan(low) or np.isnan(high):
        return None

    literal_val = randstate.uniform(low, high)
    if round_val:
        literal_val = int(literal_val)
    return literal_val


class GenQuery:
    """
    Represents a straightforward SELECT ... FROM ... [JOIN ...] WHERE ...
    for point queries. We'll omit group by, aggregation, etc.
    """

    def __init__(self, select_columns, joins, predicates, start_table, join_tables, alias_dict=None, limit=None):
        """
        select_columns: List of (table, col_name) to select
        joins: List of (table_left, col_left, table_right, col_right, left_outer)
        predicates: a PredicateOperator or single ColumnPredicate
        start_table: the initial table in the FROM clause
        join_tables: set of all tables used in the join
        alias_dict: optional dictionary of table -> alias
        limit: optional integer limit
        """
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
        """
        Construct the final SQL. 
        E.g. SELECT "t1"."colA", "t2"."colB" FROM "t1" JOIN "t2" ON ...
        WHERE "t1"."id" = 123
        """
        # Build SELECT list
        if len(self.select_columns) == 0:
            # if no columns were chosen, fallback to "*"
            select_str = '*'
        else:
            selected = [f'"{tbl}"."{col}"' for (tbl, col) in self.select_columns]
            select_str = ', '.join(selected)

        # Build the FROM + JOIN part
        join_str = self._build_from_join_str()

        # Build predicates
        predicate_str = str(self.predicates)

        # Build limit
        limit_str = f" LIMIT {self.limit}" if self.limit is not None else ""

        sql_query = f"SELECT {select_str} FROM {join_str}{predicate_str}{limit_str}"
        if semicolon:
            sql_query += ";"
        return sql_query

    def _build_from_join_str(self):
        """
        Build the FROM and JOIN clauses, applying any aliases if needed.
        """
        already_repl = set()

        def repl_alias(t, no_alias_intro=False):
            if t in self.alias_dict:
                alias_t = self.alias_dict[t]
                # If we've already introduced the alias, just return alias
                if t in already_repl or no_alias_intro:
                    return alias_t
                else:
                    already_repl.add(t)
                    return f'"{t}" {alias_t}'
            return f'"{t}"'

        # Start with the base table
        join_clause = repl_alias(self.start_table)
        # Add each join
        for (table_l, column_l, table_r, column_r, left_outer) in self.joins:
            join_kw = "LEFT OUTER JOIN" if left_outer else "JOIN"
            join_clause += f" {join_kw} {repl_alias(table_r)} ON "
            join_cond = []
            for col_l, col_r in zip(column_l, column_r):
                join_cond.append(f'{repl_alias(table_l, no_alias_intro=True)}."{col_l}" = '
                                 f'{repl_alias(table_r, no_alias_intro=True)}."{col_r}"')
            join_clause += ' AND '.join(join_cond)
        return join_clause


def sample_acyclic_join(no_joins, relationships_table, schema, randstate, left_outer_join_ratio=0.0):
    """
    Randomly pick an initial table, then add up to no_joins join steps
    to form a chain or star of joins. Return:
     (start_table, list_of_joins, set_of_joined_tables)
    """
    joins = []
    start_t = rand_choice(randstate, schema.tables)
    join_tables = {start_t}

    for _ in range(no_joins):
        possible_joins = find_possible_joins(join_tables, relationships_table)
        if len(possible_joins) == 0:
            break
        t, col_l, table_r, col_r = rand_choice(randstate, possible_joins)
        join_tables.add(table_r)

        left_outer_join = False
        if left_outer_join_ratio > 0 and randstate.rand() < left_outer_join_ratio:
            left_outer_join = True

        joins.append((t, col_l, table_r, col_r, left_outer_join))

    return start_t, joins, join_tables


def find_possible_joins(join_tables, relationships_table):
    """
    Look up all possible ways to join any table in `join_tables`
    with some new table that is not yet in the set.
    relationships_table[t] is a list of [column_left, table_right, column_right]
    """
    possible_joins = []
    for t in join_tables:
        for (column_l, table_r, column_r) in relationships_table[t]:
            if table_r not in join_tables:
                possible_joins.append((t, column_l, table_r, column_r))
    return possible_joins


def analyze_columns_for_point_queries(column_stats, join_tables):
    """
    Among the joined tables, figure out which columns are:
      - candidates for point predicates (ideally high-cardinality or PK-like).
    Return dictionary: table -> list of candidate columns
    """
    # A simple heuristic: pick columns that have high num_unique or are INT
    # to represent IDs or keys. You can refine as needed.
    table_point_cols = collections.defaultdict(list)
    for t in join_tables:
        # We assume each table has stats in column_stats[t]
        for col_name, stats in vars(vars(column_stats)[t]).items():
            if stats.datatype in {str(Datatype.INT), str(Datatype.CATEGORICAL), str(Datatype.FLOAT)}:
                # Heuristic: if it's at least moderately unique, it's good for point condition
                if stats.num_unique > 10:  # threshold can be adjusted
                    table_point_cols[t].append((col_name, stats))
    return table_point_cols


def sample_point_predicates(table_point_cols, randstate, max_no_predicates=1):
    """
    For transactional queries, we typically want 1 or a few equality conditions
    on columns with high cardinality (like IDs).

    We limit the total number of predicates to max_no_predicates
    (spread across the available tables).
    """
    all_tables = list(table_point_cols.keys())
    if not all_tables:
        return PredicateOperator(LogicalOperator.AND, [])

    # We can pick some subset of tables to produce predicates on
    # E.g. if max_no_predicates = 1, we only pick one table's column to filter.
    chosen_tables = rand_choice(randstate, all_tables, no_elements=min(len(all_tables), max_no_predicates),
                                replace=False)

    predicates = []
    for t in chosen_tables:
        possible_cols = table_point_cols[t]
        if not possible_cols:
            continue

        # Pick one random high-card column
        col_name, col_stats = rand_choice(randstate, possible_cols)
        # Sample a literal from that column's distribution
        literal_val = None
        if col_stats.datatype == str(Datatype.INT):
            literal_val = sample_literal_from_percentiles(col_stats.percentiles, randstate, round_val=True)
        elif col_stats.datatype == str(Datatype.FLOAT):
            literal_val = sample_literal_from_percentiles(col_stats.percentiles, randstate, round_val=False)
        else:
            # CATEGORICAL or anything else: pick from unique_vals if available
            if col_stats.unique_vals:
                literal_val = rand_choice(randstate, col_stats.unique_vals)
            else:
                # fallback numeric
                literal_val = sample_literal_from_percentiles(col_stats.percentiles, randstate, round_val=True)

        if literal_val is None:
            # skip if we couldn't sample
            continue
        if isinstance(literal_val, str):
            # quote strings
            literal_str = f"'{literal_val}'"
        else:
            literal_str = f"{literal_val}"

        # Build the point/equality condition
        p = ColumnPredicate(t, col_name, Operator.EQ, literal_str)
        predicates.append(p)

    if len(predicates) == 0:
        return PredicateOperator(LogicalOperator.AND, [])

    # Combine them with AND
    return PredicateOperator(LogicalOperator.AND, predicates)


def build_select_list(join_tables, column_stats, randstate, max_cols_per_table=2):
    """
    For a transactional query, we might select a few columns from each table (or just one).
    This function picks up to max_cols_per_table columns from each table in the join.
    """
    select_list = []
    for t in join_tables:
        # gather columns from the stats
        all_cols = list(vars(vars(column_stats)[t]).keys())
        # pick a random subset
        no_cols = min(len(all_cols), max_cols_per_table)
        chosen = rand_choice(randstate, all_cols, no_elements=no_cols, replace=False)
        for c in chosen:
            select_list.append((t, c))
    return select_list


def generate_workload(dataset,
                      target_path,
                      num_queries=100,
                      max_no_joins=3,
                      max_no_predicates=1,
                      max_cols_per_table=2,
                      seed=0,
                      force=False,
                      left_outer_join_ratio=0.0,
                      limit_per_query=None):
    """
    Generate a set of transactional/point queries suitable for row-based execution.
    - dataset: path or identifier for dataset
    - target_path: file path where the generated SQL workload will be written
    - num_queries: how many queries to generate
    - max_no_joins: maximum number of joins per query
    - max_no_predicates: maximum number of equality filters across all tables
    - max_cols_per_table: how many columns to select from each table
    - seed: random seed
    - force: if True, overwrite existing workload
    - left_outer_join_ratio: fraction of joins that become LEFT OUTER instead of INNER
    - limit_per_query: if provided, puts a LIMIT n at the end of each query
    """
    randstate = np.random.RandomState(seed)

    if os.path.exists(target_path) and not force:
        print("Workload already generated (use force=True to overwrite).")
        return

    # 1. Load statistics and schema
    # read the schema file
    # column_stats = load_column_statistics(dataset)
    column_stats = load_json(f"../datasets/{dataset}/column_statistics.json")
    # string_stats = load_string_statistics(dataset)
    string_stats = load_json(f"../datasets/{dataset}/string_statistics.json")
    # schema = load_schema_json(dataset)
    schema_file = f'../datasets/{dataset}/schema.json'
    schema = load_json(schema_file)

    # 2. Build relationship index
    relationships_table = collections.defaultdict(list)
    for table_l, column_l, table_r, column_r in schema.relationships:
        if not isinstance(column_l, list):
            column_l = [column_l]
        if not isinstance(column_r, list):
            column_r = [column_r]
        relationships_table[table_l].append([column_l, table_r, column_r])
        relationships_table[table_r].append([column_r, table_l, column_l])

    queries = []

    # 3. Generate queries
    from tqdm import tqdm
    for _ in tqdm(range(num_queries)):
        # (a) Decide how many joins
        no_joins = randstate.randint(0, max_no_joins + 1)
        start_table, joins, joined_tables = sample_acyclic_join(
            no_joins, relationships_table, schema, randstate, left_outer_join_ratio
        )

        # (b) Pick select columns from each table
        select_cols = build_select_list(joined_tables, column_stats, randstate, max_cols_per_table)

        # (c) Pick up to max_no_predicates equality filters across these tables
        table_point_cols = analyze_columns_for_point_queries(column_stats, joined_tables)
        predicates = sample_point_predicates(table_point_cols, randstate, max_no_predicates=max_no_predicates)

        # (d) Build the final GenQuery
        q = GenQuery(select_cols, joins, predicates, start_table, joined_tables, limit=limit_per_query)

        sql_query = q.generate_sql_query()
        queries.append(sql_query)

    # 4. Write out the queries to file
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write('\n'.join(queries))


# -------------------------------------------------------
# If you want to test quickly (pseudo-code):
#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'tpch_sf1')
    parser.add_argument('--data_dir', type=str, default='/home/wuy/query_costs')
    # parser.add_argument("--target_path", type=str, default= '/home/wuy/query_costs/workloads/tpcds_sf1/TP_queries.sql')
    parser.add_argument("--num_queries", type=int, default=100000)
    args = parser.parse_args()

    target_path = os.path.join(args.data_dir, 'workloads', args.dataset, 'TP_queries.sql')
    print(f"Generating workload for {args.dataset} to {target_path}")

    generate_workload(
        dataset=args.dataset,
        target_path=target_path,
        num_queries=args.num_queries,
        max_no_joins=0,
        max_no_predicates=5,
        max_cols_per_table=3,
        seed=42,
        force=True
    )
# -------------------------------------------------------
