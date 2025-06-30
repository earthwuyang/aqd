import json
import os

import numpy as np
import pandas as pd

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Datatype):
            return str(obj)
        elif isinstance(obj, pd._libs.missing.NAType):
            return None
        else:
            return super(CustomEncoder, self).default(obj)


def column_stats(column, columntype, categorical_threshold=10000):  
    """
    Default method for encoding the datasets

    Args:
    column is pandas.core.series.Seires
    """
    nan_ratio = sum(column.isna()) / len(column)  # DataFrame.isna returns a boolean same-sized object indicating if the values are NA. such as None or np.NaN. Characters such as empty strings '' or numpy.inf.
    stats = dict(nan_ratio=nan_ratio)
    if column.dtype == object:  # original column def type is string, date, time, etc.

        if len(column.unique()) > categorical_threshold:
            stats.update(dict(datatype=Datatype.MISC))

        else:
            vals_sorted_by_occurence = list(column.value_counts().index)
            stats.update(dict(
                datatype=Datatype.CATEGORICAL,
                unique_vals=vals_sorted_by_occurence,
                num_unique=len(column.unique())
            ))

    else: # integer, float64

        percentiles = list(column.quantile(q=[0.1 * i for i in range(11)]))

        stats.update(dict(
            max=column.max(),
            min=column.min(),
            mean=column.mean(),
            num_unique=len(column.unique()),
            percentiles=percentiles,
        ))

        if columntype == 'char':
            stats.update(dict(datatype=Datatype.STRING_FLOAT))
        else:
            if column.dtype == int:
                stats.update(dict(datatype=Datatype.INT))

            else:
                stats.update(dict(datatype=Datatype.FLOAT))

    return stats


def generate_stats(data_dir, dataset, force=True):
    # read the schema file
    column_stats_path = os.path.join(os.path.dirname(__file__), f'../../cross_db_benchmark/datasets/{dataset}/column_statistics.json')
    if os.path.exists(column_stats_path) and not force:
        print("Column stats already created")
        return

    schema = load_schema_json(dataset)

    column_type_file = os.path.join(os.path.dirname(__file__), f'../datasets/{dataset}/column_type.json')
    if not os.path.exists(column_type_file):
        print(f"column types not extracted, {column_type_file} does not exist. See cross_db_benchmark/datasets/tpc_ds/scripts/script_to_get_column_type.py first.")
        exit()
    with open(column_type_file) as f:
        column_type = json.load(f)

    tables = {}
    for table, columns in column_type.items():
        tables[table] = columns.keys()

    # Define a mapping of the types in column_type to pandas dtypes
    type_mapping = {
        'char': 'str',   # Treat 'char' as string
        'int': 'Int64',  # Use pandas' nullable integer type
        'float': 'float', # Standard float type
        'date': 'str'
    }

    # Generate the dtype argument for read_csv based on column_type
    def get_dtypes_for_table(table, column_type):
        return {col: type_mapping.get(col_type, 'str') for col, col_type in column_type[table].items()}


    # read individual table csvs and derive statistics
    joint_column_stats = dict()
    all_files = os.listdir(data_dir)
    # all_files = [f for f in all_files if f.endswith('.csv')]
    all_files = [f for f in all_files if 'tbl' in f or 'dat' in f or 'csv' in f]
    for t in schema.tables:
        print(f"Generating statistics for {t}")

        column_stats_table = dict()
        # Get the dtype mapping for the current table
        dtype_mapping = get_dtypes_for_table(t, column_type)
        df_table_list = []
        for f in all_files:
            
            # if f.startswith(t):
            # if f.split('.')[0] == t:
                
            import re
            pattern = re.compile(rf"^{re.escape(t)}(_\d+(_\d+)*)?\.(dat|tbl|csv)$", re.I)

            if pattern.match(f):
                print(f"Processing file {f}")
                try:
                    table_dir = os.path.join(data_dir, f)
                    # print(f"schema:{vars(schema.csv_kwargs)}, names={tables[t]}, dtype={dtype_mapping}")
                    # -------- 🔍 ① 先抽样一行，判断真实列数 --------
                    expected_cols = len(tables[t])
                    sep = getattr(schema.csv_kwargs, "sep", "|")  

                    with open(table_dir, encoding=getattr(schema.csv_kwargs, "encoding", "utf-8")) as fh:
                        first_non_blank = next(line for line in fh if line.strip())      # 找到第一行非空行
                    actual_cols = first_non_blank.rstrip("\n").count(sep) + 1           # “|” 数 + 1 = 列数

                    if actual_cols not in (expected_cols, expected_cols + 1):
                        raise ValueError(f"{f}: unexpected column count {actual_cols} "
                                        f"(expected {expected_cols} or {expected_cols+1})")

                    # -------- 🏷️ ② 如果多 1 列，就临时加占位列名 --------
                    names_for_read = list(tables[t]) + ["__dummy__"] if actual_cols == expected_cols + 1 else tables[t]
                    
                    # print(f"reading table: {table_dir}")
                    df_table = pd.read_csv(table_dir, **vars(schema.csv_kwargs), names=names_for_read, dtype=dtype_mapping, header=None)
                    # print(f"df_table: {df_table}")
                    if actual_cols == expected_cols + 1:
                        df_table.drop(columns="__dummy__", inplace=True)

                    df_table_list.append(df_table)
                except Exception as e:
                    print(f"Error processing file {f}: {e}")
                    continue

        # Concatenate all tables into one dataframe
        df_table = pd.concat(df_table_list, ignore_index=True)

        # # Function to check if a column has mixed types
        # def has_mixed_types(series):
        #     # Check if the column has more than one unique dtype (excluding NaNs)
        #     return len(set(series.dropna().map(type))) > 1

        # # Iterate over each column and convert columns with mixed types to string
        # for col in df_table.columns:
        #     if has_mixed_types(df_table[col]):
        #         print(f"!!! Column '{col}' has mixed types. Converting to string.")
        #         # Convert the entire column to string
        #         df_table[col] = df_table[col].astype(str)

        for column in df_table.columns:
            # print(f"column {column}")
            # print(f"df_table:\n {df_table}")
            # print(f"column_type {column_type}")
            column_stats_table[column] = column_stats(df_table[column], columntype = column_type[t][column])

        joint_column_stats[t] = column_stats_table

    # save to json
    with open(column_stats_path, 'w') as outfile:
        # workaround for numpy and other custom datatypes
        json.dump(joint_column_stats, outfile, cls=CustomEncoder)
