import collections
import json
import os

import pandas as pd

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_column_statistics


def generate_string_stats(data_dir, dataset, force=True, max_sample_vals=100000, min_str_occ=0.01,
                          verbose=False):
    # read the schema file
    string_stats_path = os.path.join(os.path.dirname(__file__), '../../cross_db_benchmark/datasets/', dataset, 'string_statistics.json')
    if os.path.exists(string_stats_path) and not force:
        print("String stats already created")
        return

    schema = load_schema_json(dataset)
    column_stats = load_column_statistics(dataset)

    column_type_file = os.path.join(os.path.dirname(__file__), f'../../cross_db_benchmark/datasets/{dataset}/column_type.json')
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


    cols_with_freq_words = 0
    string_stats = dict()
    for table, cols in vars(column_stats).items():

        string_stats[table] = dict()
        # Get the dtype mapping for the current table
        dtype_mapping = get_dtypes_for_table(table, column_type)
        df_table_list = []
        if verbose:
            print(f"Generating string statistics for {table}")
        all_files = os.listdir(data_dir)
        table_files = [f for f in all_files if f.split('.')[0]==table and ('tbl' in f or 'csv' in f or 'dat' in f)]
        for f in table_files:
            # if verbose:
            #     print(f"file: {f}")
            table_dir = os.path.join(data_dir, f)

            # -------- 🔍 ① 先抽样一行，判断真实列数 --------
            expected_cols = len(tables[table])
            sep = getattr(schema.csv_kwargs, "sep", "|")  

            with open(table_dir, encoding=getattr(schema.csv_kwargs, "encoding", "utf-8")) as fh:
                first_non_blank = next(line for line in fh if line.strip())      # 找到第一行非空行
            # if verbose:
            #     print(f"first_non_blank: {first_non_blank}")
            actual_cols = first_non_blank.rstrip("\n").count(sep) + 1           # “|” 数 + 1 = 列数

            if actual_cols not in (expected_cols, expected_cols + 1):
                raise ValueError(f"{f}: unexpected column count {actual_cols} "
                                f"(expected {expected_cols} or {expected_cols+1})")

            # -------- 🏷️ ② 如果多 1 列，就临时加占位列名 --------
            names_for_read = list(tables[table]) + ["__dummy__"] if actual_cols == expected_cols + 1 else tables[table]
            
            print(f"Processing file {f}")
            try:
                df_table = pd.read_csv(table_dir, nrows=max_sample_vals, **vars(schema.csv_kwargs), names=names_for_read, dtype=dtype_mapping)
                
                if actual_cols == expected_cols + 1:
                    df_table.drop(columns="__dummy__", inplace=True)
               
                df_table_list.append(df_table)
            except Exception as e:
                print(f"Error reading file {f}: {e}")
                continue

        # Concatenate all dataframes for the current table
        df_table = pd.concat(df_table_list, ignore_index=True)
        
        # Function to check if a column has mixed types
        def has_mixed_types(series):
            # Check if the column has more than one unique dtype (excluding NaNs)
            return len(set(series.dropna().map(type))) > 1

        # Iterate over each column and convert columns with mixed types to string
        for col in df_table.columns:
            if has_mixed_types(df_table[col]):
                print(f"!!! Column '{col}' has mixed types. Converting to string.")
                # Convert the entire column to string
                df_table[col] = df_table[col].astype(str)

        for c, col_stats in vars(cols).items():
            if col_stats.datatype in {str(Datatype.CATEGORICAL), str(Datatype.MISC)}:
                col_vals = df_table[c]
                # do not consider too many values
                col_vals = col_vals[:max_sample_vals]
                len_strs = len(col_vals)

                # check how often a word occurs
                word_vals = collections.defaultdict(int)
                try:
                    split_col_vals = col_vals.str.split(' ')
                except:
                    continue

                for scol_vals in split_col_vals:
                    if not isinstance(scol_vals, list):
                        continue
                    for v in scol_vals:
                        if not isinstance(v, str):
                            continue
                        word_vals[v] += 1

                # how often should a word appear
                min_expected_occ = max(int(len_strs * min_str_occ), 1)

                freq_str_words = list()
                for val, occ in word_vals.items():
                    if occ > min_expected_occ:
                        freq_str_words.append(val)

                if len(freq_str_words) > 0:
                    if verbose:
                        print(f"Found {len(freq_str_words)} frequent words for {c} "
                              f"(expected {min_expected_occ}/{len_strs})")

                    cols_with_freq_words += 1
                    string_stats[table][c] = dict(freq_str_words=freq_str_words)

    # save to json
    with open(string_stats_path, 'w') as outfile:
        print(f"Found {cols_with_freq_words} string-queryable columns for dataset {dataset}")
        # workaround for numpy and other custom datatypes
        json.dump(string_stats, outfile)
