## Generate Training Data

### create database, create schema, load data, create indexes, alter all tables to be columnar

### generate column statistics
modify the 'dir' in the code to the place where csv or tbl data of the specific dataset is stored.
```
cd zsce
python generate_column_stats.py --dataset tpch_sf100
```

### generate string statistics
modify the 'dir' in the code to the place where csv or tbl data of the specific dataset is stored.

```
cd zsce
python generate_string_stats.py --dataset tpch_sf100
```

### generate analytical workloads
```
cd zsce
python generate_zsce_queries.py --workload_dir "/home/wuy/query_costs/workloads" --dataset "tpch_sf100"
```

### generate TP workloads (point queries)
```
cd zsce/cross_db_benchmark/benchmark_tools/
python generate_TP_workload.py --data_dir "/home/wuy/query_costs" --dataset "tpch_sf100"
```

### execute queries
```
python preprocessing/collect_query_costs_including_fann_model_and_hybrid.py --dataset tpch_sf100 --db tpch_sf100
```

## train model
```
make
./router --model lightgbm (--skip_train --data_dirs=tpch_sf100,tpch_sf1,tpdcs_sf1)
```
Currently supports lightgbm, rowmlp (implemented with FANN), dtree (decision tree), forest (random forest), all implemented in c++.