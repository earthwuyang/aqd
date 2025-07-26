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



## python_impl
### first extract features
```
python feature_analysis.py extract tpch_sf1
```

### train 
```
python feature_analysis.py train airline --use_idx shap_idx.npy
```

## resource control
set global enable_resource_control=on;

mysql> create polar_resource_control rc_tp max_cpu 30;
Query OK, 0 rows affected (0.00 sec)

mysql> create polar_resource_control rc_ap max_cpu 60;
Query OK, 0 rows affected (0.01 sec)

mysql> create user tp_user identified by '***';
Query OK, 0 rows affected (0.00 sec)

mysql> create user ap_user identified by '***';
Query OK, 0 rows affected (0.01 sec)

mysql> set polar_resource_control rc_tp for user tp_user;
Query OK, 0 rows affected (0.01 sec)

mysql> set polar_resource_control rc_ap for user ap_user;
Query OK, 0 rows affected (0.01 sec)


## main files explanation
### train gap-regression lightgbm
in binary_classification:
Makefile, main.cpp
common.cpp, common.hpp, cpu_meter.cpp, cpu_meter.h, decision_tree_model.cpp, fannmlp_model.cp, gin_model.cpp, global_stats.cpp, global_stats.hpp, lightgbm_model.cpp, lightgbm_model.h, shap_util.hpp, thresholds.hpp, vib.hpp

usage: make -j$($nproc)

./router --mix

### train two row/column time regression lightgbm
in time_regression:
Makefile, main.cpp
common.cpp, common.hpp, cpu_meter.cpp, cpu_meter.h, decision_tree_model.cpp, fannmlp_model.cp, gin_model.cpp, global_stats.cpp, global_stats.hpp, lightgbm_model.cpp, lightgbm_model.h, shap_util.hpp, thresholds.hpp, vib.hpp

usage: make -j$($nproc)

./router --mix

### benchmark files usage
python combined_benchmark.py --mysqld_pid <pid> --rounds 1 -n 1000 --warmup_queries 100


### collect training data
python preprocessing/collect_query_costs_trace_all_datasets.py

## shap analysis and draw heatmap
python shap_analysis.py