# PolarDB Row-Column Routing System

A machine learning-based query routing system for PolarDB that optimizes query execution by intelligently routing queries between row-store and column-store formats.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Training Pipeline](#training-pipeline)
- [Model Training](#model-training)
- [Resource Management](#resource-management)
- [Benchmarking](#benchmarking)
- [Project Structure](#project-structure)
- [Advanced Configuration](#advanced-configuration)

## Overview

This project implements an intelligent query routing system that uses machine learning models (LightGBM, Random Forest, Decision Tree, MLP) to determine the optimal storage format (row vs. column) for query execution in PolarDB.

### Key Components
- **PolarDB Version**: `PolarDB_802` (linthompson_new)
- **Routing Model**: `ensemble_model_32_features`
- **Supported ML Models**: LightGBM, RowMLP (FANN), Decision Tree, Random Forest

## Quick Start

### 1. Start MySQL Server
```bash
# Run PolarDB with constrained resources
bash run_polardb_constrained_resource.sh

# Start mysqld with custom configuration
./mysqld --defaults-file=/home/jitu.wy/mypolardb/db/fann_model.cnf &
```

### 2. Install Dependencies
```bash
yum install fann-devel
```

### 3. Train a Model
```bash
make
./router --model lightgbm
```

## Installation

### Prerequisites
- PolarDB 802
- FANN development libraries
- Python 3.x with required packages
- C++ compiler with C++11 support

### Setup Steps
1. Install FANN libraries: `yum install fann-devel`
2. Clone the repository
3. Build the project: `make -j$(nproc)`

## Training Pipeline

### Step 1: Database Setup
Create database, schema, load data, create indexes, and convert tables to columnar format.

### Step 2: Generate Statistics

#### Column Statistics
```bash
cd zsce
# Modify 'dir' in code to point to CSV/TBL data location
python generate_column_stats.py --dataset tpch_sf100
```

#### String Statistics
```bash
cd zsce
# Modify 'dir' in code to point to CSV/TBL data location
python generate_string_stats.py --dataset tpch_sf100
```

### Step 3: Generate Workloads

#### Analytical Workloads (OLAP)
```bash
cd zsce
python generate_zsce_queries.py \
    --workload_dir "/home/wuy/query_costs/workloads" \
    --dataset "tpch_sf100"
```

#### Transactional Workloads (OLTP - Point Queries)
```bash
cd zsce/cross_db_benchmark/benchmark_tools/
python generate_TP_workload.py \
    --data_dir "/home/wuy/query_costs" \
    --dataset "tpch_sf100"
```

### Step 4: Execute Queries and Collect Costs
```bash
python preprocessing/collect_query_costs_including_fann_model_and_hybrid.py \
    --dataset tpch_sf100 \
    --db tpch_sf100
```

For all datasets:
```bash
python preprocessing/collect_query_costs_trace_all_datasets.py
```

## Model Training

### C++ Implementation

#### Binary Classification Models
Build and train gap-regression LightGBM:
```bash
cd binary_classification
make -j$(nproc)
./router --mix
```

#### Time Regression Models
Train row/column time regression models:
```bash
cd time_regression
make -j$(nproc)
./router --mix
```

#### Available Model Options
```bash
# Train specific model
./router --model lightgbm

# Skip training and use existing model
./router --skip_train --data_dirs=tpch_sf100,tpch_sf1,tpdcs_sf1
```

Supported models:
- `lightgbm` - LightGBM gradient boosting
- `rowmlp` - MLP implemented with FANN
- `dtree` - Decision Tree
- `forest` - Random Forest

### Python Implementation

#### Feature Extraction
```bash
python feature_analysis.py extract tpch_sf1
```

#### Model Training
```bash
python feature_analysis.py train airline --use_idx shap_idx.npy
```

#### SHAP Analysis
Generate SHAP analysis and heatmaps:
```bash
python shap_analysis.py
```

## Resource Management

### Enable Resource Control
```sql
SET GLOBAL enable_resource_control = ON;
```

### Create Resource Control Groups
```sql
-- Create resource control for TP (transactional) workloads
CREATE POLAR_RESOURCE_CONTROL rc_tp MAX_CPU 30;

-- Create resource control for AP (analytical) workloads
CREATE POLAR_RESOURCE_CONTROL rc_ap MAX_CPU 60;
```

### Configure Users with Resource Limits
```sql
-- Create users
CREATE USER tp_user IDENTIFIED BY '***';
CREATE USER ap_user IDENTIFIED BY '***';

-- Assign resource controls to users
SET POLAR_RESOURCE_CONTROL rc_tp FOR USER tp_user;
SET POLAR_RESOURCE_CONTROL rc_ap FOR USER ap_user;
```

### CGroup Configuration (System-level Resource Constraints)
```bash
# Create cgroup
cgcreate -g cpu,memory:mysqld_grp

# Limit to 10% of a CPU core (10ms out of every 100ms)
echo 100000 | sudo tee /sys/fs/cgroup/cpu/mysqld_grp/cpu.cfs_period_us
echo 10000  | sudo tee /sys/fs/cgroup/cpu/mysqld_grp/cpu.cfs_quota_us

# Move mysqld process to the cgroup (replace 73474 with actual PID)
cgclassify -g cpu,memory:mysqld_grp 73474
```

## Benchmarking

Run combined benchmarks:
```bash
python combined_benchmark.py \
    --mysqld_pid <pid> \
    --rounds 1 \
    -n 1000 \
    --warmup_queries 100
```

## Project Structure

### Core Components

#### Binary Classification Module
Location: `binary_classification/`

Key files:
- `main.cpp` - Entry point
- `lightgbm_model.cpp/h` - LightGBM implementation
- `decision_tree_model.cpp` - Decision tree implementation
- `fannmlp_model.cpp` - FANN MLP implementation
- `common.cpp/hpp` - Common utilities
- `shap_util.hpp` - SHAP value utilities

#### Time Regression Module
Location: `time_regression/`

Contains similar structure to binary classification but optimized for regression tasks.

#### Supporting Files
- `cpu_meter.cpp/h` - CPU usage monitoring
- `global_stats.cpp/hpp` - Global statistics management
- `thresholds.hpp` - Threshold configurations
- `vib.hpp` - Variational Information Bottleneck utilities

## Advanced Configuration

### Custom Dataset Training
To train on multiple datasets:
```bash
./router --skip_train --data_dirs=dataset1,dataset2,dataset3
```

### Feature Selection
Use SHAP-based feature selection:
```bash
python feature_analysis.py train <dataset> --use_idx shap_idx.npy
```

### Performance Monitoring
Monitor CPU usage during training and inference using the built-in CPU meter functionality.

