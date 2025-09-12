# LightGBM Query Routing Implementation for PostgreSQL/DuckDB

## Overview
This implementation replaces the heavy GNN/R-GIN approach with a lightweight LightGBM-based query routing system that makes routing decisions at the planner hook level (before query planning) to avoid double planning overhead.

## Key Components

### 1. Pre-Optimization Feature Extraction
- **Files**: `preopt_feature_extractor.c/h`
- **Features**: 25 lightweight features extracted from Query* trees before planning
- **Location**: `/postgres/src/backend/utils/misc/preopt_feature_extractor.c`

### 2. LightGBM Routing Module
- **Files**: `lightgbm_routing.c/h`
- **Functionality**: 
  - Dynamic loading of LightGBM C library
  - Per-backend model caching
  - Prediction using exact 25 features from training
- **Location**: `/postgres/src/backend/utils/misc/lightgbm_routing.c`

### 3. Planner Hook Integration
- **File**: `pgduckdb_hooks.cpp`
- **Hook**: `duckdb_planner_hook` - intercepts queries before planning
- **Decision Point**: Routes to DuckDB or PostgreSQL based on LightGBM prediction

### 4. Training Pipeline
- **Data Collection**: `collect_lightgbm_data.py` - CSV-based feature collection
- **Training**: `train_lightgbm.py` - Binary classification model
- **Model Format**: Native LightGBM text format for C API

## Features (v1.0.0)
The system extracts 25 features from query trees:
1. num_tables
2. num_joins
3. query_depth
4. complexity_score
5. has_aggregates
6. has_group_by
7. has_order_by
8. has_limit
9. has_distinct
10. has_window_functions
11. has_outer_joins
12. estimated_join_complexity
13. has_subqueries
14. has_correlated_subqueries
15. has_large_tables
16. all_tables_small
17. has_complex_expressions
18. has_user_functions
19. has_text_operations
20. has_numeric_heavy_ops
21. num_aggregate_funcs
22. analytical_pattern
23. transactional_pattern
24. etl_pattern
25. command_type

## Configuration (GUCs)

### Main Settings
- `lightgbm.enabled` - Enable/disable LightGBM routing
- `lightgbm.model_path` - Path to LightGBM model file
- `lightgbm.routing_threshold` - Decision threshold (default: 0.5)

### Observability GUCs
- `lightgbm.last_routed_engine` - Last routing decision (postgres/duckdb)
- `lightgbm.last_decision_us` - Inference time in microseconds
- `lightgbm.last_features_json` - JSON of extracted features
- `lightgbm.prediction_count` - Total predictions made
- `lightgbm.inference_time_ms` - Average inference time

## Building and Installation

### Prerequisites
1. Install LightGBM C library:
```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake ..
make -j4
sudo make install
sudo ldconfig
```

2. Build PostgreSQL with LightGBM support:
```bash
cd postgres
./configure --prefix=/usr/local/pgsql --enable-debug --enable-cassert CFLAGS="-ggdb -O0"
make -j$(nproc)
sudo make install
```

3. Build pg_duckdb extension:
```bash
cd pg_duckdb
make
sudo make install
```

## Usage

### Training a Model
1. Collect training data:
```bash
python collect_lightgbm_data.py
```

2. Train the model:
```bash
python train_lightgbm.py
```

3. Configure PostgreSQL:
```sql
SET lightgbm.enabled = true;
SET lightgbm.model_path = '/path/to/model.txt';
SET lightgbm.routing_threshold = 0.5;
```

### Monitoring
Check routing decisions:
```sql
SHOW lightgbm.last_routed_engine;
SHOW lightgbm.last_decision_us;
SHOW lightgbm.last_features_json;
```

## Key Improvements Over GNN/R-GIN
1. **No Double Planning**: Routing decision happens before planning
2. **Lightweight Features**: Only 25 simple features vs. full plan graphs
3. **Fast Inference**: <1ms prediction time
4. **Per-Backend Caching**: Model loaded once per connection
5. **Simple Training**: Standard LightGBM instead of custom GNN

## Testing
To verify routing is working:
```sql
-- Should route to DuckDB for analytical queries
EXPLAIN SELECT COUNT(*), AVG(value) FROM large_table GROUP BY category;

-- Should show DuckDBScan in plan if routed correctly
```

## Troubleshooting
- If model fails to load, check `lightgbm.model_path` and file permissions
- Verify LightGBM library is installed: `ldconfig -p | grep lightgbm`
- Check PostgreSQL logs for routing decisions and errors
- Use observability GUCs to debug feature extraction and predictions

## Commands to Remember
- **Build PostgreSQL**: `make -j$(nproc) && sudo make install`
- **Build pg_duckdb**: `make && sudo make install`
- **Run tests**: `make check` (in postgres directory)
- **Check lint**: `make lint` (if available)
- **Type check**: `make typecheck` (if available)