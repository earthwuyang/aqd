# PostgreSQL 17 + pg_duckdb with LightGBM Query Routing

A high-performance PostgreSQL 17 environment with pg_duckdb extension and LightGBM-based query routing that makes routing decisions at the planner hook level (before query planning) to avoid double planning overhead.

## 🚀 Key Features

- **PostgreSQL 17** with pre-optimization query tree feature extraction
- **pg_duckdb v1.1.0** - DuckDB analytical engine integration  
- **LightGBM Model** - Lightweight gradient boosting for fast query routing (<1ms)
- **Pre-planning routing** - Decisions made before query planning to avoid overhead
- **Per-backend model caching** - Model loaded once per connection
- **25 lightweight features** - Extracted from Query trees without planning
- **9 benchmark databases** - 7 CTU datasets + TPC-H/TPC-DS (SF=1)
- **Dual execution collection** - Training data from both engines
- **Comprehensive observability** - GUCs for monitoring routing decisions

## 📋 Prerequisites

```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    build-essential libreadline-dev zlib1g-dev \
    flex bison libxml2-dev libxslt1-dev libssl-dev \
    cmake ninja-build pkg-config

# Python packages  
pip install psycopg2-binary mysql-connector-python tqdm numpy pandas lightgbm scikit-learn

# LightGBM C library
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake ..
make -j4
sudo make install
sudo ldconfig
```

## 📁 Repository Structure

```
pg_duckdb_postgres/
├── postgres/                    # Modified PostgreSQL 17 source
│   ├── src/include/utils/
│   │   ├── lightgbm_routing.h  # LightGBM routing API
│   │   └── preopt_feature_extractor.h # Pre-optimization features
│   └── src/backend/utils/misc/
│       ├── lightgbm_routing.c  # LightGBM C API integration
│       └── preopt_feature_extractor.c # Feature extraction
├── pg_duckdb/                   # pg_duckdb extension with planner hook
├── pgsql/                       # PostgreSQL installation
├── data/                        # PostgreSQL data directory
├── lightgbm_models/             # Trained LightGBM models
├── lightgbm_training_data/      # CSV training data
├── benchmark_queries/           # Generated queries with cache
├── dual_execution_data/         # Execution timing data
├── collect_lightgbm_data.py    # Data collection script
├── train_lightgbm.py           # Model training script
└── CLAUDE.md                    # Implementation documentation
```

## 🔧 Building and Installation

### 1. Build PostgreSQL with LightGBM support

```bash
cd postgres
./configure --prefix=/home/wuy/DB/pg_duckdb_postgres/pgsql \
    --enable-debug --enable-cassert CFLAGS="-ggdb -O0"
make -j$(nproc)
make install
```

### 2. Build pg_duckdb extension

```bash
cd pg_duckdb
export PATH=/home/wuy/DB/pg_duckdb_postgres/pgsql/bin:$PATH
make
make install
```

### 3. Initialize database

```bash
export PATH=/home/wuy/DB/pg_duckdb_postgres/pgsql/bin:$PATH
initdb -D data
pg_ctl -D data start
createdb test
```

## 🎯 LightGBM Query Routing

### Configuration (GUCs)

```sql
-- Main settings
SET lightgbm.enabled = true;
SET lightgbm.model_path = '/path/to/lightgbm_model.txt';
SET lightgbm.routing_threshold = 0.0;  -- >0 favors DuckDB

-- Feature extraction control
SET lightgbm.enable_preopt_feature_extraction = true;  -- default ON
SET lightgbm.enable_plan_logging = false;              -- default OFF for performance

-- Observability (read-only)
SHOW lightgbm.last_routed_engine;     -- 'postgres' or 'duckdb'
SHOW lightgbm.last_decision_us;       -- routing overhead in microseconds
SHOW lightgbm.last_features_json;     -- extracted features as JSON
SHOW lightgbm.prediction_count;       -- total predictions made
SHOW lightgbm.inference_time_ms;      -- average inference time
```

### Pre-Optimization Features (v1.0.0)

The system extracts 25 lightweight features from Query trees before planning:

1. **Query structure**: num_tables, num_joins, query_depth, complexity_score
2. **Query clauses**: has_aggregates, has_group_by, has_order_by, has_limit, has_distinct
3. **Advanced features**: has_window_functions, has_outer_joins, has_subqueries
4. **Complexity indicators**: has_correlated_subqueries, has_complex_expressions
5. **Function analysis**: has_user_functions, has_text_operations, has_numeric_heavy_ops
6. **Pattern detection**: analytical_pattern, transactional_pattern, etl_pattern
7. **Command type**: SELECT=0, INSERT=1, UPDATE=2, DELETE=3, OTHER=4

## 📊 Training Pipeline

### 1. Collect training data

```bash
# Collects features and dual-engine execution times
python collect_lightgbm_data.py \
    --databases tpch,tpcds,accidents,airline \
    --queries-per-db 1000 \
    --output lightgbm_training_data/training.csv
```

### 2. Train LightGBM model

```bash
# Trains regression model on log(pg_time/duck_time)
python train_lightgbm.py \
    --input lightgbm_training_data/training.csv \
    --output lightgbm_models/model.txt \
    --calibrate-threshold  # Optimizes for minimum makespan
```

### 3. Deploy model

```sql
-- In PostgreSQL
ALTER SYSTEM SET lightgbm.model_path = '/absolute/path/to/model.txt';
ALTER SYSTEM SET lightgbm.enabled = true;
SELECT pg_reload_conf();
```

## 🔍 Verification

### Check routing decisions

```sql
-- Run a query
SELECT COUNT(*), AVG(value) FROM large_table GROUP BY category;

-- Check which engine was selected
SHOW lightgbm.last_routed_engine;  -- Should show 'duckdb' for analytical

-- Verify with EXPLAIN
EXPLAIN SELECT COUNT(*), AVG(value) FROM large_table GROUP BY category;
-- Should show "Custom Scan (DuckDBScan)" if routed to DuckDB
```

### Monitor performance

```sql
-- View routing overhead
SHOW lightgbm.last_decision_us;  -- Should be <1000 (under 1ms)

-- View extracted features
SHOW lightgbm.last_features_json;

-- Check prediction statistics
SHOW lightgbm.prediction_count;
SHOW lightgbm.inference_time_ms;
```

## 🏃 Running Benchmarks

```bash
# Generate benchmark queries
python generate_benchmark_queries.py --num-ap 1000 --num-tp 1000

# Run benchmark comparison
python run_benchmark.py \
    --methods default,cost,lightgbm \
    --databases tpch,tpcds \
    --persistent-connections \
    --interleave \
    --output results.csv
```

## 🔬 Key Improvements Over Previous Approaches

| Aspect | GNN/R-GIN Approach | LightGBM Approach |
|--------|-------------------|-------------------|
| **Decision Point** | After planning (too late) | Before planning (planner hook) |
| **Features** | Full plan graphs (heavy) | 25 pre-opt features (lightweight) |
| **Inference Time** | >10ms | <1ms |
| **Model Loading** | Per query | Once per backend |
| **Training** | Complex GNN | Standard gradient boosting |
| **Double Planning** | Yes (plan then re-plan) | No (route then plan once) |
| **Feature Extraction** | Requires planning | Query tree only |

## 🐛 Troubleshooting

- **Model fails to load**: Check file path and permissions, verify LightGBM library is installed
- **Wrong engine selected**: Check threshold calibration, verify features match training
- **High routing overhead**: Ensure model is cached per-backend, not reloaded per query
- **Features mismatch**: Verify feature schema version matches between training and kernel

## 📚 Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed implementation documentation
- [ChatGPT Review](docs/chatgpt_review.md) - Architecture recommendations

## 🧪 Testing

```bash
# PostgreSQL regression tests
cd postgres
make check

# pg_duckdb tests
cd pg_duckdb
make installcheck
```

## 📝 License

PostgreSQL License for PostgreSQL components, MIT License for routing implementation.

## 🙏 Acknowledgments

- PostgreSQL Community
- DuckDB Labs (pg_duckdb extension)
- Microsoft Research (LightGBM)
- CTU Prague (benchmark datasets)