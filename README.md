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
SET lightgbm.routing_strategy = 'lightgbm';  -- 'lightgbm' or 'threshold'
SET lightgbm.cost_threshold = 50000;         -- Postgres cost cutoff when using threshold routing
SET lightgbm.routing_threshold = 0.0;        -- LightGBM score threshold (>0 favors DuckDB)

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

### Threshold-Based Routing

The planner hook can now choose between LightGBM predictions and a simple cost threshold.

- `lightgbm.routing_strategy` selects the policy (`lightgbm` by default, set to `threshold` to use planner cost).
- `lightgbm.cost_threshold` sets the Postgres `total_cost` ceiling for running on Postgres; higher costs are routed to DuckDB.
- `lightgbm.routing_threshold` continues to control the LightGBM score cutoff when the ML strategy is active.

To sweep multiple thresholds, pass `--thresholds` to `performance_test.py` (for example `--thresholds 10000,50000,150000`). The runner now executes all-postgres, all-duckdb, each requested threshold policy, and LightGBM in one pass and reports comparative statistics.

### Pre-Optimization Features (v2.1.0)

Routing decisions are driven by a 60-element feature vector extracted directly from the PostgreSQL `Query` tree—no planning required. Features fall into the following themes (see `LIGHTGBM_FEATURE_NAMES` in `src/include/utils/preopt_feature_extractor.h` for the exact order):

1. **Query structure** – table/join counts, heuristic complexity score, query depth, command type.
2. **Clause presence** – booleans for aggregates, GROUP BY, ORDER BY, LIMIT, DISTINCT, window functions, outer joins, subqueries, correlated subqueries.
3. **Expression complexity** – whether the statement uses complex expressions, user-defined or text-heavy operators, numeric-heavy expressions, aggregate count, and workload patterns (analytical / transactional / ETL).
4. **Join analysis** – counts per join type (inner/left/right/full/cross) to hint at star schemas vs OLTP joins.
5. **Predicate categories** – simple equality, range, LIKE, IN, EXISTS checks, plus flags for parameters, CTE usage, recursive CTEs, lateral joins.
6. **Selectivity & cardinality heuristics** – rough buckets for expected selectivity/cardinality and whether index usage or partition pruning is likely; also captures parallel-safety, volatile functions, and coarse cost buckets.
7. **Projection & result metrics** *(new in v2.1.0)* – per-table projected byte totals, average/max projected-row fractions, projected column counts broken down by type (text, numeric, JSON), estimated result row width, and LIMIT/ORDER BY interactions.

These additions provide quantitative signals about projection sparsity and top‑K patterns—key indicators for choosing between row- and column-stores.

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

## 🧪 Sample Routing Experiment

The table below shows wall-clock timings for `SELECT COUNT(*) FROM lineitem` on the `tpch_sf1` database running locally.  Each mode was executed twice in a single session using `psycopg2`; averages are reported after warming up buffers.  A LightGBM model was loaded but routes defaulted to PostgreSQL because no learned threshold favoured DuckDB for this query.

| Mode                | Settings Summary                                       | Avg Time (ms) | Routed Engine | Notes                         |
|---------------------|---------------------------------------------------------|---------------|----------------|-------------------------------|
| All PostgreSQL      | `duckdb.force_execution = false`, `lightgbm.enabled = false` | 228.9         | postgres       | Baseline planner/executor     |
| All DuckDB          | `duckdb.force_execution = true`, `lightgbm.enabled = false`  | 328.8         | duckdb         | Forced into DuckDB scan       |
| Threshold (50k)     | `routing_strategy='threshold'`, `cost_threshold=50000`       | 212.3         | postgres       | Cost below threshold retained |
| LightGBM (score)    | `routing_strategy='lightgbm'`, `routing_threshold=0.0`       | 215.7         | postgres       | Model predicted Postgres      |

Even without a trained model that favours DuckDB, the new threshold mode makes it easy to sweep cost cutoffs and confirm that this analytical count still performs best on PostgreSQL for the chosen data layout.  After recompiling the extension, rerun the experiment above or invoke `performance_test.py --thresholds` to profile additional workloads.

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
