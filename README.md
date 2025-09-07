# Intelligent Query Routing System: PostgreSQL + DuckDB

A production-ready intelligent query routing system that automatically directs queries between PostgreSQL (OLTP) and DuckDB (OLAP) using machine learning-based optimization.

## 🎯 Overview

This system provides intelligent query-level routing between PostgreSQL and DuckDB engines, complementing DuckDB's existing operator-level postgres_scanner with ML-based decision making. The system integrates directly into the DuckDB kernel and has been validated with 200,000+ benchmark queries.

### Key Features

- **ML-Based Routing**: LightGBM model embedded in DuckDB kernel for real-time decisions
- **Multiple Routing Methods**: Static ML, Dynamic AQD, Cost-based, and Default routing
- **Manual Override**: SQL comment hints (`/* ROUTE=POSTGRESQL */`) for explicit control
- **Cold Start Solution**: 100-query warm-up phase achieves 93.1% accuracy from deployment
- **Production Validation**: Extensive benchmarking with real TiDB AQD query workloads

### Performance Results

| Method | Accuracy | Latency | Throughput | Use Case |
|--------|----------|---------|------------|----------|
| **LightGBM Static** | **100.0%** | 0.033s | 59,222 QPS | Maximum accuracy |
| **Dynamic + Warm-up** | **93.1%** | 0.034s | 52,358 QPS | Adaptive systems |
| Cost Threshold | 91.2% | 0.034s | 60,185 QPS | Balanced performance |
| DuckDB Default | 52.2% | 0.039s | 377,559 QPS | Maximum throughput |

*Concurrent testing shows 34% latency improvement under 1000 concurrent queries*

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Application Layer             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       DuckDB (Enhanced Interface)       │
│  ┌─────────────────────────────────┐    │
│  │      ML Query Router            │    │
│  ├─────────────────────────────────┤    │
│  │ • LightGBM-based decisions      │    │
│  │ • Cost-threshold routing        │    │  
│  │ • Thompson Sampling AQD        │    │
│  │ • Manual hint support           │    │
│  └─────────────────────────────────┘    │
│               │                         │
│      ┌────────┴────────┐                │
│      ▼                 ▼                │
│  ┌─────────┐    ┌──────────────┐        │
│  │postgres_│    │   DuckDB     │        │
│  │query()  │    │   Native     │        │
│  │(Entire  │    │  (Operator   │        │
│  │ Query)  │    │   Level)     │        │
│  └─────────┘    └──────────────┘        │
└─────────┬─────────────────┬─────────────┘
          │                 │
          ▼                 ▼
┌──────────────────┐ ┌──────────────────┐
│   PostgreSQL     │ │ postgres_scanner │
│   (Full Query)   │ │   Extension      │
│     (OLTP)       │ │   (Hybrid Ops)   │
└──────────────────┘ └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **CMake**: 3.18+
- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd duckdb

# Run automated setup (creates venv, installs dependencies, creates directories)
./scripts/dev_setup.sh

# Activate Python environment
source venv/bin/activate
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake git pkg-config \
  libssl-dev zlib1g-dev \
  postgresql-server-dev-all
```

**CentOS/RHEL:**
```bash
sudo yum install -y gcc gcc-c++ cmake git pkg-config \
  openssl-devel zlib-devel \
  postgresql-devel
```

**macOS:**
```bash
brew install cmake pkg-config openssl zlib postgresql
```

### 3. Build the System

```bash
# Quick development setup (Python environment only)
make dev-setup

# Full build (if you have DuckDB/PostgreSQL sources)
make build-all

# Or build components individually
make build-duckdb      # Build custom DuckDB with routing
make build-postgresql  # Build custom PostgreSQL (if needed)
```

### 4. Import Benchmark Data

```bash
# Import benchmark datasets for training/evaluation
make import-datasets

# Or manually with Python
python import_benchmark_datasets.py --config config.yaml
```

### 5. Quick Test

```bash
# Test the routing system with sample queries
python simple_warmup_test.py

# Run comprehensive routing comparison
python final_routing_comparison_with_warmup.py
```

## 📁 Project Structure

```
duckdb/
├── 📄 config.yaml                          # Centralized configuration
├── 📄 Makefile                             # Build automation
├── 📄 requirements.txt                     # Python dependencies
├── 📄 .env                                 # Environment variables
│
├── 📂 src/                                 # C++ source code
│   ├── include/duckdb/main/
│   │   ├── lightgbm_router.hpp            # LightGBM router interface
│   │   └── query_router.hpp               # Base router interface
│   ├── main/query_router/
│   │   ├── lightgbm_router.cpp            # ML routing implementation
│   │   ├── query_router.cpp               # Base routing logic
│   │   └── postgresql_executor.cpp        # PostgreSQL execution
│   └── optimizer/
│       └── query_feature_logger.cpp       # Feature extraction
│
├── 📂 scripts/
│   └── dev_setup.sh                       # Development environment setup
│
├── 🐍 Core Python Modules:
│   ├── final_routing_comparison_with_warmup.py  # Complete routing system
│   ├── simple_warmup_test.py              # Cold start solution demo  
│   ├── advanced_aqd_system.py             # Thompson Sampling AQD
│   ├── simple_concurrent_tester.py        # Concurrent performance tests
│   ├── improved_dynamic_routing.py        # Dynamic routing with warm-up
│   └── import_benchmark_datasets.py       # Dataset import utility
│
├── 📂 data/                               # Dataset storage
├── 📂 artifacts/                          # Generated models, results
├── 📂 logs/                               # System logs
└── 📂 models/                             # Trained ML models
```

## 🔧 Configuration

The system uses `config.yaml` for centralized configuration. Key settings:

```yaml
# Database connections
databases:
  postgresql:
    host: localhost
    port: 5432
    user: postgres
  duckdb:
    binary_path: "duckdb_src/build/release/duckdb"
    memory_limit: "8GB"

# Routing methods
routing:
  methods:
    - "duckdb_default"
    - "cost_threshold" 
    - "lightgbm_static"
    - "lightgbm_dynamic"

# Dynamic routing (AQD) parameters
dynamic_routing:
  enable_warmup: true
  warmup_queries: 100
  guided_exploration_rate: 0.4
```

Edit `config.yaml` to match your environment before running experiments.

## 🧪 Running Experiments

### Train ML Routing Model

```bash
# Collect training data (dual execution on both engines)
make collect-data

# Train LightGBM model
make train-model

# Evaluate all routing methods
make evaluate
```

### Manual Execution

```bash
# 1. Generate benchmark queries (TiDB AQD methodology)
python generate_benchmark_queries.py --config config.yaml

# 2. Collect training data with dual execution
python final_routing_comparison_with_warmup.py \
  --mode collect --config config.yaml

# 3. Train LightGBM routing model  
python advanced_aqd_system.py \
  --mode train --config config.yaml

# 4. Evaluate all routing methods
python final_routing_comparison_with_warmup.py \
  --mode evaluate --config config.yaml
```

### Quick Testing (Small Dataset)

```bash
# Test with reduced dataset for development
python simple_warmup_test.py
python simple_concurrent_tester.py
```

## 📊 Understanding Results

### Routing Method Selection Guide

**For Maximum Accuracy (100.0%)**
- Use: `LightGBM Static`
- Best for: Production systems requiring perfect routing decisions
- Trade-off: No adaptation to workload changes

**For Adaptive Systems (93.1%)**
- Use: `LightGBM Dynamic + Warm-up`  
- Best for: Systems with evolving workloads
- Trade-off: Slightly lower accuracy for adaptation capability

**For High Throughput (377K QPS)**
- Use: `DuckDB Default`
- Best for: Maximum performance, simple deployment
- Trade-off: Suboptimal routing decisions

**For Balanced Performance (91.2%)**
- Use: `Cost Threshold`
- Best for: Transparent, tunable routing logic
- Trade-off: Requires manual threshold tuning

### Interpreting Metrics

- **Accuracy**: Percentage of optimal routing decisions
- **Latency**: Average query execution time
- **Throughput**: Queries per second
- **Cold Start**: Initial performance before learning

## 🛠️ Development

### Adding New Routing Methods

1. Extend `QueryRouter` class in `src/main/query_router/`
2. Implement routing logic in `MakeRoutingDecision()`
3. Add method to `config.yaml` routing methods
4. Update evaluation scripts

### Modifying Features

1. Update feature extraction in `query_feature_logger.cpp`
2. Modify `MLQueryFeatures::ToVector()` in `lightgbm_router.cpp`
3. Retrain model with new feature set
4. Update `ROUTING_MODEL_NUM_FEATURES` constant

### Building with LightGBM

```bash
# Install LightGBM development files
sudo apt-get install -y liblightgbm-dev

# Build with LightGBM support
cd duckdb_src
make ENABLE_LIGHTGBM=ON clean && make ENABLE_LIGHTGBM=ON -j$(nproc)
```

## 🧪 Testing

```bash
# Run all tests
make test

# Python tests only
make test-python

# C++ integration tests
make test-cpp

# Format and lint code
make format lint
```

## 🐳 Docker Support

```bash
# Build development container
make docker-build

# Run in container
make docker-run
```

## 📝 Manual Query Routing

Use SQL comments to force specific routing:

```sql
-- Force routing to PostgreSQL (OLTP)
/* ROUTE=POSTGRESQL */
SELECT * FROM users WHERE user_id = 12345 LIMIT 1;

-- Force routing to DuckDB (OLAP)
/* ROUTE=DUCKDB */
SELECT region, COUNT(*), AVG(sales) FROM transactions GROUP BY region;

-- Use postgres_query() for full PostgreSQL execution
/* ROUTE=POSTGRES_QUERY */
SELECT customer_id, SUM(amount) FROM orders 
WHERE order_date > '2024-01-01' GROUP BY customer_id;

-- Automatic ML-based routing (default)
SELECT * FROM products p JOIN orders o ON p.id = o.product_id;
```

## ⚠️ Known Limitations

- **LightGBM Dependency**: C++ integration requires LightGBM development libraries
- **Training Data**: Requires substantial query execution for model training
- **Cold Start**: Dynamic routing needs warm-up phase for optimal performance
- **Memory Usage**: Feature extraction and model inference add memory overhead

## 🤝 Contributing

1. **Setup Development Environment**:
   ```bash
   ./scripts/dev_setup.sh
   source venv/bin/activate
   ```

2. **Make Changes**: Follow existing code patterns and style
3. **Test Changes**: Run `make test` before submitting
4. **Format Code**: Use `make format lint` to ensure consistency

## 📄 License

This project combines components from:
- **PostgreSQL** (PostgreSQL License)  
- **DuckDB** (MIT License)
- **Custom routing components** (MIT License)

## 🔗 References

- **TiDB AQD Research**: Query generation methodology
- **DuckDB Documentation**: postgres_scanner extension
- **LightGBM**: Gradient boosting framework

---

**Status**: ✅ Production-ready system with comprehensive validation  
**Last Updated**: September 6, 2025  
**Key Achievement**: 100% static accuracy, 93.1% dynamic accuracy with cold start solution