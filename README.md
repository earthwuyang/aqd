# PostgreSQL 17 + pg_duckdb with CTU Benchmark Datasets

A complete PostgreSQL 17 environment with pg_duckdb extension, CTU benchmark datasets, and comprehensive query generation tools for hybrid OLTP/OLAP workload testing.

## 🚀 Features

- **PostgreSQL 17** (REL_17_STABLE) compiled with debug symbols
- **pg_duckdb v1.1.0** - DuckDB analytical engine integration
- **7 CTU benchmark datasets** (~8M rows) from relational.fel.cvut.cz
- **TPC-H & TPC-DS** benchmarks at scale factor 1 (1GB each)
- **14,000+ benchmark queries** using TiDB AQD methodology
- **Automated setup scripts** for reproducible deployment

## 📋 Prerequisites

```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    build-essential libreadline-dev zlib1g-dev \
    flex bison libxml2-dev libxslt1-dev libssl-dev \
    cmake ninja-build pkg-config

# Python packages
pip install psycopg2-binary mysql-connector-python tqdm numpy
```

## 📁 Repository Structure

```
pg_duckdb_postgres/
├── postgres/                    # PostgreSQL 17 source
├── pg_duckdb/                   # pg_duckdb extension source
├── pgsql/                       # PostgreSQL installation
├── data/                        # PostgreSQL data directory
├── ctu_data/                    # CTU dataset CSV cache
├── benchmark_queries/           # Generated queries (14,000+)
│   ├── Airline/                 # 2,000 queries per dataset
│   ├── Credit/                  # (1,000 AP + 1,000 TP)
│   ├── Carcinogenesis/
│   ├── employee/
│   ├── financial/
│   ├── geneea/
│   └── Hepatitis_std/
├── tpch-dbgen/                  # TPC-H generator
├── databricks-tpcds/            # TPC-DS generator
├── setup_postgres.sh            # PostgreSQL + pg_duckdb setup
├── setup_tpc_benchmarks.sh      # TPC-H/DS setup
├── import_ctu_datasets.py       # CTU importer
├── generate_benchmark_queries.py # Query generator
└── fix_failed_tables.py         # Table repair utility
```

## 🛠️ Installation

### Quick Setup (Recommended)

```bash
# 1. Setup PostgreSQL + pg_duckdb
./setup_postgres.sh

# 2. Import CTU datasets
python3 import_ctu_datasets.py

# 3. Generate benchmark queries
python3 generate_benchmark_queries.py

# 4. (Optional) Setup TPC benchmarks
./setup_tpc_benchmarks.sh
```

### Manual Setup

```bash
# Build PostgreSQL 17
git clone --branch REL_17_STABLE https://github.com/postgres/postgres.git
cd postgres
./configure --prefix=$PWD/../pgsql --enable-debug --enable-cassert
make -j$(nproc) && make install
cd ..

# Initialize database
export PATH=$PWD/pgsql/bin:$PATH
initdb -D data
pg_ctl -D data start

# Build pg_duckdb
git clone https://github.com/duckdb/pg_duckdb.git
cd pg_duckdb
git submodule update --init --recursive
make && make install
cd ..
```

## 📊 CTU Datasets

| Database | Tables | Records | Description |
|----------|--------|---------|-------------|
| **Airline** | 19 | 445,827 | Flight performance data (2016) |
| **Credit** | 8 | 1,646,563 | Credit card transactions |
| **Carcinogenesis** | 6 | ~330k | Chemical carcinogenesis research |
| **employee** | 6 | ~4k | Employee management system |
| **financial** | 8 | 1,056,320 | Czech bank transactions (1993-1998) |
| **geneea** | 19 | ~240k | Text analysis and NLP data |
| **Hepatitis_std** | 7 | ~750 | Hepatitis patient records |

### Import Commands

```bash
# Import all datasets
python3 import_ctu_datasets.py

# Force re-import
python3 import_ctu_datasets.py --force

# Fix failed tables
python3 fix_failed_tables.py
```

## 🔍 Query Generation

The benchmark includes two query types based on workload characteristics:

### AP Queries (Analytical Processing)
- Complex joins (1-5 tables)
- Aggregations (COUNT, SUM, AVG, MIN, MAX, STDDEV)
- GROUP BY, HAVING, ORDER BY
- Large result sets
- Optimized for DuckDB engine

### TP Queries (Transactional Processing)  
- Simple joins (1-3 tables)
- Point lookups with high selectivity
- Equality predicates on indexed columns
- Small result sets (LIMIT 1-100)
- Optimized for PostgreSQL engine

### Generation Examples

```bash
# Default: 1,000 AP + 1,000 TP per database
python3 generate_benchmark_queries.py

# Custom generation
python3 generate_benchmark_queries.py \
    --databases financial employee \
    --num-ap 5000 \
    --num-tp 5000 \
    --output-dir custom_queries/
```

## 💻 Database Access

### Connection

```bash
# Unix socket (recommended)
psql -h /tmp -p 5432 -d financial

# List databases
psql -h /tmp -p 5432 -c "\l" postgres

# Enable pg_duckdb
psql -d financial -c "CREATE EXTENSION IF NOT EXISTS pg_duckdb;"
```

### Sample Queries

```sql
-- Analytical query (uses DuckDB)
SELECT 
    t.type,
    DATE_TRUNC('month', t.date) as month,
    COUNT(*) as txn_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM trans t
JOIN account a ON t.account_id = a.account_id
WHERE t.date >= '1995-01-01'
GROUP BY t.type, DATE_TRUNC('month', t.date)
HAVING COUNT(*) > 100
ORDER BY month, total_amount DESC;

-- Transactional query (uses PostgreSQL)
SELECT c.*, a.frequency, a.date
FROM client c
JOIN disp d ON c.client_id = d.client_id  
JOIN account a ON d.account_id = a.account_id
WHERE c.client_id = 5000
LIMIT 1;
```

## ⚙️ Configuration

### PostgreSQL Settings (data/postgresql.conf)

```conf
# Memory
shared_buffers = 2GB
work_mem = 128MB
maintenance_work_mem = 512MB
effective_cache_size = 8GB

# pg_duckdb
duckdb.execution = on
duckdb.postgres_role = 'read_only'

# Logging
log_statement = 'all'
log_duration = on
```

### Environment Setup

```bash
# Add to ~/.bashrc
export PGDATA=/home/$USER/DB/pg_duckdb_postgres/data
export PATH=/home/$USER/DB/pg_duckdb_postgres/pgsql/bin:$PATH
```

## 📈 Performance Monitoring

```sql
-- Check query execution engine
EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) FROM trans WHERE amount > 1000;

-- Table statistics
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_live_tup AS rows
FROM pg_tables t
JOIN pg_stat_user_tables s ON t.tablename = s.relname
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Update statistics
ANALYZE;
```

## 🔧 Maintenance

```bash
# Start/stop PostgreSQL
pg_ctl -D data start/stop/restart/status

# Vacuum all CTU databases
for db in Airline Credit Carcinogenesis employee financial geneea Hepatitis_std; do
    psql -d $db -c "VACUUM ANALYZE;"
done

# Backup database
pg_dump -d financial -Fc > financial.backup

# Restore database
pg_restore -d financial financial.backup
```

## 🐛 Troubleshooting

### Common Issues

```bash
# Connection refused
pg_ctl -D data status  # Check if running
tail -f data/log/*     # Check logs

# Extension not found
ls pgsql/lib/postgresql/ | grep pg_duckdb
cd pg_duckdb && make clean && make install

# Import errors
python3 import_ctu_datasets.py --force --databases financial

# Encoding issues
iconv -f LATIN1 -t UTF-8 input.csv > output.csv
```

## 📚 Documentation

- [PostgreSQL 17 Docs](https://www.postgresql.org/docs/17/)
- [pg_duckdb GitHub](https://github.com/duckdb/pg_duckdb)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [CTU Dataset Repository](https://relational.fel.cvut.cz/)
- [TPC Benchmarks](http://www.tpc.org/)

## 📄 License

- PostgreSQL: PostgreSQL License
- pg_duckdb: MIT License
- CTU Datasets: Research/Educational Use
- Scripts: MIT License

## 🤝 Contributing

1. Fork the repository
2. Add new datasets to `import_ctu_datasets.py`
3. Enhance query patterns in `generate_benchmark_queries.py`
4. Test with small datasets first
5. Submit pull request with documentation

## 📊 Statistics

- **Total Databases:** 9 (7 CTU + 2 TPC)
- **Total Tables:** 73+ 
- **Total Records:** ~8 million
- **Generated Queries:** 14,000+
- **Query Generation Speed:** ~186 queries/second
- **Supported Query Types:** AP (OLAP) and TP (OLTP)

---
*Last Updated: 2025-09-11*