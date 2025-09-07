# DuckDB postgres_scanner Extension Installation Guide

## Installation Methods

### Method 1: Direct Installation from DuckDB (Easiest)
Once DuckDB is running, you can install postgres_scanner directly:

```sql
-- Install the extension (only needs to be done once)
INSTALL postgres;

-- Load the extension for use
LOAD postgres;
```

**Note**: This will download the extension from DuckDB's official extension repository.

### Method 2: Building from Source (For Development)

#### Prerequisites
1. **PostgreSQL Development Headers**: Required for libpq
   - We have these from our PostgreSQL 16.4 build at `~/DB/duckdb/postgresql`
   - pg_config available at: `~/DB/duckdb/postgresql/bin/pg_config`

2. **OpenSSL Development Libraries**: Required for secure connections
   ```bash
   sudo apt-get install libssl-dev
   ```

3. **libpq Library**: Provided by our PostgreSQL build
   - Headers: `~/DB/duckdb/postgresql/include`
   - Library: `~/DB/duckdb/postgresql/lib`

#### Build Steps
```bash
# Clone the repository
cd ~/DB/duckdb
git clone https://github.com/duckdb/postgres_scanner.git

# Initialize submodules
cd postgres_scanner
git submodule init
git pull --recurse-submodules

# Build with proper paths
export PostgreSQL_ROOT=~/DB/duckdb/postgresql
export OPENSSL_ROOT_DIR=/usr
make

# Or specify DuckDB directory if using custom build
make DUCKDB_DIR=../duckdb_src
```

#### Common Build Issues and Solutions

1. **OpenSSL not found**: 
   ```bash
   export OPENSSL_ROOT_DIR=/usr
   ```

2. **PostgreSQL headers not found**:
   ```bash
   export PostgreSQL_ROOT=~/DB/duckdb/postgresql
   export PKG_CONFIG_PATH=$PostgreSQL_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

3. **CMake cache issues**: 
   ```bash
   rm -rf build
   make clean
   make
   ```

## Usage Methods

### Method 1: ATTACH Command (Recommended)
```sql
-- Attach a PostgreSQL database
ATTACH 'host=localhost port=5433 dbname=postgres user=wuy' AS pg_db (TYPE postgres);

-- Query PostgreSQL tables through DuckDB
SELECT * FROM pg_db.schema_name.table_name;
```

### Method 2: postgres_scan Function
```sql
-- Direct scan of a PostgreSQL table
SELECT * FROM postgres_scan(
    'host=localhost port=5433 dbname=postgres user=wuy',
    'public',
    'table_name'
);
```

### Method 3: Using Secrets for Authentication
```sql
-- Create a secret for PostgreSQL connection
CREATE SECRET pg_secret (
    TYPE postgres,
    HOST '127.0.0.1',
    PORT 5433,
    DATABASE postgres,
    USER 'wuy',
    PASSWORD ''
);

-- Use the secret in ATTACH
ATTACH '' AS pg_db (TYPE postgres, SECRET pg_secret);
```

## Connection String Parameters
Common libpq connection parameters:
- `host`: PostgreSQL server hostname (default: localhost)
- `port`: PostgreSQL server port (default: 5432, ours: 5433)
- `dbname`: Database name
- `user`: PostgreSQL username
- `password`: PostgreSQL password
- `sslmode`: SSL connection mode (disable/require/verify-ca/verify-full)

## Environment Variables
You can also use PostgreSQL environment variables:
```bash
export PGHOST=localhost
export PGPORT=5433
export PGDATABASE=postgres
export PGUSER=wuy
```

## Testing Connection
```sql
-- After installation and loading
ATTACH 'host=localhost port=5433 dbname=postgres' AS test_pg (TYPE postgres);
SHOW ALL TABLES FROM test_pg;
```

## Important Notes
1. postgres_scanner reads PostgreSQL data into DuckDB for processing
2. All analytical processing happens in DuckDB's columnar engine
3. Supports projection and filter pushdown to PostgreSQL
4. Uses binary COPY protocol for efficient data transfer
5. Does NOT route queries - only reads PostgreSQL tables

## For Our Project
Since postgres_scanner doesn't provide query routing, we need to:
1. Use it for testing both engines with same data
2. Implement our own routing logic in DuckDB's optimizer
3. Create a separate PostgreSQL executor for routed queries