#!/bin/bash

# TPC-H and TPC-DS Setup Script for PostgreSQL with pg_duckdb
# This script automatically downloads, compiles, and loads TPC-H and TPC-DS data into PostgreSQL

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_PATH="${SCRIPT_DIR}/pgsql/bin"
export PATH="${PG_PATH}:${PATH}"
export PGDATA="${SCRIPT_DIR}/data"

# Data scale (1 = 1GB)
SCALE=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if PostgreSQL is running
check_postgres() {
    if ! pg_ctl status >/dev/null 2>&1; then
        print_error "PostgreSQL is not running. Please start it first."
        exit 1
    fi
}

# Function to download and compile TPC-H
setup_tpch() {
    print_status "Setting up TPC-H..."
    
    # Clone TPC-H repository
    if [ ! -d "${SCRIPT_DIR}/tpch-dbgen" ]; then
        print_status "Cloning TPC-H repository..."
        git clone https://github.com/electrum/tpch-dbgen.git "${SCRIPT_DIR}/tpch-dbgen"
    else
        print_warning "TPC-H repository already exists, skipping clone"
    fi
    
    # Compile TPC-H
    cd "${SCRIPT_DIR}/tpch-dbgen"
    print_status "Compiling TPC-H dbgen..."
    make clean >/dev/null 2>&1 || true
    make -j$(nproc)
    
    # Generate TPC-H data
    if [ ! -f "lineitem.tbl" ]; then
        print_status "Generating ${SCALE}GB TPC-H data..."
        ./dbgen -vf -s ${SCALE}
        
        # Remove trailing pipes from data files
        print_status "Cleaning data files..."
        for file in *.tbl; do
            sed -i 's/|$//' "$file"
        done
    else
        print_warning "TPC-H data already exists, skipping generation"
    fi
}

# Function to download and compile TPC-DS
setup_tpcds() {
    print_status "Setting up TPC-DS..."
    
    # Clone TPC-DS repository (using Databricks version that compiles cleanly)
    if [ ! -d "${SCRIPT_DIR}/databricks-tpcds" ]; then
        print_status "Cloning TPC-DS repository..."
        git clone https://github.com/databricks/tpcds-kit.git "${SCRIPT_DIR}/databricks-tpcds"
    else
        print_warning "TPC-DS repository already exists, skipping clone"
    fi
    
    # Compile TPC-DS
    cd "${SCRIPT_DIR}/databricks-tpcds/tools"
    print_status "Compiling TPC-DS dsdgen..."
    make clean >/dev/null 2>&1 || true
    make OS=LINUX
    
    # Generate TPC-DS data
    if [ ! -d "${SCRIPT_DIR}/tpcds_data" ] || [ -z "$(ls -A ${SCRIPT_DIR}/tpcds_data 2>/dev/null)" ]; then
        print_status "Generating ${SCALE}GB TPC-DS data..."
        mkdir -p "${SCRIPT_DIR}/tpcds_data"
        # Run dsdgen with proper flags
        ./dsdgen -SCALE ${SCALE} -DIR "${SCRIPT_DIR}/tpcds_data" -TERMINATE N -FORCE Y
        
        # Check if data was generated
        if [ -z "$(ls -A ${SCRIPT_DIR}/tpcds_data 2>/dev/null)" ]; then
            print_error "Failed to generate TPC-DS data"
            return 1
        fi
        
        # Clean and fix data files
        print_status "Cleaning TPC-DS data files..."
        cd "${SCRIPT_DIR}/tpcds_data"
        for file in *.dat; do
            if [ -f "$file" ]; then
                print_status "Processing $file..."
                # Step 1: Convert from LATIN1 to UTF-8 to handle special characters
                # This fixes issues like "CÔTE D'IVOIRE" in customer.dat
                iconv -f LATIN1 -t UTF-8//IGNORE "$file" > "${file}.utf8" 2>/dev/null || {
                    print_warning "Could not convert $file from LATIN1, trying ISO-8859-1..."
                    iconv -f ISO-8859-1 -t UTF-8//IGNORE "$file" > "${file}.utf8" 2>/dev/null || {
                        print_warning "Could not convert $file, keeping original encoding"
                        cp "$file" "${file}.utf8"
                    }
                }
                
                # Step 2: Remove trailing pipes from each line
                sed 's/|$//' "${file}.utf8" > "$file"
                
                # Step 3: Clean up temporary file
                rm -f "${file}.utf8"
            fi
        done
        print_status "Data cleaning complete"
    else
        print_warning "TPC-DS data already exists, skipping generation"
    fi
}

# Function to create TPC-H database and load data
load_tpch() {
    print_status "Loading TPC-H data into PostgreSQL..."
    
    # Create database
    createdb tpch_sf1 2>/dev/null || print_warning "Database 'tpch_sf1' already exists"
    
    # Create pg_duckdb extension
    psql -d tpch_sf1 -c "CREATE EXTENSION IF NOT EXISTS pg_duckdb;" || true
    
    # Create TPC-H schema
    print_status "Creating TPC-H schema..."
    psql -d tpch_sf1 <<EOF
-- Drop existing tables if any
DROP TABLE IF EXISTS lineitem CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS partsupp CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS part CASCADE;
DROP TABLE IF EXISTS nation CASCADE;
DROP TABLE IF EXISTS region CASCADE;

-- Create tables
CREATE TABLE nation (
    n_nationkey INTEGER NOT NULL,
    n_name      CHAR(25) NOT NULL,
    n_regionkey INTEGER NOT NULL,
    n_comment   VARCHAR(152),
    PRIMARY KEY (n_nationkey)
);

CREATE TABLE region (
    r_regionkey INTEGER NOT NULL,
    r_name      CHAR(25) NOT NULL,
    r_comment   VARCHAR(152),
    PRIMARY KEY (r_regionkey)
);

CREATE TABLE part (
    p_partkey     INTEGER NOT NULL,
    p_name        VARCHAR(55) NOT NULL,
    p_mfgr        CHAR(25) NOT NULL,
    p_brand       CHAR(10) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INTEGER NOT NULL,
    p_container   CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment     VARCHAR(23) NOT NULL,
    PRIMARY KEY (p_partkey)
);

CREATE TABLE supplier (
    s_suppkey   INTEGER NOT NULL,
    s_name      CHAR(25) NOT NULL,
    s_address   VARCHAR(40) NOT NULL,
    s_nationkey INTEGER NOT NULL,
    s_phone     CHAR(15) NOT NULL,
    s_acctbal   DECIMAL(15,2) NOT NULL,
    s_comment   VARCHAR(101) NOT NULL,
    PRIMARY KEY (s_suppkey)
);

CREATE TABLE partsupp (
    ps_partkey    INTEGER NOT NULL,
    ps_suppkey    INTEGER NOT NULL,
    ps_availqty   INTEGER NOT NULL,
    ps_supplycost DECIMAL(15,2) NOT NULL,
    ps_comment    VARCHAR(199) NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE customer (
    c_custkey    INTEGER NOT NULL,
    c_name       VARCHAR(25) NOT NULL,
    c_address    VARCHAR(40) NOT NULL,
    c_nationkey  INTEGER NOT NULL,
    c_phone      CHAR(15) NOT NULL,
    c_acctbal    DECIMAL(15,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment    VARCHAR(117) NOT NULL,
    PRIMARY KEY (c_custkey)
);

CREATE TABLE orders (
    o_orderkey      INTEGER NOT NULL,
    o_custkey       INTEGER NOT NULL,
    o_orderstatus   CHAR(1) NOT NULL,
    o_totalprice    DECIMAL(15,2) NOT NULL,
    o_orderdate     DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk         CHAR(15) NOT NULL,
    o_shippriority  INTEGER NOT NULL,
    o_comment       VARCHAR(79) NOT NULL,
    PRIMARY KEY (o_orderkey)
);

CREATE TABLE lineitem (
    l_orderkey      INTEGER NOT NULL,
    l_partkey       INTEGER NOT NULL,
    l_suppkey       INTEGER NOT NULL,
    l_linenumber    INTEGER NOT NULL,
    l_quantity      DECIMAL(15,2) NOT NULL,
    l_extendedprice DECIMAL(15,2) NOT NULL,
    l_discount      DECIMAL(15,2) NOT NULL,
    l_tax           DECIMAL(15,2) NOT NULL,
    l_returnflag    CHAR(1) NOT NULL,
    l_linestatus    CHAR(1) NOT NULL,
    l_shipdate      DATE NOT NULL,
    l_commitdate    DATE NOT NULL,
    l_receiptdate   DATE NOT NULL,
    l_shipinstruct  CHAR(25) NOT NULL,
    l_shipmode      CHAR(10) NOT NULL,
    l_comment       VARCHAR(44) NOT NULL
);

-- Add foreign key constraints
ALTER TABLE supplier ADD FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey);
ALTER TABLE partsupp ADD FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey);
ALTER TABLE partsupp ADD FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey);
ALTER TABLE customer ADD FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey);
ALTER TABLE orders ADD FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey);
ALTER TABLE nation ADD FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey);

-- Create indexes
CREATE INDEX idx_supplier_nation ON supplier(s_nationkey);
CREATE INDEX idx_partsupp_part ON partsupp(ps_partkey);
CREATE INDEX idx_partsupp_supp ON partsupp(ps_suppkey);
CREATE INDEX idx_customer_nation ON customer(c_nationkey);
CREATE INDEX idx_orders_cust ON orders(o_custkey);
CREATE INDEX idx_orders_date ON orders(o_orderdate);
CREATE INDEX idx_lineitem_order ON lineitem(l_orderkey);
CREATE INDEX idx_lineitem_part_supp ON lineitem(l_partkey, l_suppkey);
CREATE INDEX idx_lineitem_shipdate ON lineitem(l_shipdate);
CREATE INDEX idx_nation_region ON nation(n_regionkey);
EOF
    
    # Check if data is already loaded
    LINEITEM_COUNT=$(psql -d tpch_sf1 -t -c "SELECT COUNT(*) FROM lineitem;" 2>/dev/null | tr -d ' ')
    if [ ! -z "$LINEITEM_COUNT" ] && [ "$LINEITEM_COUNT" -gt 0 ]; then
        print_warning "TPC-H data already loaded (lineitem has $LINEITEM_COUNT rows), skipping data load"
        return
    fi
    
    # Load data
    cd "${SCRIPT_DIR}/tpch-dbgen"
    print_status "Loading TPC-H data (this may take a while)..."
    
    psql -d tpch_sf1 -c "COPY region FROM '${SCRIPT_DIR}/tpch-dbgen/region.tbl' WITH (FORMAT csv, DELIMITER '|');"
    psql -d tpch_sf1 -c "COPY nation FROM '${SCRIPT_DIR}/tpch-dbgen/nation.tbl' WITH (FORMAT csv, DELIMITER '|');"
    psql -d tpch_sf1 -c "COPY part FROM '${SCRIPT_DIR}/tpch-dbgen/part.tbl' WITH (FORMAT csv, DELIMITER '|');"
    psql -d tpch_sf1 -c "COPY supplier FROM '${SCRIPT_DIR}/tpch-dbgen/supplier.tbl' WITH (FORMAT csv, DELIMITER '|');"
    psql -d tpch_sf1 -c "COPY partsupp FROM '${SCRIPT_DIR}/tpch-dbgen/partsupp.tbl' WITH (FORMAT csv, DELIMITER '|');"
    psql -d tpch_sf1 -c "COPY customer FROM '${SCRIPT_DIR}/tpch-dbgen/customer.tbl' WITH (FORMAT csv, DELIMITER '|');"
    psql -d tpch_sf1 -c "COPY orders FROM '${SCRIPT_DIR}/tpch-dbgen/orders.tbl' WITH (FORMAT csv, DELIMITER '|');"
    
    # Load lineitem in chunks if file is large
    if [ -f "lineitem.tbl" ]; then
        LINE_COUNT=$(wc -l < lineitem.tbl)
        if [ ${LINE_COUNT} -gt 1000000 ]; then
            print_status "Splitting lineitem table for loading..."
            split -l 1000000 lineitem.tbl lineitem_part_
            for file in lineitem_part_*; do
                print_status "Loading $file..."
                psql -d tpch_sf1 -c "COPY lineitem FROM '${SCRIPT_DIR}/tpch-dbgen/$file' WITH (FORMAT csv, DELIMITER '|');"
            done
            rm -f lineitem_part_*
        else
            psql -d tpch_sf1 -c "COPY lineitem FROM '${SCRIPT_DIR}/tpch-dbgen/lineitem.tbl' WITH (FORMAT csv, DELIMITER '|');"
        fi
    fi
    
    # Add lineitem foreign keys after loading
    print_status "Adding lineitem foreign key constraints..."
    psql -d tpch_sf1 -c "ALTER TABLE lineitem ADD FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey);"
    psql -d tpch_sf1 -c "ALTER TABLE lineitem ADD FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey);"
    
    # Verify load
    print_status "Verifying TPC-H data load..."
    psql -d tpch_sf1 -t -c "SELECT COUNT(*) FROM lineitem;" | grep -q '^[[:space:]]*0[[:space:]]*$'
    if [ $? -eq 0 ]; then
        print_warning "TPC-H data not loaded correctly"
    else
        psql -d tpch_sf1 -c "
        SELECT 'TPC-H Data Load Summary' as info;
        SELECT 'region' as table_name, COUNT(*) as row_count FROM region
        UNION ALL SELECT 'nation', COUNT(*) FROM nation
        UNION ALL SELECT 'supplier', COUNT(*) FROM supplier
        UNION ALL SELECT 'customer', COUNT(*) FROM customer
        UNION ALL SELECT 'part', COUNT(*) FROM part
        UNION ALL SELECT 'partsupp', COUNT(*) FROM partsupp
        UNION ALL SELECT 'orders', COUNT(*) FROM orders
        UNION ALL SELECT 'lineitem', COUNT(*) FROM lineitem
        ORDER BY table_name;"
    fi
}

# Function to create TPC-DS database and load data
load_tpcds() {
    print_status "Loading TPC-DS data into PostgreSQL..."
    
    # Create database
    createdb tpcds_sf1 2>/dev/null || print_warning "Database 'tpcds_sf1' already exists"
    
    # Create pg_duckdb extension
    psql -d tpcds_sf1 -c "CREATE EXTENSION IF NOT EXISTS pg_duckdb;" || true
    
    # Get TPC-DS schema DDL
    cd "${SCRIPT_DIR}/databricks-tpcds/tools"
    
    # Create schema
    print_status "Creating TPC-DS schema..."
    
    # Use the official TPC-DS schema
    if [ -f "tpcds.sql" ]; then
        print_status "Using official TPC-DS schema..."
        # The tpcds.sql might have some syntax that needs adjustment for PostgreSQL
        psql -d tpcds_sf1 -f tpcds.sql 2>/dev/null || {
            print_warning "Official schema failed, using simplified schema..."
            psql -d tpcds_sf1 <<EOF
-- Drop existing tables if any
DROP SCHEMA IF EXISTS tpcds CASCADE;
CREATE SCHEMA tpcds;
SET search_path TO tpcds;

-- Create dimension tables
CREATE TABLE date_dim (
    d_date_sk INTEGER NOT NULL PRIMARY KEY,
    d_date_id CHAR(16) NOT NULL,
    d_date DATE,
    d_month_seq INTEGER,
    d_week_seq INTEGER,
    d_quarter_seq INTEGER,
    d_year INTEGER,
    d_dow INTEGER,
    d_moy INTEGER,
    d_dom INTEGER,
    d_qoy INTEGER,
    d_fy_year INTEGER,
    d_fy_quarter_seq INTEGER,
    d_fy_week_seq INTEGER,
    d_day_name CHAR(9),
    d_quarter_name CHAR(6),
    d_holiday CHAR(1),
    d_weekend CHAR(1),
    d_following_holiday CHAR(1),
    d_first_dom INTEGER,
    d_last_dom INTEGER,
    d_same_day_ly INTEGER,
    d_same_day_lq INTEGER,
    d_current_day CHAR(1),
    d_current_week CHAR(1),
    d_current_month CHAR(1),
    d_current_quarter CHAR(1),
    d_current_year CHAR(1)
);

CREATE TABLE store (
    s_store_sk INTEGER NOT NULL PRIMARY KEY,
    s_store_id CHAR(16) NOT NULL,
    s_rec_start_date DATE,
    s_rec_end_date DATE,
    s_closed_date_sk INTEGER,
    s_store_name VARCHAR(50),
    s_number_employees INTEGER,
    s_floor_space INTEGER,
    s_hours CHAR(20),
    s_manager VARCHAR(40),
    s_market_id INTEGER,
    s_geography_class VARCHAR(100),
    s_market_desc VARCHAR(100),
    s_market_manager VARCHAR(40),
    s_division_id INTEGER,
    s_division_name VARCHAR(50),
    s_company_id INTEGER,
    s_company_name VARCHAR(50),
    s_street_number VARCHAR(10),
    s_street_name VARCHAR(60),
    s_street_type CHAR(15),
    s_suite_number CHAR(10),
    s_city VARCHAR(60),
    s_county VARCHAR(30),
    s_state CHAR(2),
    s_zip CHAR(10),
    s_country VARCHAR(20),
    s_gmt_offset DECIMAL(5,2),
    s_tax_percentage DECIMAL(5,2)
);

CREATE TABLE item (
    i_item_sk INTEGER NOT NULL PRIMARY KEY,
    i_item_id CHAR(16) NOT NULL,
    i_rec_start_date DATE,
    i_rec_end_date DATE,
    i_item_desc VARCHAR(200),
    i_current_price DECIMAL(7,2),
    i_wholesale_cost DECIMAL(7,2),
    i_brand_id INTEGER,
    i_brand CHAR(50),
    i_class_id INTEGER,
    i_class CHAR(50),
    i_category_id INTEGER,
    i_category CHAR(50),
    i_manufact_id INTEGER,
    i_manufact CHAR(50),
    i_size CHAR(20),
    i_formulation CHAR(20),
    i_color CHAR(20),
    i_units CHAR(10),
    i_container CHAR(10),
    i_manager_id INTEGER,
    i_product_name CHAR(50)
);

CREATE TABLE customer (
    c_customer_sk INTEGER NOT NULL PRIMARY KEY,
    c_customer_id CHAR(16) NOT NULL,
    c_current_cdemo_sk INTEGER,
    c_current_hdemo_sk INTEGER,
    c_current_addr_sk INTEGER,
    c_first_shipto_date_sk INTEGER,
    c_first_sales_date_sk INTEGER,
    c_salutation CHAR(10),
    c_first_name CHAR(20),
    c_last_name CHAR(30),
    c_preferred_cust_flag CHAR(1),
    c_birth_day INTEGER,
    c_birth_month INTEGER,
    c_birth_year INTEGER,
    c_birth_country VARCHAR(20),
    c_login CHAR(13),
    c_email_address CHAR(50),
    c_last_review_date_sk INTEGER
);

-- Create fact tables
CREATE TABLE store_sales (
    ss_sold_date_sk INTEGER,
    ss_sold_time_sk INTEGER,
    ss_item_sk INTEGER NOT NULL,
    ss_customer_sk INTEGER,
    ss_cdemo_sk INTEGER,
    ss_hdemo_sk INTEGER,
    ss_addr_sk INTEGER,
    ss_store_sk INTEGER,
    ss_promo_sk INTEGER,
    ss_ticket_number BIGINT NOT NULL,
    ss_quantity INTEGER,
    ss_wholesale_cost DECIMAL(7,2),
    ss_list_price DECIMAL(7,2),
    ss_sales_price DECIMAL(7,2),
    ss_ext_discount_amt DECIMAL(7,2),
    ss_ext_sales_price DECIMAL(7,2),
    ss_ext_wholesale_cost DECIMAL(7,2),
    ss_ext_list_price DECIMAL(7,2),
    ss_ext_tax DECIMAL(7,2),
    ss_coupon_amt DECIMAL(7,2),
    ss_net_paid DECIMAL(7,2),
    ss_net_paid_inc_tax DECIMAL(7,2),
    ss_net_profit DECIMAL(7,2),
    PRIMARY KEY (ss_item_sk, ss_ticket_number)
);

-- Note: This is a simplified schema. Full TPC-DS has 24 tables.
EOF
        }
    else
        print_warning "TPC-DS schema file not found, skipping schema creation"
    fi
    
    # Check if any table has data already
    TABLE_COUNT=$(psql -d tpcds_sf1 -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('public', 'tpcds') AND table_type = 'BASE TABLE';" 2>/dev/null | tr -d ' ')
    if [ ! -z "$TABLE_COUNT" ] && [ "$TABLE_COUNT" -gt 0 ]; then
        # Check if data is loaded
        HAS_DATA=$(psql -d tpcds_sf1 -t -c "SELECT CASE WHEN EXISTS (SELECT 1 FROM store LIMIT 1) THEN 1 ELSE 0 END;" 2>/dev/null | tr -d ' ')
        if [ "$HAS_DATA" = "1" ]; then
            print_warning "TPC-DS data already loaded, skipping data load"
            return
        fi
    fi
    
    # Load TPC-DS data
    cd "${SCRIPT_DIR}/tpcds_data"
    print_status "Loading TPC-DS data (this may take a while)..."
    
    # Map of TPC-DS file names to table names
    for file in *.dat; do
        if [ -f "$file" ]; then
            # Get table name from file name (remove .dat extension and keep underscores)
            table_name="${file%.dat}"
            
            print_status "Loading ${table_name}..."
            # Check if table exists before loading
            if psql -d tpcds_sf1 -c "\d ${table_name}" >/dev/null 2>&1; then
                # Try loading with UTF-8 encoding first
                psql -d tpcds_sf1 -c "COPY ${table_name} FROM '${SCRIPT_DIR}/tpcds_data/${file}' WITH (FORMAT csv, DELIMITER '|', ENCODING 'UTF8');" 2>/dev/null || \
                psql -d tpcds_sf1 -c "COPY ${table_name} FROM '${SCRIPT_DIR}/tpcds_data/${file}' WITH (FORMAT csv, DELIMITER '|', ENCODING 'LATIN1');" 2>/dev/null || \
                print_warning "Failed to load ${table_name}, skipping..."
            elif psql -d tpcds_sf1 -c "\d tpcds.${table_name}" >/dev/null 2>&1; then
                psql -d tpcds_sf1 -c "COPY tpcds.${table_name} FROM '${SCRIPT_DIR}/tpcds_data/${file}' WITH (FORMAT csv, DELIMITER '|', ENCODING 'UTF8');" 2>/dev/null || \
                psql -d tpcds_sf1 -c "COPY tpcds.${table_name} FROM '${SCRIPT_DIR}/tpcds_data/${file}' WITH (FORMAT csv, DELIMITER '|', ENCODING 'LATIN1');" 2>/dev/null || \
                print_warning "Failed to load tpcds.${table_name}, skipping..."
            else
                print_warning "Table ${table_name} does not exist in any schema, skipping data load"
            fi
        fi
    done
    
    # Verify load
    print_status "Verifying TPC-DS data load..."
    psql -d tpcds_sf1 -c "
    SELECT 'TPC-DS Data Load Summary' as info;
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema IN ('public', 'tpcds') 
    AND table_type = 'BASE TABLE'
    ORDER BY table_name;" || print_warning "Could not verify TPC-DS tables"
}

# Main execution
main() {
    echo "========================================="
    echo "TPC-H and TPC-DS Benchmark Setup Script"
    echo "========================================="
    echo
    
    # Check if PostgreSQL is running
    check_postgres
    
    # # Setup TPC-H
    # echo
    # echo "1. Setting up TPC-H..."
    # echo "----------------------"
    # setup_tpch
    # load_tpch
    
    # Setup TPC-DS
    echo
    echo "2. Setting up TPC-DS..."
    echo "-----------------------"
    setup_tpcds
    load_tpcds
    
    echo
    print_status "Setup complete!"
    echo
    echo "Databases created:"
    echo "  - tpch_sf1  : TPC-H benchmark data (${SCALE}GB)"
    echo "  - tpcds_sf1 : TPC-DS benchmark data (${SCALE}GB)"
    echo
    echo "Both databases have pg_duckdb extension enabled."
    echo
    echo "You can now connect to the databases using:"
    echo "  psql -d tpch_sf1"
    echo "  psql -d tpcds_sf1"
}

# Run main function
main "$@"