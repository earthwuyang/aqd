#!/usr/bin/env python3
"""
Import benchmark datasets with proper data types.

Enhancements:
- First export MySQL tables to cached CSV files under `data/benchmark_data/<dataset>/`.
- Skip re-downloading CSVs and re-importing if datasets already exist in PostgreSQL and DuckDB.
- Use `--force` to redownload CSVs and drop/reimport datasets in both engines.
- Careful MySQL→PostgreSQL type mapping (incl. unsigned, enums, bit, json).
"""

import psycopg2
import mysql.connector
import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import subprocess
import argparse

# Connection configurations
MYSQL_CONFIG = {
    'host': 'relational.fel.cvut.cz',
    'port': 3306,
    'user': 'guest',
    'password': 'ctu-relational'
}

POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'wuy'
}

BASE_DIR = Path(__file__).resolve().parent

DUCKDB_CONFIG = {
    'binary': str(BASE_DIR / 'duckdb' / 'duckdb'),
    'databases_dir': str(BASE_DIR / 'data' / 'duckdb_databases'),
    'central_db': str(BASE_DIR / 'data' / 'benchmark_datasets.db')
}

# Test with a few datasets first
SELECTED_DATABASES = [
    'financial',      # Financial transactions
    'imdb_small',    # IMDB movie data
    'Basketball_men', # Basketball statistics
    'northwind',      # Classic business data
    'sakila',        # Movie rental data
    'world',         # World cities/countries
    'tpch_sf1',      # TPC-H benchmark
    'tpcds_sf1',     # TPC-DS benchmark
]

DATA_DIR = str(BASE_DIR / 'data' / 'benchmark_data')

class BenchmarkDatasetImporter:
    def __init__(self, force: bool = False):
        self.mysql_conn = None
        self.pg_conn = None
        self.stats = {
            'databases_imported': 0,
            'tables_imported': 0,
            'total_rows': 0,
            'start_time': datetime.now()
        }
        self.imported_datasets = []
        self.force = force

    def connect(self):
        """Connect to all databases"""
        print("🔌 Connecting to databases...")
        
        # Connect to MySQL
        print("  Connecting to CTU MySQL server...")
        self.mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        print("  ✓ Connected to CTU MySQL server")
        
        # Connect to PostgreSQL
        print("  Connecting to PostgreSQL...")
        admin_config = POSTGRESQL_CONFIG.copy()
        admin_config['database'] = 'postgres'
        self.pg_conn = psycopg2.connect(**admin_config)
        self.pg_conn.autocommit = True
        print("  ✓ Connected to PostgreSQL")
        
        # Setup DuckDB
        os.makedirs(DUCKDB_CONFIG['databases_dir'], exist_ok=True)
        print("  ✓ DuckDB directory ready")
        # Ensure central DuckDB file exists
        try:
            open(DUCKDB_CONFIG['central_db'], 'ab').close()
            print("  ✓ Central DuckDB database ready")
        except Exception as e:
            print(f"  ⚠ Could not precreate central DuckDB DB: {e}")
        
        print("✅ All database connections established")
    
    def get_mysql_column_types(self, database, table):
        """Get column types from MySQL"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{database}' AND TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = {}
        for row in cursor.fetchall():
            col_name, data_type, col_type, nullable = row
            columns[col_name] = {
                'mysql_type': data_type.upper(),
                'full_type': col_type,
                'nullable': nullable == 'YES'
            }
        
        cursor.close()
        return columns
    
    def map_mysql_to_pg_type(self, mysql_type_info):
        """Map MySQL types to PostgreSQL types"""
        mysql_type = mysql_type_info['mysql_type']
        full_type = mysql_type_info['full_type'].upper()
        is_unsigned = 'UNSIGNED' in full_type
        
        # Integer types
        if mysql_type in ['TINYINT', 'SMALLINT']:
            # Treat TINYINT as SMALLINT; if (1) and not unsigned handle separately in boolean
            return 'SMALLINT'
        elif mysql_type in ['MEDIUMINT', 'INT', 'INTEGER']:
            return 'BIGINT' if is_unsigned else 'INTEGER'
        elif mysql_type == 'BIGINT':
            return 'NUMERIC(20)' if is_unsigned else 'BIGINT'
        
        # Decimal/Float types
        elif mysql_type in ['FLOAT', 'REAL']:
            return 'REAL'
        elif mysql_type in ['DOUBLE', 'DOUBLE PRECISION']:
            return 'DOUBLE PRECISION'
        elif mysql_type in ['DECIMAL', 'NUMERIC', 'DEC']:
            # Extract precision if available
            if '(' in full_type:
                return full_type.replace('DECIMAL', 'NUMERIC').replace('DEC', 'NUMERIC')
            return 'NUMERIC'
        
        # Date/Time types
        elif mysql_type == 'DATE':
            return 'DATE'
        elif mysql_type == 'TIME':
            return 'TIME'
        elif mysql_type in ['DATETIME', 'TIMESTAMP']:
            return 'TIMESTAMP'
        elif mysql_type == 'YEAR':
            return 'SMALLINT'
        
        # String types
        elif mysql_type in ['CHAR', 'VARCHAR']:
            if '(' in full_type:
                # Preserve length
                return full_type
            return 'VARCHAR(255)'
        elif mysql_type in ['TINYTEXT', 'TEXT', 'MEDIUMTEXT', 'LONGTEXT', 'ENUM', 'SET']:
            return 'TEXT'
        
        # Binary types
        elif mysql_type in ['BINARY', 'VARBINARY', 'BLOB', 'TINYBLOB', 'MEDIUMBLOB', 'LONGBLOB']:
            return 'BYTEA'
        
        # Boolean
        elif mysql_type == 'BOOLEAN' or (mysql_type == 'TINYINT' and '(1)' in full_type):
            return 'BOOLEAN'

        # Bit strings
        elif mysql_type == 'BIT':
            if '(1)' in full_type:
                return 'BOOLEAN'
            # Map to BIT VARYING for portability
            return 'BIT VARYING'
        
        # JSON
        elif mysql_type == 'JSON':
            return 'JSONB'
        
        # Default
        else:
            return 'TEXT'
    
    def export_to_csv_typed(self, database, table, output_path):
        """Export table data to CSV preserving types"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        
        # Get column info
        column_types = self.get_mysql_column_types(database, table)
        columns = list(column_types.keys())
        
        # Export data
        cursor.execute(f"SELECT * FROM `{table}`")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)  # Header
            
            batch_size = 10000
            row_count = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                for row in rows:
                    processed_row = []
                    for i, val in enumerate(row):
                        if val is None:
                            processed_row.append('')
                        elif isinstance(val, bytes):
                            processed_row.append(val.decode('utf-8', errors='ignore'))
                        elif isinstance(val, (datetime, )):
                            processed_row.append(val.isoformat())
                        else:
                            # Keep original value, don't convert to string unnecessarily
                            processed_row.append(val)
                    writer.writerow(processed_row)
                    row_count += 1
        
        cursor.close()
        return row_count, columns, column_types

    def get_mysql_relationships(self, database):
        """Fetch foreign key relationships from MySQL INFORMATION_SCHEMA and group multi-column keys"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        sql = (
            "SELECT CONSTRAINT_NAME, TABLE_NAME AS child_table, COLUMN_NAME AS child_column, "
            "REFERENCED_TABLE_NAME AS parent_table, REFERENCED_COLUMN_NAME AS parent_column, ORDINAL_POSITION "
            "FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE "
            f"WHERE TABLE_SCHEMA = '{database}' AND REFERENCED_TABLE_NAME IS NOT NULL "
            "ORDER BY CONSTRAINT_NAME, ORDINAL_POSITION"
        )
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()

        rels = {}
        for constraint_name, child_table, child_col, parent_table, parent_col, ord_pos in rows:
            key = (constraint_name, child_table, parent_table)
            if key not in rels:
                rels[key] = {
                    'constraint_name': constraint_name,
                    'child_table': child_table,
                    'child_columns': [],
                    'parent_table': parent_table,
                    'parent_columns': []
                }
            rels[key]['child_columns'].append(child_col)
            rels[key]['parent_columns'].append(parent_col)

        # Convert to list
        relationships = list(rels.values())
        return relationships

    def get_mysql_primary_keys(self, database):
        """Fetch primary keys (possibly multi-column) from MySQL"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        sql = (
            "SELECT TABLE_NAME, CONSTRAINT_NAME, COLUMN_NAME, ORDINAL_POSITION "
            "FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE "
            f"WHERE TABLE_SCHEMA = '{database}' AND CONSTRAINT_NAME = 'PRIMARY' "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()

        pks = {}
        for table_name, constraint_name, column_name, ord_pos in rows:
            pks.setdefault(table_name, []).append(column_name)
        return pks

    def create_postgresql_primary_keys(self, database, primary_keys, existing_tables):
        """Create primary keys in PostgreSQL using MySQL PK metadata"""
        if not primary_keys:
            return True
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cursor = db_conn.cursor()
            for table, cols in primary_keys.items():
                if table not in existing_tables or not cols:
                    continue
                cols_quoted = ', '.join([f'"{c}"' for c in cols])
                constraint_name = f'{table}_pkey'
                sql = f'ALTER TABLE "{table}" ADD CONSTRAINT "{constraint_name}" PRIMARY KEY ({cols_quoted});'
                try:
                    cursor.execute(sql)
                    print(f"    ✓ Added PRIMARY KEY on {database}.{table} ({', '.join(cols)})")
                except Exception as e:
                    print(f"    ⚠ Skipping PK for {database}.{table}: {e}")
            return True
        except Exception as e:
            print(f"    ✗ Failed to add primary keys in {database}: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'db_conn' in locals():
                db_conn.close()

    def create_postgresql_foreign_keys(self, database, relationships, existing_tables):
        """Create foreign keys in PostgreSQL using MySQL FK metadata. Uses NOT VALID to avoid heavy validation."""
        if not relationships:
            return True
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cursor = db_conn.cursor()
            fk_count = 0
            for idx, rel in enumerate(relationships):
                child = rel['child_table']
                parent = rel['parent_table']
                child_cols = rel['child_columns']
                parent_cols = rel['parent_columns']
                if child not in existing_tables or parent not in existing_tables:
                    continue
                if not child_cols or not parent_cols or len(child_cols) != len(parent_cols):
                    continue
                child_cols_quoted = ', '.join([f'"{c}"' for c in child_cols])
                parent_cols_quoted = ', '.join([f'"{c}"' for c in parent_cols])
                cname = rel.get('constraint_name') or f'fk_{child}_{parent}_{idx}'
                cname = cname.replace('"', '')
                sql = f'ALTER TABLE "{child}" ADD CONSTRAINT "{cname}" FOREIGN KEY ({child_cols_quoted}) REFERENCES "{parent}" ({parent_cols_quoted}) NOT VALID;'
                try:
                    cursor.execute(sql)
                    fk_count += 1
                except Exception as e:
                    print(f"    ⚠ Skipping FK {child}->{parent} on {database}: {e}")
            if fk_count:
                print(f"    ✓ Added {fk_count} foreign keys in {database} (NOT VALID)")
            return True
        except Exception as e:
            print(f"    ✗ Failed to add foreign keys in {database}: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'db_conn' in locals():
                db_conn.close()

    def save_relationships_metadata(self, database, relationships):
        """Write relationships to JSON under data/benchmark_data/<database>/relationships.json"""
        out_dir = os.path.join(DATA_DIR, database)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'relationships.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'database': database, 'relationships': relationships}, f, indent=2)
        print(f"  ✓ Saved relationships metadata: {path}")

    def write_relationships_to_central_duckdb(self, database, relationships):
        """Store relationships in central DuckDB as a metadata table per schema for convenience"""
        if not relationships:
            return True
        db_file = DUCKDB_CONFIG['central_db']
        # Create schema if not exists and relationships table
        create_sql = f'''
            CREATE SCHEMA IF NOT EXISTS "{database}";
            CREATE TABLE IF NOT EXISTS "{database}"."__relationships" (
                constraint_name VARCHAR,
                child_table VARCHAR,
                child_columns JSON,
                parent_table VARCHAR,
                parent_columns JSON
            );
            DELETE FROM "{database}"."__relationships";
        '''
        result = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', create_sql], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ✗ Failed to prepare relationships table in central DuckDB: {result.stderr}")
            return False
        # Build INSERT statements
        values = []
        for r in relationships:
            cn = r['constraint_name'].replace("'", "''") if r.get('constraint_name') else ''
            ct = r['child_table'].replace("'", "''")
            pt = r['parent_table'].replace("'", "''")
            cc_json = json.dumps(r['child_columns'])
            pc_json = json.dumps(r['parent_columns'])
            values.append(f"('{cn}','{ct}', '{cc_json}', '{pt}', '{pc_json}')")
        if values:
            insert_sql = f"INSERT INTO \"{database}\".\"__relationships\" VALUES " + ",".join(values) + ";"
            ins_res = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', insert_sql], capture_output=True, text=True)
            if ins_res.returncode != 0:
                print(f"    ✗ Failed to insert relationships in central DuckDB: {ins_res.stderr}")
                return False
        print(f"    ✓ Wrote {len(relationships)} relationships to central DuckDB metadata table")
        return True
    
    def create_postgresql_database(self, database):
        """Create PostgreSQL database"""
        cursor = self.pg_conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        if cursor.fetchone():
            print(f"    Database '{database}' already exists")
            cursor.close()
            return True
        
        try:
            cursor.execute(f'CREATE DATABASE "{database}"')
            print(f"    ✓ Created database: {database}")
            cursor.close()
            return True
        except Exception as e:
            print(f"    ✗ Failed to create database {database}: {e}")
            cursor.close()
            return False

    def drop_postgresql_database(self, database):
        """Drop PostgreSQL database if exists"""
        cursor = self.pg_conn.cursor()
        # Terminate connections and drop
        try:
            cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (database,))
            cursor.execute(f'DROP DATABASE IF EXISTS "{database}"')
            print(f"    ✓ Dropped PostgreSQL database: {database}")
            return True
        except Exception as e:
            print(f"    ✗ Failed to drop PostgreSQL database {database}: {e}")
            return False
        finally:
            cursor.close()
    
    def create_postgresql_table_typed(self, database, table, column_types, csv_path):
        """Create table in PostgreSQL with proper types"""
        # Ensure database exists
        if not self.create_postgresql_database(database):
            return False
        
        # Connect to specific database
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cursor = db_conn.cursor()
            
            # Create table with proper types
            column_defs = []
            for col_name, type_info in column_types.items():
                pg_type = self.map_mysql_to_pg_type(type_info)
                null_clause = '' if type_info['nullable'] else ' NOT NULL'
                column_defs.append(f'"{col_name}" {pg_type}{null_clause}')
            
            create_sql = f'''
                DROP TABLE IF EXISTS "{table}" CASCADE;
                CREATE TABLE "{table}" (
                    {', '.join(column_defs)}
                );
            '''
            cursor.execute(create_sql)
            
            # Load data
            copy_sql = f'''
                COPY "{table}" FROM '{csv_path}'
                WITH (FORMAT CSV, HEADER TRUE, NULL '', ENCODING 'UTF8');
            '''
            cursor.execute(copy_sql)
            
            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            row_count = cursor.fetchone()[0]
            
            print(f"    ✓ PostgreSQL: {database}.{table} ({row_count} rows)")
            return True
            
        except Exception as e:
            print(f"    ✗ PostgreSQL failed for {database}.{table}: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'db_conn' in locals():
                db_conn.close()
    
    def create_duckdb_table(self, database, table, csv_path):
        """Create table in DuckDB (auto-detects types)"""
        db_file = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        
        # DuckDB is excellent at type detection
        load_sql = f'''
            DROP TABLE IF EXISTS "{table}";
            CREATE TABLE "{table}" AS 
            SELECT * FROM read_csv_auto('{csv_path}', header=true, nullstr='');
        '''
        
        result = subprocess.run(
            [DUCKDB_CONFIG['binary'], db_file, '-c', load_sql],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            # Get row count
            count_sql = f'SELECT COUNT(*) FROM "{table}"'
            count_result = subprocess.run(
                [DUCKDB_CONFIG['binary'], db_file, '-c', count_sql],
                capture_output=True, text=True
            )
            
            if count_result.returncode == 0:
                row_count = count_result.stdout.strip().split('\n')[-1].strip()
                print(f"    ✓ DuckDB: {database}.{table} ({row_count} rows)")
            return True
        else:
            print(f"    ✗ DuckDB failed for {database}.{table}: {result.stderr}")
            return False

    def create_duckdb_table_central(self, database, table, csv_path):
        """Create table in central DuckDB DB under schema = dataset name"""
        db_file = DUCKDB_CONFIG['central_db']
        load_sql = f'''
            CREATE SCHEMA IF NOT EXISTS "{database}";
            DROP TABLE IF EXISTS "{database}"."{table}";
            CREATE TABLE "{database}"."{table}" AS 
            SELECT * FROM read_csv_auto('{csv_path}', header=true, nullstr='');
        '''
        result = subprocess.run(
            [DUCKDB_CONFIG['binary'], db_file, '-c', load_sql],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ✗ Central DuckDB failed for {database}.{table}: {result.stderr}")
            return False
        # Optionally report row count
        count_sql = f'SELECT COUNT(*) FROM "{database}"."{table}"'
        count_result = subprocess.run(
            [DUCKDB_CONFIG['binary'], db_file, '-c', count_sql],
            capture_output=True, text=True
        )
        if count_result.returncode == 0:
            row_count = count_result.stdout.strip().split('\n')[-1].strip()
            print(f"    ✓ Central DuckDB: {database}.{table} ({row_count} rows)")
        return True

    def create_duckdb_primary_keys(self, database, primary_keys, existing_tables):
        """Create PK constraints in DuckDB (per-dataset DB and central schema)."""
        if not primary_keys:
            return True
        # Per-dataset DB
        per_db = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        for table, cols in primary_keys.items():
            if table not in existing_tables or not cols:
                continue
            cols_quoted = ', '.join([f'"{c}"' for c in cols])
            cname = f'{table}_pkey'
            sql = f'ALTER TABLE "{table}" ADD CONSTRAINT "{cname}" PRIMARY KEY ({cols_quoted});'
            if os.path.exists(per_db):
                res = subprocess.run([DUCKDB_CONFIG['binary'], per_db, '-c', sql], capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"    ⚠ DuckDB(per-db) PK {database}.{table} skipped: {res.stderr.strip()}")
        # Central DB
        central_db = DUCKDB_CONFIG['central_db']
        for table, cols in primary_keys.items():
            if table not in existing_tables or not cols:
                continue
            cols_quoted = ', '.join([f'"{c}"' for c in cols])
            cname = f'{table}_pkey'
            sql = f'ALTER TABLE "{database}"."{table}" ADD CONSTRAINT "{cname}" PRIMARY KEY ({cols_quoted});'
            res = subprocess.run([DUCKDB_CONFIG['binary'], central_db, '-c', sql], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"    ⚠ DuckDB(central) PK {database}.{table} skipped: {res.stderr.strip()}")
        return True

    def create_duckdb_foreign_keys(self, database, relationships, existing_tables):
        """Create FK constraints in DuckDB (per-dataset DB and central schema)."""
        if not relationships:
            return True
        per_db = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        central_db = DUCKDB_CONFIG['central_db']
        fk_count_per = 0
        fk_count_central = 0
        for idx, rel in enumerate(relationships):
            child = rel['child_table']
            parent = rel['parent_table']
            ccols = rel['child_columns']
            pcols = rel['parent_columns']
            if child not in existing_tables or parent not in existing_tables:
                continue
            if not ccols or not pcols or len(ccols) != len(pcols):
                continue
            ccols_q = ', '.join([f'"{c}"' for c in ccols])
            pcols_q = ', '.join([f'"{c}"' for c in pcols])
            cname = rel.get('constraint_name') or f'fk_{child}_{parent}_{idx}'
            cname = cname.replace('"', '')
            # Per-dataset
            if os.path.exists(per_db):
                sql_per = f'ALTER TABLE "{child}" ADD CONSTRAINT "{cname}" FOREIGN KEY ({ccols_q}) REFERENCES "{parent}" ({pcols_q});'
                res1 = subprocess.run([DUCKDB_CONFIG['binary'], per_db, '-c', sql_per], capture_output=True, text=True)
                if res1.returncode != 0:
                    print(f"    ⚠ DuckDB(per-db) FK {child}->{parent} skipped: {res1.stderr.strip()}")
                else:
                    fk_count_per += 1
            # Central (schema-qualified)
            sql_cent = f'ALTER TABLE "{database}"."{child}" ADD CONSTRAINT "{cname}" FOREIGN KEY ({ccols_q}) REFERENCES "{database}"."{parent}" ({pcols_q});'
            res2 = subprocess.run([DUCKDB_CONFIG['binary'], central_db, '-c', sql_cent], capture_output=True, text=True)
            if res2.returncode != 0:
                print(f"    ⚠ DuckDB(central) FK {database}.{child}->{parent} skipped: {res2.stderr.strip()}")
            else:
                fk_count_central += 1
        if fk_count_per:
            print(f"    ✓ Added {fk_count_per} DuckDB per-db FKs in {database}")
        if fk_count_central:
            print(f"    ✓ Added {fk_count_central} DuckDB central FKs in {database}")
        return True

    def drop_duckdb_database(self, database):
        """Drop DuckDB database (remove file)"""
        db_file = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        try:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"    ✓ Dropped DuckDB database: {database}")
            return True
        except Exception as e:
            print(f"    ✗ Failed to drop DuckDB database {database}: {e}")
            return False

    def drop_duckdb_schema_central(self, database):
        """Drop schema for dataset in central DuckDB DB"""
        db_file = DUCKDB_CONFIG['central_db']
        drop_sql = f'DROP SCHEMA IF EXISTS "{database}" CASCADE;'
        result = subprocess.run(
            [DUCKDB_CONFIG['binary'], db_file, '-c', drop_sql],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ✗ Failed to drop central DuckDB schema {database}: {result.stderr}")
            return False
        print(f"    ✓ Dropped central DuckDB schema: {database}")
        return True

    def postgres_dataset_exists(self, database: str, expected_tables: list) -> bool:
        """Check if PostgreSQL database exists and has all expected tables"""
        cursor = self.pg_conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        if not cursor.fetchone():
            cursor.close()
            return False
        cursor.close()

        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        try:
            db_conn = psycopg2.connect(**db_config)
            cur = db_conn.cursor()
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            existing = {r[0] for r in cur.fetchall()}
            cur.close()
            db_conn.close()
            return all(t in existing for t in expected_tables)
        except Exception:
            return False

    def duckdb_dataset_exists(self, database: str, expected_tables: list) -> bool:
        """Check if dataset schema exists in central DuckDB and includes expected tables"""
        db_file = DUCKDB_CONFIG['central_db']
        if not os.path.exists(db_file):
            return False
        # Check schema exists
        schema_sql = f"SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name='{database}'"
        res_schema = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', schema_sql], capture_output=True, text=True)
        if res_schema.returncode != 0:
            return False
        try:
            lines = [l.strip() for l in res_schema.stdout.splitlines() if l.strip()]
            if int(lines[-1]) == 0:
                return False
        except Exception:
            return False
        # Check each table exists in that schema
        for t in expected_tables:
            check_sql = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='{database}' AND table_name='{t}'"
            res = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', check_sql], capture_output=True, text=True)
            if res.returncode != 0:
                return False
            try:
                lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                if int(lines[-1]) == 0:
                    return False
            except Exception:
                return False
        return True
    
    def import_database(self, database):
        """Download CSVs from MySQL, then import into PostgreSQL and DuckDB with force/skip logic"""
        print(f"\n📦 Processing dataset: {database}")

        # Check if database exists on MySQL
        cursor = self.mysql_conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        cursor.close()
        if database not in databases:
            print(f"  ✗ Database '{database}' not found on CTU server")
            return False

        # List tables in MySQL
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        cursor.execute("SHOW TABLES")
        all_tables = [t[0] for t in cursor.fetchall()]
        cursor.close()
        if not all_tables:
            print(f"  ✗ No tables in database '{database}'")
            return False

        print(f"  Found {len(all_tables)} tables")

        # Prepare CSVs: redownload if --force or missing
        total_rows_downloaded = 0
        column_types_per_table = {}
        non_empty_tables = []
        for table in all_tables:
            print(f"\n  📋 Table: {table}")
            csv_path = os.path.join(DATA_DIR, database, f"{table}.csv")

            # Count rows in MySQL to decide skip for empty
            cursor = self.mysql_conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM `{database}`.`{table}`")
            count = cursor.fetchone()[0]
            cursor.close()
            if count == 0:
                print("    ⚠ Skipping empty table")
                continue
            non_empty_tables.append(table)

            need_download = self.force or (not os.path.exists(csv_path))
            if need_download:
                print("    ⬇️  Exporting from MySQL to CSV...")
                row_count, columns, column_types = self.export_to_csv_typed(database, table, csv_path)
                total_rows_downloaded += row_count
                column_types_per_table[table] = column_types
                print(f"    ✓ Exported {row_count} rows to {csv_path}")
            else:
                # Load types for later import
                column_types_per_table[table] = self.get_mysql_column_types(database, table)
                print("    ↪ Using cached CSV (pass --force to redownload)")

        # Check dataset existence in targets (consider only non-empty tables)
        tables = non_empty_tables
        pg_exists = self.postgres_dataset_exists(database, tables)
        duck_exists = self.duckdb_dataset_exists(database, tables)

        if pg_exists and duck_exists and not self.force:
            print("  ✅ Dataset already present in PostgreSQL and DuckDB. Skipping import.")
            return True

        # Drop datasets if forcing
        if self.force:
            print("  🔁 --force specified: dropping existing datasets before import")
            self.drop_postgresql_database(database)
            self.drop_duckdb_database(database)
            self.drop_duckdb_schema_central(database)

        # Import to PostgreSQL if needed
        if not pg_exists or self.force:
            print("  🐘 Importing into PostgreSQL...")
            for table in tables:
                csv_path = os.path.join(DATA_DIR, database, f"{table}.csv")
                if not os.path.exists(csv_path):
                    # Table might be empty or skipped
                    continue
                column_types = column_types_per_table.get(table) or self.get_mysql_column_types(database, table)
                self.create_postgresql_table_typed(database, table, column_types, csv_path)

        # Import to DuckDB if needed
        if not duck_exists or self.force:
            print("  🦆 Importing into DuckDB...")
            for table in tables:
                csv_path = os.path.join(DATA_DIR, database, f"{table}.csv")
                if not os.path.exists(csv_path):
                    continue
                # Per-dataset DB (optional)
                self.create_duckdb_table(database, table, csv_path)
                # Central consolidated DB with schema=dataset
                self.create_duckdb_table_central(database, table, csv_path)

        # After imports, fetch PK/FK metadata and add constraints to PostgreSQL
        existing_tables = set(tables)
        try:
            pks = self.get_mysql_primary_keys(database)
        except Exception as e:
            print(f"  ⚠ Failed to fetch primary keys for {database}: {e}")
            pks = {}
        try:
            relationships = self.get_mysql_relationships(database)
        except Exception as e:
            print(f"  ⚠ Failed to fetch relationships for {database}: {e}")
            relationships = []

        # Create PKs first, then FKs (FKs require unique/PK on parent)
        self.create_postgresql_primary_keys(database, pks, existing_tables)
        self.create_postgresql_foreign_keys(database, relationships, existing_tables)
        # Also add constraints in DuckDB (best-effort, may be skipped if unsupported)
        self.create_duckdb_primary_keys(database, pks, existing_tables)
        self.create_duckdb_foreign_keys(database, relationships, existing_tables)

        # Persist relationships metadata
        try:
            self.save_relationships_metadata(database, relationships)
            self.write_relationships_to_central_duckdb(database, relationships)
        except Exception as e:
            print(f"  ⚠ Failed to persist relationships metadata: {e}")

        # Update stats (roughly counted by CSVs we handled)
        self.stats['tables_imported'] += len([t for t in tables if os.path.exists(os.path.join(DATA_DIR, database, f"{t}.csv"))])
        self.stats['total_rows'] += total_rows_downloaded
        self.stats['databases_imported'] += 1
        self.imported_datasets.append(database)
        return True
    
    def run(self):
        """Run the complete import process"""
        print("🚀 Starting Benchmark Dataset Import (with proper types)")
        print("=" * 60)
        
        self.connect()
        
        # Import selected databases
        for database in SELECTED_DATABASES:
            try:
                self.import_database(database)
            except Exception as e:
                print(f"  ✗ Error importing {database}: {e}")
                continue
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Import Summary")
        print(f"  Databases imported: {self.stats['databases_imported']}")
        print(f"  Tables imported: {self.stats['tables_imported']}")
        print(f"  Total rows: {self.stats['total_rows']:,}")
        print(f"  Duration: {datetime.now() - self.stats['start_time']}")
        
        # Save metadata
        metadata_path = os.path.join(DATA_DIR, 'imported_datasets.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'datasets': self.imported_datasets,
                'stats': {
                    'databases': self.stats['databases_imported'],
                    'tables': self.stats['tables_imported'],
                    'rows': self.stats['total_rows']
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n✅ Import complete! Metadata saved to {metadata_path}")
        
        # Close connections
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.pg_conn:
            self.pg_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import benchmark datasets from MySQL to PostgreSQL and DuckDB")
    parser.add_argument('--force', action='store_true', help='Redownload CSVs and reimport: drop and recreate datasets')
    args = parser.parse_args()

    importer = BenchmarkDatasetImporter(force=args.force)
    importer.run()
