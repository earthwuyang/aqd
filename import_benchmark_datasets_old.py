#!/usr/bin/env python3
"""
Import 10+ benchmark datasets into both PostgreSQL and DuckDB
Based on TiDB importing methodology but adapted for our query routing system

Imports from CTU relational server: relational.fel.cvut.cz
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
    # database will be set dynamically - admin uses 'postgres', datasets get their own databases
}

# Resolve repository base directory relative to this file
BASE_DIR = Path(__file__).resolve().parent

# DuckDB configuration (relative to repo)
DUCKDB_CONFIG = {
    'binary': str(BASE_DIR / 'duckdb' / 'duckdb'),
    'databases_dir': str(BASE_DIR / 'data' / 'duckdb_databases')
}

# Select 10+ databases known to have good data for benchmarking
SELECTED_DATABASES = [
    'accidents',      # Traffic accidents data - good for analytics
    'financial',      # Financial transactions - mixed OLTP/OLAP
    'Basketball_men', # Basketball statistics - analytical queries
    'northwind',      # Classic business data - balanced workload
    'sakila',        # Movie rental data - complex relationships
    'employees',     # Employee data - HR analytics
    'world',         # World cities/countries - geographic queries
    'imdb_small',    # IMDB movie data - text and analytical queries
    'hepatitis_std', # Medical data - statistical analysis
    'genes',         # Genetic data - scientific queries
    'oscar_nominees', # Oscar nominees - entertainment data
    'Airline',       # Airline data - time series analysis
    'ccs',           # Computer science data
    'Credit',        # Credit scoring data
    'Seznam'         # Czech web data
]

# Data directory (relative to repo)
DATA_DIR = str(BASE_DIR / 'data' / 'benchmark_data')

class BenchmarkDatasetImporter:
    def __init__(self):
        self.mysql_conn = None
        self.pg_conn = None
        self.stats = {
            'databases_imported': 0,
            'tables_imported': 0,
            'total_rows': 0,
            'start_time': datetime.now()
        }
        self.imported_datasets = []

    def ensure_pg_conn(self):
        """Ensure PostgreSQL connection is alive; reconnect if needed."""
        try:
            if self.pg_conn is None or self.pg_conn.closed:
                raise Exception("pg conn closed")
            cur = self.pg_conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        except Exception:
            try:
                if self.pg_conn:
                    try:
                        self.pg_conn.close()
                    except Exception:
                        pass
                # Connect to the default 'postgres' database for administrative operations
                admin_config = POSTGRESQL_CONFIG.copy()
                admin_config['database'] = 'postgres'
                self.pg_conn = psycopg2.connect(**admin_config)
                self.pg_conn.autocommit = True
                print("  ↻ Reconnected to PostgreSQL")
            except Exception as e:
                print(f"  ✗ Reconnect to PostgreSQL failed: {e}")
                raise
    
    def connect(self):
        """Connect to all databases"""
        print("🔌 Connecting to databases...")
        
        # Connect to MySQL (CTU server)
        print("  Connecting to CTU MySQL server...")
        self.mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        print("  ✓ Connected to CTU MySQL server")
        
        # Connect to PostgreSQL (admin connection)
        print("  Connecting to PostgreSQL...")
        admin_config = POSTGRESQL_CONFIG.copy()
        admin_config['database'] = 'postgres'
        self.pg_conn = psycopg2.connect(**admin_config)
        self.pg_conn.autocommit = True
        print("  ✓ Connected to PostgreSQL")
        
        # Ensure DuckDB databases directory exists
        print("  Setting up DuckDB...")
        os.makedirs(DUCKDB_CONFIG['databases_dir'], exist_ok=True)
        print("  ✓ DuckDB directory ready")
        
        print("✅ All database connections established")
    
    def check_database_exists(self, database):
        """Check if database exists on MySQL server"""
        cursor = self.mysql_conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        cursor.close()
        return database in databases
    
    def get_table_info(self, database):
        """Get tables and row counts for a database"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        cursor.execute("SHOW TABLES")
        tables = [t[0] for t in cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                count = cursor.fetchone()[0]
                if count > 0:  # Only include non-empty tables
                    table_info[table] = count
            except Exception as e:
                print(f"    Warning: Could not count rows in {table}: {e}")
                pass
        
        cursor.close()
        return table_info
    
    def export_to_csv(self, database, table, output_path):
        """Export table data to CSV for loading into PostgreSQL/DuckDB"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        
        # Get column names
        cursor.execute(f"DESCRIBE `{table}`")
        columns = [col[0] for col in cursor.fetchall()]
        
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
                
                # Process rows to handle special types
                for row in rows:
                    processed_row = []
                    for val in row:
                        if val is None:
                            processed_row.append('')
                        elif isinstance(val, bytes):
                            processed_row.append(val.decode('utf-8', errors='ignore'))
                        else:
                            processed_row.append(str(val))
                    writer.writerow(processed_row)
                    row_count += 1
        
        cursor.close()
        return row_count, columns
    
    def map_duckdb_to_pg_type(self, duck_type):
        t = duck_type.upper()
        if any(k in t for k in ["INT", "HUGEINT", "BIGINT", "SMALLINT", "TINYINT"]):
            return "BIGINT"
        if any(k in t for k in ["DOUBLE", "REAL", "FLOAT", "DECIMAL", "NUMERIC"]):
            return "DOUBLE PRECISION"
        if "BOOLEAN" in t:
            return "BOOLEAN"
        if "DATE" in t and "TIME" not in t:
            return "DATE"
        if any(k in t for k in ["TIMESTAMP", "DATETIME"]):
            return "TIMESTAMP"
        return "TEXT"

    def get_duckdb_column_types(self, database, table):
        import subprocess
        # Use the specific database file for this dataset
        db_file = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        
        sql = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """
        result = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', sql],
                                capture_output=True, text=True)
        col_types = {}
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.splitlines() if ln.strip() and not ln.lower().startswith('column_name')]
            for ln in lines:
                parts = [p.strip() for p in ln.split('|')]
                if len(parts) >= 2:
                    col_types[parts[0]] = parts[1]
        return col_types

    def check_postgresql_database_exists(self, database):
        """Check if PostgreSQL database already exists"""
        self.ensure_pg_conn()
        cursor = self.pg_conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

    def create_postgresql_database(self, database):
        """Create a separate database in PostgreSQL"""
        self.ensure_pg_conn()
        cursor = self.pg_conn.cursor()
        
        # Check if database already exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        if cursor.fetchone():
            print(f"    ✓ Database '{database}' already exists in PostgreSQL")
            cursor.close()
            return True
        
        try:
            # Create new database (cannot be done in a transaction)
            cursor.execute(f'CREATE DATABASE "{database}"')
            print(f"    ✓ Created PostgreSQL database: {database}")
            cursor.close()
            return True
        except Exception as e:
            print(f"    ✗ Failed to create PostgreSQL database {database}: {e}")
            cursor.close()
            return False

    def create_postgresql_table(self, database, table, columns, csv_path):
        """Create table in PostgreSQL database and load CSV data with inferred types"""
        # Ensure the database exists
        if not self.create_postgresql_database(database):
            return False
        
        # Connect to the specific database
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cursor = db_conn.cursor()
            
            duck_types = self.get_duckdb_column_types(database, table)
            column_defs = []
            for col in columns:
                pg_t = self.map_duckdb_to_pg_type(duck_types.get(col, 'TEXT'))
                column_defs.append(f'"{col}" {pg_t}')

            create_sql = f'''
                DROP TABLE IF EXISTS "{table}";
                CREATE TABLE "{table}" (
                    {', '.join(column_defs)}
                );
            '''
            cursor.execute(create_sql)

            copy_sql = f'''
                COPY "{table}" FROM '{csv_path}'
                WITH (FORMAT CSV, HEADER TRUE, NULL '', ENCODING 'UTF8');
            '''
            cursor.execute(copy_sql)
            print(f"    ✓ Loaded into PostgreSQL database: {database}.{table}")
            return True
        except Exception as e:
            print(f"    ✗ PostgreSQL load failed for {database}.{table}: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'db_conn' in locals():
                db_conn.close()
    
    def create_duckdb_table(self, database, table, csv_path):
        """Create table in DuckDB and load CSV data"""
        # Create separate database file for this dataset
        db_file = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        
        # Load CSV directly (DuckDB auto-detects types) - no schema needed since each dataset has its own database
        load_sql = f'''
            DROP TABLE IF EXISTS "{table}";
            CREATE TABLE "{table}" AS 
            SELECT * FROM read_csv_auto('{csv_path}', header=true, null_padding=true, parallel=false);
        '''
        
        result = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', load_sql],
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ Loaded into DuckDB: {database}.{table}")
            return True
        else:
            print(f"    ✗ DuckDB load failed for {database}.{table}: {result.stderr}")
            return False
    
    def import_table(self, database, table, skip_postgresql=False):
        """Import a single table into both PostgreSQL and DuckDB"""
        print(f"    📊 Importing table: {table}")
        
        # Export to CSV
        csv_path = f"{DATA_DIR}/{database}/{table}.csv"
        try:
            row_count, columns = self.export_to_csv(database, table, csv_path)
            print(f"      Exported {row_count} rows to CSV")
        except Exception as e:
            print(f"      ✗ CSV export failed: {e}")
            return 0
        
        # Always import into DuckDB (separate database file for each dataset)
        duck_success = self.create_duckdb_table(database, table, csv_path)
        
        # Conditionally import into PostgreSQL
        if skip_postgresql:
            print(f"      ⏭️ Skipping PostgreSQL import (database already exists)")
            pg_success = True  # Consider it successful since we're skipping intentionally
        else:
            pg_success = self.create_postgresql_table(database, table, columns, csv_path)
        
        # Clean up CSV file to save space
        try:
            os.remove(csv_path)
        except:
            pass
        
        if pg_success or duck_success:
            self.stats['tables_imported'] += 1
            self.stats['total_rows'] += row_count
            return row_count
        
        return 0
    
    def import_database(self, database):
        """Import an entire database"""
        print(f"\n{'='*60}")
        print(f"📊 Importing database: {database}")
        print(f"{'='*60}")
        
        # Check if database exists
        if not self.check_database_exists(database):
            print(f"  ✗ Database '{database}' not found on MySQL server")
            return False
        
        # Get table info
        table_info = self.get_table_info(database)
        if not table_info:
            print(f"  ✗ Database '{database}' has no non-empty tables")
            return False
        
        total_rows = sum(table_info.values())
        print(f"  📈 Found {len(table_info)} non-empty tables with {total_rows:,} total rows")
        
        # Skip if too small
        if total_rows < 100:
            print(f"  ⚠️ Skipping {database} - insufficient data ({total_rows} rows)")
            return False
        
        # Check if PostgreSQL database already exists
        pg_exists = self.check_postgresql_database_exists(database)
        if pg_exists:
            print(f"  📋 PostgreSQL database '{database}' already exists - skipping PostgreSQL import")
            print(f"  🔄 Re-importing into DuckDB for consistency...")
        
        # Import each table
        successful_tables = 0
        for table, expected_rows in table_info.items():
            try:
                rows = self.import_table(database, table, skip_postgresql=pg_exists)
                if rows > 0:
                    successful_tables += 1
            except Exception as e:
                print(f"      ✗ Failed to import {table}: {e}")
                continue
        
        if successful_tables > 0:
            self.stats['databases_imported'] += 1
            self.imported_datasets.append({
                'name': database,
                'tables': successful_tables,
                'total_rows': total_rows
            })
            print(f"  ✅ Successfully imported {successful_tables} tables from {database}")
            return True
        else:
            print(f"  ✗ No tables successfully imported from {database}")
            return False
    
    def run_with_tmux(self):
        """Main import process using tmux for long-running operations"""
        print("🚀 Starting Benchmark Dataset Import")
        print("=" * 60)
        
        self.connect()
        
        # Create data directory
        os.makedirs(DATA_DIR, exist_ok=True)
        
        successful_imports = []
        
        # Import selected databases
        for database in SELECTED_DATABASES:
            if self.import_database(database):
                successful_imports.append(database)
                if len(successful_imports) >= 10:
                    print(f"\n✅ Successfully imported {len(successful_imports)} datasets - target reached!")
                    break
            
            # Small delay between databases
            time.sleep(1)
        
        # Print summary
        duration = datetime.now() - self.stats['start_time']
        print("\n" + "="*60)
        print("📈 BENCHMARK DATASET IMPORT SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Databases imported: {self.stats['databases_imported']}")
        print(f"📋 Tables imported: {self.stats['tables_imported']}")
        print(f"📈 Total rows imported: {self.stats['total_rows']:,}")
        print(f"\n✅ Successfully imported datasets:")
        for dataset in self.imported_datasets:
            print(f"   🗄️  {dataset['name']}: {dataset['tables']} tables, {dataset['total_rows']:,} rows")
        
        # Close connections
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.pg_conn:
            self.pg_conn.close()
        
        # Save results
        results = {
            'datasets': self.imported_datasets,
            'stats': {
                'databases_imported': self.stats['databases_imported'],
                'tables_imported': self.stats['tables_imported'],
                'total_rows': self.stats['total_rows'],
                'duration_seconds': duration.total_seconds()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results file under repo data directory
        results_path = BASE_DIR / 'data' / 'imported_benchmark_datasets.json'
        os.makedirs(results_path.parent, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Import results saved to {results_path}")
        
        if len(successful_imports) >= 10:
            print(f"\n🎉 Successfully imported {len(successful_imports)} benchmark datasets!")
            print("   Ready for enhanced training data collection!")
            return True
        else:
            print(f"\n⚠️  Only imported {len(successful_imports)} datasets (target: 10)")
            return False

def run_import_in_tmux():
    """Run the import process in a tmux session"""
    import shlex
    
    session_name = "benchmark_import"
    
    # Kill existing session if any
    subprocess.run(["tmux", "kill-session", "-t", session_name], 
                  capture_output=True)
    
    # Create new tmux session and run import from this repo directory
    base_dir = Path(__file__).resolve().parent
    python_cmd = (
        f"cd {base_dir} && "
        "python3 -c "
        "'from import_benchmark_datasets import BenchmarkDatasetImporter; "
        "importer = BenchmarkDatasetImporter(); "
        "result = importer.run_with_tmux(); "
        "print(\"Import completed with result:\", result)'"
    )
    
    tmux_cmd = [
        "tmux", "new-session", "-d", "-s", session_name,
        "bash", "-c", python_cmd
    ]
    
    print(f"🚀 Starting benchmark data import in tmux session: {session_name}")
    print(f"📋 Monitor progress with: tmux attach -t {session_name}")
    print(f"⏹️  Stop with: tmux kill-session -t {session_name}")
    
    result = subprocess.run(tmux_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Tmux session started successfully")
        print(f"🔍 Check status: tmux list-sessions")
        return True
    else:
        print(f"✗ Failed to start tmux session: {result.stderr}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tmux":
        # Run in tmux
        success = run_import_in_tmux()
        sys.exit(0 if success else 1)
    else:
        # Run directly
        importer = BenchmarkDatasetImporter()
        success = importer.run_with_tmux()
        sys.exit(0 if success else 1)
