#!/usr/bin/env python3
"""
Import TPC-H and TPC-DS datasets (1GB each) into PostgreSQL and DuckDB.

Steps per suite:
- Clone kit repo (tpch-kit / tpcds-kit)
- Build data generator binaries (dbgen / dsdgen)
- Generate 1GB data into repo data directory
- Convert raw .tbl/.dat files (| delimited with trailing pipe) to CSV with headers
- Load into DuckDB (auto-typed) and PostgreSQL (types mapped from DuckDB)

All paths are relative to this repo.
"""

import os
import sys
import csv
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import psycopg2


# Base directories and configs
BASE_DIR = Path(__file__).resolve().parent
BUILD_DIR = BASE_DIR / 'build' / 'tpc'
DATA_DIR = BASE_DIR / 'data' / 'tpc'

POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'wuy',
    'database': 'postgres'
}

DUCKDB_CONFIG = {
    'binary': str(BASE_DIR / 'duckdb' / 'duckdb'),
    'databases_dir': str(BASE_DIR / 'data' / 'duckdb_databases')
}


class TPCImporter:
    def __init__(self, sf: int = 1):
        self.sf = sf
        self.pg_conn = None
        self.stats = {
            'start_time': datetime.now(),
            'tpch': {'tables': 0, 'rows': 0},
            'tpcds': {'tables': 0, 'rows': 0},
        }

    # ---------- Utilities ----------
    def run_cmd(self, cmd: List[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=check)

    def ensure_pg_conn(self):
        try:
            if self.pg_conn is None or self.pg_conn.closed:
                raise Exception('pg conn closed')
            cur = self.pg_conn.cursor()
            cur.execute('SELECT 1')
            cur.close()
        except Exception:
            try:
                if self.pg_conn:
                    try:
                        self.pg_conn.close()
                    except Exception:
                        pass
                self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
                self.pg_conn.autocommit = True
                print('  ↻ Reconnected to PostgreSQL')
            except Exception as e:
                print(f'  ✗ Reconnect to PostgreSQL failed: {e}')
                raise

    def map_duckdb_to_pg_type(self, duck_type: str) -> str:
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

    def get_duckdb_column_types(self, database: str, table: str) -> Dict[str, str]:
        sql = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """
        # Use the specific database file for this dataset
        db_file = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        result = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', sql],
                                capture_output=True, text=True)
        col_types: Dict[str, str] = {}
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.splitlines() if ln.strip() and not ln.lower().startswith('column_name')]
            for ln in lines:
                parts = [p.strip() for p in ln.split('|')]
                if len(parts) >= 2:
                    col_types[parts[0]] = parts[1]
        return col_types

    # ---------- Repo setup and data generation ----------
    def prepare_dirs(self):
        (BUILD_DIR / 'tpch').mkdir(parents=True, exist_ok=True)
        (BUILD_DIR / 'tpcds').mkdir(parents=True, exist_ok=True)
        (DATA_DIR / 'tpch' / 'raw').mkdir(parents=True, exist_ok=True)
        (DATA_DIR / 'tpch' / 'csv').mkdir(parents=True, exist_ok=True)
        (DATA_DIR / 'tpcds' / 'raw').mkdir(parents=True, exist_ok=True)
        (DATA_DIR / 'tpcds' / 'csv').mkdir(parents=True, exist_ok=True)

    def clone_and_build_tpch(self):
        repo_dir = BUILD_DIR / 'tpch' / 'tpch-kit'
        if not repo_dir.exists():
            print('⏬ Cloning tpch-kit...')
            self.run_cmd(['git', 'clone', '--depth', '1', 'https://github.com/gregrahn/tpch-kit.git', str(repo_dir)])
        
        # Build
        dbgen_dir = repo_dir / 'dbgen'
        if not (dbgen_dir / 'dbgen').exists():
            print('🔧 Building tpch-kit dbgen...')
            # Some repos need edit of makefile for GCC; try make straightforwardly
            self.run_cmd(['make'], cwd=dbgen_dir)
            if not (dbgen_dir / 'dbgen').exists():
                raise RuntimeError('Failed to build tpch dbgen')
        else:
            print('✓ TPC-H dbgen already built')
            
        # Generate data
        out_dir = DATA_DIR / 'tpch' / 'raw'
        if not list(out_dir.glob('*.tbl')):
            print(f'🧪 Generating TPC-H data at SF={self.sf}...')
            self.run_cmd(['./dbgen', '-f', '-s', str(self.sf)], cwd=dbgen_dir)
            # Move .tbl files to raw dir
            for tbl in dbgen_dir.glob('*.tbl'):
                shutil.move(str(tbl), out_dir / tbl.name)
            print('✓ TPC-H data generated')
        else:
            print('✓ TPC-H data already exists')

    def clone_and_build_tpcds(self):
        repo_dir = BUILD_DIR / 'tpcds' / 'tpcds-kit'
        if not repo_dir.exists():
            print('⏬ Cloning tpcds-kit...')
            self.run_cmd(['git', 'clone', '--depth', '1', 'https://github.com/gregrahn/tpcds-kit.git', str(repo_dir)])
            
        # Build
        tools_dir = repo_dir / 'tools'
        if not (tools_dir / 'dsdgen').exists():
            print('🔧 Building tpcds-kit dsdgen...')
            # Try common build incantations
            try:
                self.run_cmd(['make', 'OS=LINUX'], cwd=tools_dir)
            except subprocess.CalledProcessError:
                self.run_cmd(['make', '-f', 'Makefile.suite'], cwd=tools_dir)
            if not (tools_dir / 'dsdgen').exists():
                raise RuntimeError('Failed to build tpcds dsdgen')
        else:
            print('✓ TPC-DS dsdgen already built')
            
        # Generate data
        out_dir = DATA_DIR / 'tpcds' / 'raw'
        if not list(out_dir.glob('*.dat')):
            print(f'🧪 Generating TPC-DS data at SF={self.sf}...')
            self.run_cmd(['./dsdgen', '-FORCE', '-SCALE', str(self.sf), '-DIR', str(out_dir)], cwd=tools_dir)
            print('✓ TPC-DS data generated')
        else:
            print('✓ TPC-DS data already exists')

    # ---------- Conversion ----------
    @staticmethod
    def convert_pipe_file_to_csv(src_path: Path, dst_path: Path) -> Tuple[int, int]:
        """Convert a pipe-delimited file with trailing pipe to CSV with header.
        Returns (rows, cols).
        """
        rows = 0
        cols = 0
        with open(src_path, 'r', encoding='utf-8', errors='ignore') as fin, \
             open(dst_path, 'w', newline='', encoding='utf-8') as fout:
            writer = None
            for line in fin:
                # Remove ending newline and trailing pipe
                line = line.rstrip('\n')
                if line.endswith('|'):
                    line = line[:-1]
                parts = line.split('|') if '|' in line else line.split(',')
                if writer is None:
                    cols = len(parts) if parts[0] != '' else 0
                    headers = [f'col{i+1}' for i in range(cols)]
                    writer = csv.writer(fout)
                    writer.writerow(headers)
                if cols == 0:
                    continue
                writer.writerow(parts[:cols])
                rows += 1
        return rows, cols

    def convert_all_to_csv(self, suite: str):
        raw_dir = DATA_DIR / suite / 'raw'
        csv_dir = DATA_DIR / suite / 'csv'
        total_rows = 0
        tables = 0
        for file in sorted(raw_dir.glob('*.tbl')) + sorted(raw_dir.glob('*.dat')):
            table = file.stem
            dst = csv_dir / f'{table}.csv'
            r, c = self.convert_pipe_file_to_csv(file, dst)
            if r > 0 and c > 0:
                tables += 1
                total_rows += r
                print(f'  • {suite}.{table}: {r:,} rows, {c} cols')
        self.stats[suite]['tables'] = tables
        self.stats[suite]['rows'] = total_rows

    # ---------- Loading ----------
    def load_into_duckdb(self, suite: str):
        database = (f'tpch_sf{self.sf}' if suite == 'tpch' else f'tpcds_sf{self.sf}')
        # Create separate database file for this dataset
        db_file = os.path.join(DUCKDB_CONFIG['databases_dir'], f'{database}.db')
        os.makedirs(DUCKDB_CONFIG['databases_dir'], exist_ok=True)
        
        csv_dir = DATA_DIR / suite / 'csv'
        for csv_path in sorted(csv_dir.glob('*.csv')):
            table = csv_path.stem
            load_sql = f"""
                DROP TABLE IF EXISTS "{table}";
                CREATE TABLE "{table}" AS
                SELECT * FROM read_csv_auto('{csv_path}', header=true, null_padding=true, parallel=false);
            """
            result = subprocess.run([DUCKDB_CONFIG['binary'], db_file, '-c', load_sql], capture_output=True, text=True)
            if result.returncode == 0:
                print(f'    ✓ DuckDB loaded: {database}.{table}')
            else:
                print(f'    ✗ DuckDB load failed for {database}.{table}: {result.stderr}')

    def check_postgresql_database_exists(self, database: str) -> bool:
        """Check if PostgreSQL database already exists"""
        self.ensure_pg_conn()
        cursor = self.pg_conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

    def create_postgresql_database(self, database: str) -> bool:
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

    def load_into_postgresql(self, suite: str):
        database = (f'tpch_sf{self.sf}' if suite == 'tpch' else f'tpcds_sf{self.sf}')
        
        # Check if database already exists
        if self.check_postgresql_database_exists(database):
            print(f"    📋 PostgreSQL database '{database}' already exists - skipping PostgreSQL import")
            return
        
        # Create the database first
        if not self.create_postgresql_database(database):
            print(f"    ✗ Failed to create database {database}, skipping PostgreSQL import")
            return
        
        # Connect to the specific database
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cur = db_conn.cursor()
            
            csv_dir = DATA_DIR / suite / 'csv'
            for csv_path in sorted(csv_dir.glob('*.csv')):
                table = csv_path.stem
                duck_types = self.get_duckdb_column_types(database, table)
                # Read header
                with open(csv_path, 'r', encoding='utf-8') as f:
                    header = f.readline().strip().split(',')
                cols_def = ', '.join([f'"{c}" {self.map_duckdb_to_pg_type(duck_types.get(c, "TEXT"))}' for c in header])
                ddl = f"""
                    DROP TABLE IF EXISTS "{table}";
                    CREATE TABLE "{table}" ({cols_def});
                """
                cur.execute(ddl)
                copy_sql = f"""
                    COPY "{table}" FROM '{csv_path}' WITH (FORMAT CSV, HEADER TRUE, NULL '', ENCODING 'UTF8');
                """
                try:
                    cur.execute(copy_sql)
                    print(f'    ✓ PostgreSQL loaded: {database}.{table}')
                except Exception as e:
                    print(f'    ✗ PostgreSQL load failed for {database}.{table}: {e}')
            cur.close()
            db_conn.close()
            
        except Exception as e:
            print(f"    ✗ Failed to connect to database {database}: {e}")

    # ---------- Orchestration ----------
    def connect(self):
        print('🔌 Connecting to PostgreSQL and preparing DuckDB...')
        # Connect PG
        self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
        self.pg_conn.autocommit = True
        # Ensure DuckDB databases directory exists
        os.makedirs(DUCKDB_CONFIG['databases_dir'], exist_ok=True)
        print('✓ Connections ready')

    def run(self):
        print('🚀 Importing TPC-H and TPC-DS (SF=1)...')
        self.prepare_dirs()
        self.connect()

        # TPC-H
        print('\n=== TPC-H ===')
        self.clone_and_build_tpch()
        self.convert_all_to_csv('tpch')
        print('  Loading TPC-H into DuckDB...')
        self.load_into_duckdb('tpch')
        print('  Loading TPC-H into PostgreSQL...')
        self.load_into_postgresql('tpch')

        # TPC-DS
        print('\n=== TPC-DS ===')
        self.clone_and_build_tpcds()
        self.convert_all_to_csv('tpcds')
        print('  Loading TPC-DS into DuckDB...')
        self.load_into_duckdb('tpcds')
        print('  Loading TPC-DS into PostgreSQL...')
        self.load_into_postgresql('tpcds')

        duration = datetime.now() - self.stats['start_time']
        print('\n=== SUMMARY ===')
        print(f"TPC-H: {self.stats['tpch']['tables']} tables, {self.stats['tpch']['rows']:,} rows")
        print(f"TPC-DS: {self.stats['tpcds']['tables']} tables, {self.stats['tpcds']['rows']:,} rows")
        print(f'Duration: {duration}')


def main():
    sf = 1  # 1GB
    importer = TPCImporter(sf=sf)
    importer.run()


if __name__ == '__main__':
    main()
