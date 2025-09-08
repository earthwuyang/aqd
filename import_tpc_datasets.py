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
    'database': str(BASE_DIR / 'data' / 'benchmark_datasets.db')
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
    def run(self, cmd: List[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
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

    def get_duckdb_column_types(self, schema: str, table: str) -> Dict[str, str]:
        sql = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        result = subprocess.run([DUCKDB_CONFIG['binary'], DUCKDB_CONFIG['database'], '-c', sql],
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
            self.run(['git', 'clone', '--depth', '1', 'https://github.com/gregrahn/tpch-kit.git', str(repo_dir)])
        # Build
        print('🔧 Building tpch-kit dbgen...')
        dbgen_dir = repo_dir / 'dbgen'
        # Some repos need edit of makefile for GCC; try make straightforwardly
        self.run(['make'], cwd=dbgen_dir)
        if not (dbgen_dir / 'dbgen').exists():
            raise RuntimeError('Failed to build tpch dbgen')
        # Generate data
        out_dir = DATA_DIR / 'tpch' / 'raw'
        print(f'🧪 Generating TPC-H data at SF={self.sf}...')
        self.run(['./dbgen', '-f', '-s', str(self.sf)], cwd=dbgen_dir)
        # Move .tbl files to raw dir
        for tbl in dbgen_dir.glob('*.tbl'):
            shutil.move(str(tbl), out_dir / tbl.name)
        print('✓ TPC-H data generated')

    def clone_and_build_tpcds(self):
        repo_dir = BUILD_DIR / 'tpcds' / 'tpcds-kit'
        if not repo_dir.exists():
            print('⏬ Cloning tpcds-kit...')
            self.run(['git', 'clone', '--depth', '1', 'https://github.com/gregrahn/tpcds-kit.git', str(repo_dir)])
        # Build
        print('🔧 Building tpcds-kit dsdgen...')
        tools_dir = repo_dir / 'tools'
        # Try common build incantations
        try:
            self.run(['make', 'OS=LINUX'], cwd=tools_dir)
        except subprocess.CalledProcessError:
            self.run(['make', '-f', 'Makefile.suite'], cwd=tools_dir)
        if not (tools_dir / 'dsdgen').exists():
            raise RuntimeError('Failed to build tpcds dsdgen')
        # Generate data
        out_dir = DATA_DIR / 'tpcds' / 'raw'
        print(f'🧪 Generating TPC-DS data at SF={self.sf}...')
        self.run(['./dsdgen', '-FORCE', '-SCALE', str(self.sf), '-DIR', str(out_dir)], cwd=tools_dir)
        print('✓ TPC-DS data generated')

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
        schema = 'tpch' if suite == 'tpch' else 'tpcds'
        subprocess.run([DUCKDB_CONFIG['binary'], DUCKDB_CONFIG['database'], '-c', f'CREATE SCHEMA IF NOT EXISTS "{schema}";'], capture_output=True)
        csv_dir = DATA_DIR / suite / 'csv'
        for csv_path in sorted(csv_dir.glob('*.csv')):
            table = csv_path.stem
            load_sql = f"""
                DROP TABLE IF EXISTS "{schema}"."{table}";
                CREATE TABLE "{schema}"."{table}" AS
                SELECT * FROM read_csv_auto('{csv_path}', header=true, null_padding=true);
            """
            result = subprocess.run([DUCKDB_CONFIG['binary'], DUCKDB_CONFIG['database'], '-c', load_sql], capture_output=True, text=True)
            if result.returncode == 0:
                print(f'    ✓ DuckDB loaded: {schema}.{table}')
            else:
                print(f'    ✗ DuckDB load failed for {schema}.{table}: {result.stderr}')

    def load_into_postgresql(self, suite: str):
        self.ensure_pg_conn()
        schema = 'tpch' if suite == 'tpch' else 'tpcds'
        cur = self.pg_conn.cursor()
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        csv_dir = DATA_DIR / suite / 'csv'
        for csv_path in sorted(csv_dir.glob('*.csv')):
            table = csv_path.stem
            duck_types = self.get_duckdb_column_types(schema, table)
            # Read header
            with open(csv_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
            cols_def = ', '.join([f'"{c}" {self.map_duckdb_to_pg_type(duck_types.get(c, "TEXT"))}' for c in header])
            ddl = f"""
                DROP TABLE IF EXISTS "{schema}"."{table}";
                CREATE TABLE "{schema}"."{table}" ({cols_def});
            """
            cur.execute(ddl)
            copy_sql = f"""
                COPY "{schema}"."{table}" FROM '{csv_path}' WITH (FORMAT CSV, HEADER TRUE, NULL '', ENCODING 'UTF8');
            """
            try:
                cur.execute(copy_sql)
                print(f'    ✓ PostgreSQL loaded: {schema}.{table}')
            except Exception as e:
                print(f'    ✗ PostgreSQL load failed for {schema}.{table}: {e}')
        cur.close()

    # ---------- Orchestration ----------
    def connect(self):
        print('🔌 Connecting to PostgreSQL and preparing DuckDB...')
        # Connect PG
        self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
        self.pg_conn.autocommit = True
        # Ensure DuckDB database exists
        os.makedirs(os.path.dirname(DUCKDB_CONFIG['database']), exist_ok=True)
        subprocess.run([DUCKDB_CONFIG['binary'], DUCKDB_CONFIG['database'], '-c', 'SELECT 1;'], capture_output=True)
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

