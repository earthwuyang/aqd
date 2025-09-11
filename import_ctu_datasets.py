#!/usr/bin/env python3
"""
Import specific CTU benchmark datasets into PostgreSQL with pg_duckdb extension.

This script imports the following datasets from CTU MySQL server:
- airline
- credit
- carcinogenesis
- employee
- financial
- geneea
- hepatitis
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
import argparse

# CTU MySQL Server Configuration
MYSQL_CONFIG = {
    'host': 'relational.fel.cvut.cz',
    'port': 3306,
    'user': 'guest',
    'password': 'ctu-relational'
}

# Local PostgreSQL Configuration
POSTGRESQL_CONFIG = {
    'host': '/tmp',  # Use Unix socket
    'port': 5432,
    'user': os.environ.get('USER', 'postgres')
}

# Datasets to import (with correct case from CTU server)
SELECTED_DATABASES = [
    'Airline',
    'Credit',
    'Carcinogenesis',
    'employee',
    'financial',
    'geneea',
    'Hepatitis_std'  # Note: hepatitis is called Hepatitis_std on CTU
]

# Base directory for data storage
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'ctu_data'

class CTUDatasetImporter:
    def __init__(self, force: bool = False):
        self.mysql_conn = None
        self.pg_conn = None
        self.force = force
        self.stats = {
            'databases_imported': 0,
            'tables_imported': 0,
            'total_rows': 0,
            'start_time': datetime.now()
        }
        
    def connect(self):
        """Connect to MySQL and PostgreSQL"""
        print("🔌 Connecting to databases...")
        
        # Connect to CTU MySQL server
        print("  Connecting to CTU MySQL server...")
        try:
            self.mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
            print("  ✓ Connected to CTU MySQL server")
        except Exception as e:
            print(f"  ✗ Failed to connect to CTU MySQL: {e}")
            sys.exit(1)
        
        # Connect to local PostgreSQL
        print("  Connecting to local PostgreSQL...")
        try:
            admin_config = POSTGRESQL_CONFIG.copy()
            admin_config['database'] = 'postgres'
            self.pg_conn = psycopg2.connect(**admin_config)
            self.pg_conn.autocommit = True
            print("  ✓ Connected to PostgreSQL")
        except Exception as e:
            print(f"  ✗ Failed to connect to PostgreSQL: {e}")
            print("  Make sure PostgreSQL is running with: pg_ctl status")
            sys.exit(1)
    
    def get_mysql_column_types(self, database, table):
        """Get column types from MySQL"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{database}' AND TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = {}
        for row in cursor.fetchall():
            col_name, data_type, col_type, nullable, key = row
            columns[col_name] = {
                'mysql_type': data_type.upper(),
                'full_type': col_type,
                'nullable': nullable == 'YES',
                'is_primary': key == 'PRI'
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
            if mysql_type == 'TINYINT' and '(1)' in full_type:
                return 'BOOLEAN'
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
            if '(' in full_type:
                return full_type.replace('DECIMAL', 'NUMERIC').replace('DEC', 'NUMERIC').split(' ')[0]
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
                length = full_type.split('(')[1].split(')')[0]
                return f'{mysql_type}({length})'
            return 'VARCHAR(255)'
        elif mysql_type in ['TINYTEXT', 'TEXT', 'MEDIUMTEXT', 'LONGTEXT', 'ENUM', 'SET']:
            return 'TEXT'
        
        # Binary types
        elif mysql_type in ['BINARY', 'VARBINARY', 'BLOB', 'TINYBLOB', 'MEDIUMBLOB', 'LONGBLOB']:
            return 'BYTEA'
        
        # JSON
        elif mysql_type == 'JSON':
            return 'JSONB'
        
        # BIT
        elif mysql_type == 'BIT':
            if '(1)' in full_type:
                return 'BOOLEAN'
            return 'BIT VARYING'
        
        # Default
        else:
            return 'TEXT'
    
    def export_to_csv(self, database, table, output_path):
        """Export MySQL table to CSV"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        
        # Get column info
        column_types = self.get_mysql_column_types(database, table)
        columns = list(column_types.keys())
        
        # Count rows first
        count_cursor = self.mysql_conn.cursor()
        count_cursor.execute(f"USE `{database}`")
        count_cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        total_rows = count_cursor.fetchone()[0]
        count_cursor.close()
        
        # Export data
        cursor.execute(f"SELECT * FROM `{table}`")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)  # Header
            
            batch_size = 10000
            row_count = 0
            
            with tqdm.tqdm(total=total_rows, desc=f"    Exporting {table}", unit=" rows") as pbar:
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
                            elif isinstance(val, datetime):
                                processed_row.append(val.isoformat())
                            else:
                                processed_row.append(val)
                        writer.writerow(processed_row)
                        row_count += 1
                    pbar.update(len(rows))
        
        cursor.close()
        return row_count, columns, column_types
    
    def create_postgresql_database(self, database):
        """Create PostgreSQL database with pg_duckdb extension"""
        cursor = self.pg_conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone() is not None
        
        if exists and not self.force:
            print(f"    Database '{database}' already exists (use --force to recreate)")
            cursor.close()
            return True
        
        if exists and self.force:
            # Drop existing database
            print(f"    Dropping existing database '{database}'...")
            cursor.execute(f"""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = %s
            """, (database,))
            cursor.execute(f'DROP DATABASE IF EXISTS "{database}"')
        
        # Create database
        try:
            cursor.execute(f'CREATE DATABASE "{database}"')
            print(f"    ✓ Created database: {database}")
            
            # Connect to new database and create pg_duckdb extension
            db_config = POSTGRESQL_CONFIG.copy()
            db_config['database'] = database
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            db_cursor = db_conn.cursor()
            
            # Create pg_duckdb extension
            db_cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_duckdb")
            print(f"    ✓ Created pg_duckdb extension in {database}")
            
            db_cursor.close()
            db_conn.close()
            cursor.close()
            return True
            
        except Exception as e:
            print(f"    ✗ Failed to create database {database}: {e}")
            cursor.close()
            return False
    
    def create_postgresql_table(self, database, table, column_types, csv_path):
        """Create table in PostgreSQL and load data"""
        # Connect to specific database
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cursor = db_conn.cursor()
            
            # Create table with proper types
            column_defs = []
            primary_keys = []
            
            for col_name, type_info in column_types.items():
                pg_type = self.map_mysql_to_pg_type(type_info)
                # Only add NOT NULL for primary key columns, let others be nullable
                # MySQL metadata isn't always accurate about nullability
                if type_info.get('is_primary'):
                    null_clause = ' NOT NULL'
                    primary_keys.append(f'"{col_name}"')
                else:
                    null_clause = ''
                column_defs.append(f'"{col_name}" {pg_type}{null_clause}')
            
            # Add primary key constraint if exists
            if primary_keys:
                column_defs.append(f'PRIMARY KEY ({", ".join(primary_keys)})')
            
            create_sql = f'''
                DROP TABLE IF EXISTS "{table}" CASCADE;
                CREATE TABLE "{table}" (
                    {', '.join(column_defs)}
                );
            '''
            cursor.execute(create_sql)
            
            # Load data from CSV
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Skip header
                next(f)
                cursor.copy_expert(
                    f'''COPY "{table}" FROM STDIN WITH (FORMAT CSV, NULL '', ENCODING 'UTF8')''',
                    f
                )
            
            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            row_count = cursor.fetchone()[0]
            
            print(f"    ✓ Loaded {table}: {row_count:,} rows")
            
            cursor.close()
            db_conn.close()
            return True
            
        except Exception as e:
            print(f"    ✗ Failed to create/load {database}.{table}: {e}")
            return False
    
    def get_mysql_foreign_keys(self, database):
        """Get foreign key relationships from MySQL"""
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"""
            SELECT 
                CONSTRAINT_NAME,
                TABLE_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME,
                ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s 
                AND REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY CONSTRAINT_NAME, ORDINAL_POSITION
        """, (database,))
        
        fks = {}
        for row in cursor.fetchall():
            constraint_name, table_name, column_name, ref_table, ref_column, ord_pos = row
            key = (constraint_name, table_name, ref_table)
            
            if key not in fks:
                fks[key] = {
                    'constraint_name': constraint_name,
                    'table_name': table_name,
                    'columns': [],
                    'ref_table': ref_table,
                    'ref_columns': []
                }
            
            fks[key]['columns'].append(column_name)
            fks[key]['ref_columns'].append(ref_column)
        
        cursor.close()
        return list(fks.values())
    
    def create_foreign_keys(self, database, foreign_keys):
        """Create foreign keys in PostgreSQL"""
        if not foreign_keys:
            return
        
        db_config = POSTGRESQL_CONFIG.copy()
        db_config['database'] = database
        
        try:
            db_conn = psycopg2.connect(**db_config)
            db_conn.autocommit = True
            cursor = db_conn.cursor()
            
            created = 0
            for fk in foreign_keys:
                columns = ', '.join([f'"{c}"' for c in fk['columns']])
                ref_columns = ', '.join([f'"{c}"' for c in fk['ref_columns']])
                
                sql = f'''
                    ALTER TABLE "{fk['table_name']}"
                    ADD CONSTRAINT "{fk['constraint_name']}"
                    FOREIGN KEY ({columns})
                    REFERENCES "{fk['ref_table']}" ({ref_columns})
                    NOT VALID
                '''
                
                try:
                    cursor.execute(sql)
                    created += 1
                except Exception as e:
                    print(f"      ⚠ Skipping FK {fk['constraint_name']}: {str(e)[:50]}")
            
            if created > 0:
                print(f"    ✓ Created {created} foreign key constraints")
            
            cursor.close()
            db_conn.close()
            
        except Exception as e:
            print(f"    ⚠ Failed to create foreign keys: {e}")
    
    def import_database(self, database):
        """Import a database from MySQL to PostgreSQL"""
        print(f"\n📦 Processing dataset: {database}")
        
        # Check if database exists on MySQL
        cursor = self.mysql_conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        cursor.close()
        
        if database not in databases:
            print(f"  ✗ Database '{database}' not found on CTU server")
            return False
        
        # Get list of tables
        cursor = self.mysql_conn.cursor()
        cursor.execute(f"USE `{database}`")
        cursor.execute("SHOW TABLES")
        tables = [t[0] for t in cursor.fetchall()]
        cursor.close()
        
        if not tables:
            print(f"  ✗ No tables in database '{database}'")
            return False
        
        print(f"  Found {len(tables)} tables: {', '.join(tables)}")
        
        # Create PostgreSQL database
        if not self.create_postgresql_database(database):
            return False
        
        # Export and import each table
        data_dir = DATA_DIR / database
        data_dir.mkdir(parents=True, exist_ok=True)
        
        total_rows = 0
        for table in tables:
            print(f"\n  📋 Processing table: {table}")
            
            csv_path = data_dir / f"{table}.csv"
            
            # Export from MySQL if needed
            if self.force or not csv_path.exists():
                print("    Exporting from MySQL...")
                row_count, columns, column_types = self.export_to_csv(
                    database, table, str(csv_path)
                )
                print(f"    ✓ Exported {row_count:,} rows to CSV")
            else:
                print("    Using cached CSV file")
                column_types = self.get_mysql_column_types(database, table)
                # Count rows in CSV
                with open(csv_path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
            
            # Import to PostgreSQL
            if self.create_postgresql_table(database, table, column_types, str(csv_path)):
                total_rows += row_count
                self.stats['tables_imported'] += 1
        
        # Create foreign keys
        print("\n  Setting up foreign key constraints...")
        foreign_keys = self.get_mysql_foreign_keys(database)
        if foreign_keys:
            self.create_foreign_keys(database, foreign_keys)
        else:
            print("    No foreign keys found")
        
        # Update stats
        self.stats['databases_imported'] += 1
        self.stats['total_rows'] += total_rows
        
        print(f"\n  ✅ Successfully imported {database}")
        return True
    
    def run(self):
        """Run the import process"""
        print("🚀 Starting CTU Dataset Import to PostgreSQL")
        print("=" * 60)
        
        self.connect()
        
        # Process each database
        success = []
        failed = []
        
        for database in SELECTED_DATABASES:
            try:
                if self.import_database(database):
                    success.append(database)
                else:
                    failed.append(database)
            except Exception as e:
                print(f"  ✗ Error importing {database}: {e}")
                failed.append(database)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Import Summary")
        print(f"  Successfully imported: {', '.join(success) if success else 'None'}")
        if failed:
            print(f"  Failed: {', '.join(failed)}")
        print(f"  Total databases: {self.stats['databases_imported']}")
        print(f"  Total tables: {self.stats['tables_imported']}")
        print(f"  Total rows: {self.stats['total_rows']:,}")
        print(f"  Duration: {datetime.now() - self.stats['start_time']}")
        
        # Save metadata
        metadata_path = DATA_DIR / 'import_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'databases': success,
                'stats': {
                    'databases_imported': self.stats['databases_imported'],
                    'tables_imported': self.stats['tables_imported'],
                    'total_rows': self.stats['total_rows']
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n✅ Import complete! Metadata saved to {metadata_path}")
        
        # Test pg_duckdb access
        if success:
            print("\n🦆 Testing pg_duckdb access...")
            test_db = success[0]
            db_config = POSTGRESQL_CONFIG.copy()
            db_config['database'] = test_db
            
            try:
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()
                
                # Test DuckDB query through pg_duckdb
                cursor.execute("SELECT current_setting('duckdb.postgres_role', true)")
                print(f"  ✓ pg_duckdb is active in {test_db}")
                
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"  ⚠ Could not verify pg_duckdb: {e}")
        
        # Close connections
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.pg_conn:
            self.pg_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import CTU benchmark datasets into PostgreSQL with pg_duckdb"
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force re-import: drop existing databases and re-download data'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import mysql.connector
        import psycopg2
        import tqdm
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install mysql-connector-python psycopg2-binary tqdm")
        sys.exit(1)
    
    importer = CTUDatasetImporter(force=args.force)
    importer.run()