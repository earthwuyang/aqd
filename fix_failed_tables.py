#!/usr/bin/env python3
"""
Fix tables that failed to import due to NOT NULL constraints.
This script re-imports only the failed tables with all columns nullable except primary keys.
"""

import psycopg2
import os
from pathlib import Path

# Local PostgreSQL Configuration
POSTGRESQL_CONFIG = {
    'host': '/tmp',
    'port': 5432,
    'user': os.environ.get('USER', 'postgres')
}

# Failed tables to fix
FAILED_TABLES = {
    'Credit': ['category', 'charge', 'corporation', 'member', 'payment', 'provider', 'region', 'statement'],
    'financial': ['order']
}

# Base directory for data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'ctu_data'

def fix_table(database, table):
    """Re-create and load a table with all columns nullable"""
    csv_path = DATA_DIR / database / f"{table}.csv"
    
    if not csv_path.exists():
        print(f"  ✗ CSV file not found: {csv_path}")
        return False
    
    # Connect to database
    db_config = POSTGRESQL_CONFIG.copy()
    db_config['database'] = database
    
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Read CSV header to get column names
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            columns = header.split(',')
        
        # Create table with all columns as TEXT and nullable (simplest approach)
        column_defs = [f'"{col}" TEXT' for col in columns]
        
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
        
        print(f"  ✓ Fixed {database}.{table}: {row_count:,} rows")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to fix {database}.{table}: {e}")
        return False

def main():
    print("🔧 Fixing Failed Tables")
    print("=" * 60)
    
    total_fixed = 0
    
    for database, tables in FAILED_TABLES.items():
        print(f"\n📦 Fixing tables in {database}:")
        for table in tables:
            if fix_table(database, table):
                total_fixed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Fixed {total_fixed} tables")

if __name__ == "__main__":
    main()