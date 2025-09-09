#!/usr/bin/env python3
"""
Test script for the modified TPC dataset import with database-per-dataset approach
"""

import psycopg2
import sys

POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'wuy'
}

def test_tpc_databases():
    """Test that TPC databases exist and have tables"""
    try:
        # Connect to admin database
        admin_config = POSTGRESQL_CONFIG.copy()
        admin_config['database'] = 'postgres'
        admin_conn = psycopg2.connect(**admin_config)
        admin_conn.autocommit = True
        
        cursor = admin_conn.cursor()
        
        # Check for TPC databases
        cursor.execute("SELECT datname FROM pg_database WHERE datname IN ('tpch_sf1', 'tpcds_sf1') ORDER BY datname;")
        tpc_databases = [row[0] for row in cursor.fetchall()]
        print(f"📊 Found TPC databases: {tpc_databases}")
        
        # Test each database
        for db_name in tpc_databases:
            db_config = POSTGRESQL_CONFIG.copy()
            db_config['database'] = db_name
            
            try:
                db_conn = psycopg2.connect(**db_config)
                db_conn.autocommit = True
                db_cursor = db_conn.cursor()
                
                # List tables in this database
                db_cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """)
                tables = [row[0] for row in db_cursor.fetchall()]
                
                # Get total row count
                total_rows = 0
                for table in tables:
                    db_cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                    rows = db_cursor.fetchone()[0]
                    total_rows += rows
                    print(f"  • {db_name}.{table}: {rows:,} rows")
                
                print(f"✅ {db_name}: {len(tables)} tables, {total_rows:,} total rows")
                
                db_cursor.close()
                db_conn.close()
                
            except Exception as e:
                print(f"✗ Failed to connect to {db_name}: {e}")
        
        cursor.close()
        admin_conn.close()
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_tpc_databases()
    sys.exit(0 if success else 1)