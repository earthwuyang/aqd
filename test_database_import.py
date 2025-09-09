#!/usr/bin/env python3
"""
Test script to demonstrate the new database-per-dataset approach
"""

import psycopg2
import sys

POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'wuy'
}

def test_database_creation():
    """Test creating a database and connecting to it"""
    try:
        # Connect to admin database
        admin_config = POSTGRESQL_CONFIG.copy()
        admin_config['database'] = 'postgres'
        admin_conn = psycopg2.connect(**admin_config)
        admin_conn.autocommit = True
        
        cursor = admin_conn.cursor()
        
        # Check existing databases
        cursor.execute("SELECT datname FROM pg_database WHERE datname NOT IN ('template0', 'template1', 'postgres') ORDER BY datname;")
        existing_databases = [row[0] for row in cursor.fetchall()]
        print(f"📊 Existing dataset databases: {existing_databases}")
        
        # Test creating a new database (if it doesn't exist)
        test_db = 'test_dataset'
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db,))
        if not cursor.fetchone():
            print(f"🔨 Creating test database: {test_db}")
            cursor.execute(f'CREATE DATABASE "{test_db}"')
            
            # Connect to the new database and create a test table
            db_config = POSTGRESQL_CONFIG.copy()
            db_config['database'] = test_db
            test_conn = psycopg2.connect(**db_config)
            test_conn.autocommit = True
            test_cursor = test_conn.cursor()
            
            test_cursor.execute('CREATE TABLE IF NOT EXISTS sample_table (id SERIAL, name TEXT)')
            test_cursor.execute("INSERT INTO sample_table (name) VALUES ('Test Data')")
            
            test_cursor.execute("SELECT * FROM sample_table")
            rows = test_cursor.fetchall()
            print(f"✅ Successfully created database and table with {len(rows)} rows")
            
            test_cursor.close()
            test_conn.close()
        else:
            print(f"✓ Database {test_db} already exists")
        
        cursor.close()
        admin_conn.close()
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_database_creation()
    sys.exit(0 if success else 1)