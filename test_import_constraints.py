#!/usr/bin/env python3
"""
Test script to verify constraint creation during import
"""

import psycopg2
import subprocess
import json

def check_postgres_constraints(database):
    """Check if constraints exist in PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='wuy',
            password='',
            database=database
        )
        cur = conn.cursor()
        
        # Check for primary keys
        cur.execute("""
            SELECT table_name, constraint_name, constraint_type 
            FROM information_schema.table_constraints 
            WHERE table_schema='public' 
            AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
            ORDER BY constraint_type, table_name
        """)
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return results
    except Exception as e:
        return f"Error: {e}"

def check_duckdb_constraints(db_path):
    """Check if constraints exist in DuckDB"""
    try:
        cmd = f'./duckdb/duckdb {db_path} -c "SELECT COUNT(*) FROM information_schema.table_constraints WHERE constraint_type IN (\'PRIMARY KEY\', \'FOREIGN KEY\');"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=== Constraint Verification Report ===\n")
    
    # Test databases
    test_dbs = ['financial', 'imdb_small', 'Basketball_men']
    
    for db in test_dbs:
        print(f"\n--- Database: {db} ---")
        
        # Check PostgreSQL
        print("\nPostgreSQL Constraints:")
        pg_constraints = check_postgres_constraints(db)
        if isinstance(pg_constraints, list):
            if pg_constraints:
                for table, name, ctype in pg_constraints[:5]:  # Show first 5
                    print(f"  {ctype}: {table}.{name}")
                if len(pg_constraints) > 5:
                    print(f"  ... and {len(pg_constraints)-5} more")
            else:
                print("  ❌ No PRIMARY KEY or FOREIGN KEY constraints found")
        else:
            print(f"  ❌ {pg_constraints}")
        
        # Check DuckDB
        print(f"\nDuckDB Constraints (data/duckdb_databases/{db}.db):")
        duckdb_result = check_duckdb_constraints(f"data/duckdb_databases/{db}.db")
        print(f"  {duckdb_result}")
    
    # Check if relationships.json exists
    print("\n--- Relationships Metadata ---")
    import os
    for db in test_dbs:
        rel_path = f"data/benchmark_data/{db}/relationships.json"
        if os.path.exists(rel_path):
            print(f"  ✓ {db}/relationships.json exists")
            try:
                with open(rel_path, 'r') as f:
                    rels = json.load(f)
                    print(f"    PKs: {len(rels.get('primary_keys', {}))}, FKs: {len(rels.get('foreign_keys', {}))}")
            except:
                print(f"    Unable to parse")
        else:
            print(f"  ❌ {db}/relationships.json not found")

if __name__ == "__main__":
    main()