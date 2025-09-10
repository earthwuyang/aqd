#!/usr/bin/env python3
"""
Debug why queries are failing in live evaluation
"""

import psycopg2
import json
from pathlib import Path
import random

def test_queries():
    # Load test data
    with open('/home/wuy/DB/pg_duckdb_postgres_2/data/execution_data/all_datasets_unified_training_data.json', 'r') as f:
        data = json.load(f)
    
    # Sample some queries
    random.seed(42)
    samples = random.sample(data, min(10, len(data)))
    
    print("Testing query execution...")
    print("=" * 80)
    
    for i, entry in enumerate(samples):
        dataset = entry.get('dataset', 'postgres')
        query = entry.get('query_text', '')
        
        if not query:
            continue
            
        print(f"\nTest {i+1}:")
        print(f"  Dataset: {dataset}")
        print(f"  Query (first 100 chars): {query[:100]}...")
        
        try:
            # Try to connect to the database
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                user='wuy',
                database=dataset
            )
            conn.autocommit = True
            
            with conn.cursor() as cur:
                # Try different routing methods
                for method_name, method_code in [('default', 0), ('cost_threshold', 1)]:
                    try:
                        cur.execute(f"SET aqd.routing_method = {method_code};")
                        
                        # Try to execute the query with a timeout
                        cur.execute("SET statement_timeout = '5000ms';")
                        cur.execute(query)
                        results = cur.fetchall()
                        print(f"    ✓ {method_name}: Success ({len(results)} rows)")
                        
                    except psycopg2.Error as e:
                        print(f"    ✗ {method_name}: {e.pgerror if hasattr(e, 'pgerror') else str(e)[:100]}")
                    except Exception as e:
                        print(f"    ✗ {method_name}: {str(e)[:100]}")
            
            conn.close()
            
        except psycopg2.OperationalError as e:
            print(f"    ✗ Connection failed: {str(e)[:100]}")
        except Exception as e:
            print(f"    ✗ Unexpected error: {str(e)[:100]}")
    
    print("\n" + "=" * 80)
    print("Common failure patterns:")
    print("1. Connection failures - database doesn't exist")
    print("2. Syntax errors - query malformed")
    print("3. Missing tables/columns - schema mismatch")
    print("4. Timeout - query takes too long")
    print("5. pg_duckdb errors - extension issues")

if __name__ == "__main__":
    test_queries()