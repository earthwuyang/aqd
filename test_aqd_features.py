#!/usr/bin/env python3
"""
Test AQD feature collection from PostgreSQL kernel.
"""

import psycopg2
import json
import time
from pathlib import Path

# PostgreSQL connection config
PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'wuy',
    'database': 'tpch_sf1'  # Use TPC-H database
}

def enable_aqd_features(conn):
    """Enable AQD feature logging."""
    cursor = conn.cursor()
    try:
        # Enable AQD feature logging using registered GUC variables
        cursor.execute("SET aqd.enable_feature_logging = true;")
        cursor.execute("SET aqd.log_format = 1;")  # JSON format
        cursor.execute("SET aqd.feature_log_path = '/tmp/aqd_features.json';")
        
        # Verify settings
        cursor.execute("SHOW aqd.enable_feature_logging;")
        enabled = cursor.fetchone()[0]
        
        cursor.execute("SHOW aqd.log_format;")
        format = cursor.fetchone()[0]
        
        cursor.execute("SHOW aqd.feature_log_path;")
        path = cursor.fetchone()[0]
        
        print(f"✓ AQD feature logging enabled: {enabled}")
        print(f"  - Log format: {'JSON' if format == '1' else 'CSV'}")
        print(f"  - Log path: {path}")
    except Exception as e:
        print(f"✗ Failed to enable AQD features: {e}")
        print("  Note: Using system default PostgreSQL, not our modified version")
    finally:
        cursor.close()

def run_test_queries(conn):
    """Run some test queries to generate features."""
    cursor = conn.cursor()
    
    test_queries = [
        # Simple SELECT
        "SELECT * FROM nation LIMIT 10",
        
        # JOIN query  
        """
        SELECT n.col2, COUNT(s.col1) as supplier_count
        FROM nation n
        LEFT JOIN supplier s ON n.col1 = s.col4
        GROUP BY n.col2
        ORDER BY supplier_count DESC
        LIMIT 10
        """,
        
        # Aggregation query on lineitem
        """
        SELECT col15, COUNT(*) as order_count, AVG(col5) as avg_quantity
        FROM lineitem
        WHERE col11 >= '1995-01-01' AND col11 < '1996-01-01'
        GROUP BY col15
        ORDER BY order_count DESC
        """,
        
        # Complex aggregation
        """
        SELECT col9, col10,
               SUM(col5) as sum_qty,
               SUM(col6) as sum_price,
               COUNT(*) as count_order
        FROM lineitem
        WHERE col11 <= '1998-09-02'
        GROUP BY col9, col10
        ORDER BY col9, col10
        """
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n📝 Running test query {i}...")
            print(f"   Query: {query[:60]}..." if len(query) > 60 else f"   Query: {query}")
            
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            exec_time = (time.time() - start_time) * 1000
            
            print(f"   ✓ Executed in {exec_time:.2f}ms, returned {len(results)} rows")
            
        except Exception as e:
            print(f"   ✗ Query failed: {e}")
    
    cursor.close()

def check_feature_log():
    """Check if features were logged."""
    log_path = Path("/tmp/aqd_features.json")
    
    if not log_path.exists():
        print("\n⚠️ No feature log file found at /tmp/aqd_features.json")
        return
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Try to parse as JSON lines
        lines = content.strip().split('\n')
        features = []
        for line in lines:
            if line.strip():
                try:
                    features.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        
        print(f"\n📊 Feature Log Summary:")
        print(f"   - Log file size: {log_path.stat().st_size} bytes")
        print(f"   - Number of logged queries: {len(features)}")
        
        if features:
            # Show sample of first feature
            first = features[0]
            print(f"\n   Sample feature keys from first query:")
            for key in list(first.keys())[:10]:
                if key == 'query_plan' and isinstance(first[key], dict):
                    print(f"     - {key}: <plan tree with {len(str(first[key]))} chars>")
                elif key == 'features' and isinstance(first[key], list):
                    print(f"     - {key}: {len(first[key])} features extracted")
                else:
                    value = str(first[key])[:50] + "..." if len(str(first[key])) > 50 else str(first[key])
                    print(f"     - {key}: {value}")
                    
    except Exception as e:
        print(f"\n✗ Error reading feature log: {e}")

def main():
    print("🚀 Testing AQD Feature Collection\n")
    
    # Clear existing log
    log_path = Path("/tmp/aqd_features.json")
    if log_path.exists():
        log_path.unlink()
        print("✓ Cleared existing feature log")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**PG_CONFIG)
        conn.autocommit = True
        print(f"✓ Connected to PostgreSQL database: {PG_CONFIG['database']}")
        
        # Enable AQD features
        enable_aqd_features(conn)
        
        # Run test queries
        run_test_queries(conn)
        
        # Close connection
        conn.close()
        
        # Check the feature log
        check_feature_log()
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()