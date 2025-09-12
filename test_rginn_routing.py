#!/usr/bin/env python3
"""
Test R-GIN query routing in PostgreSQL kernel.
This script tests whether queries are being routed to PostgreSQL or DuckDB
based on R-GIN model predictions.
"""

import psycopg2
import json
import os
import sys
import time

def test_query_routing():
    """Test R-GIN query routing with sample queries."""
    
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tpch_sf1",
        user="wuy"
    )
    
    # Test queries - mix of AP and TP workloads
    test_queries = [
        # Simple TP query (should favor PostgreSQL)
        {
            "type": "TP",
            "query": "SELECT l_orderkey FROM lineitem WHERE l_orderkey = 1"
        },
        # Complex AP query (should favor DuckDB)
        {
            "type": "AP",
            "query": """
                SELECT
                    l_returnflag,
                    l_linestatus,
                    SUM(l_quantity) as sum_qty,
                    SUM(l_extendedprice) as sum_base_price,
                    AVG(l_quantity) as avg_qty,
                    AVG(l_extendedprice) as avg_price,
                    AVG(l_discount) as avg_disc,
                    COUNT(*) as count_order
                FROM
                    lineitem
                WHERE
                    l_shipdate <= date '1998-12-01' - interval '90' day
                GROUP BY
                    l_returnflag,
                    l_linestatus
                ORDER BY
                    l_returnflag,
                    l_linestatus
            """
        },
        # Medium complexity query
        {
            "type": "Mixed",
            "query": """
                SELECT
                    c_custkey,
                    c_name,
                    SUM(l_extendedprice * (1 - l_discount)) as revenue
                FROM
                    customer,
                    orders,
                    lineitem
                WHERE
                    c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND o_orderdate >= date '1993-10-01'
                    AND o_orderdate < date '1993-10-01' + interval '3' month
                    AND l_returnflag = 'R'
                    AND c_nationkey = 1
                GROUP BY
                    c_custkey,
                    c_name
                ORDER BY
                    revenue DESC
                LIMIT 20
            """
        },
        # Another simple TP query
        {
            "type": "TP",
            "query": "SELECT c_name, c_phone FROM customer WHERE c_custkey = 100"
        },
        # Aggregation query
        {
            "type": "AP",
            "query": """
                SELECT
                    n_name,
                    SUM(l_extendedprice * (1 - l_discount)) as revenue
                FROM
                    customer,
                    orders,
                    lineitem,
                    supplier,
                    nation,
                    region
                WHERE
                    c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND l_suppkey = s_suppkey
                    AND c_nationkey = s_nationkey
                    AND s_nationkey = n_nationkey
                    AND n_regionkey = r_regionkey
                    AND r_name = 'ASIA'
                    AND o_orderdate >= date '1994-01-01'
                    AND o_orderdate < date '1994-01-01' + interval '1' year
                GROUP BY
                    n_name
                ORDER BY
                    revenue DESC
            """
        }
    ]
    
    print("Testing R-GIN Query Routing")
    print("=" * 60)
    print()
    
    # Check server log for routing decisions
    print("Note: Check PostgreSQL log for routing decisions")
    print("Look for messages like:")
    print("  - 'R-GIN routed query to DuckDB'")
    print("  - 'R-GIN routed query to PostgreSQL'")
    print()
    
    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}: {test['type']} Query")
        print("-" * 40)
        
        cursor = conn.cursor()
        
        try:
            # Execute query and time it
            start_time = time.time()
            cursor.execute(test['query'])
            
            # Fetch results (if any)
            if cursor.description:
                results = cursor.fetchall()
                num_rows = len(results)
            else:
                num_rows = cursor.rowcount
                
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            print(f"✓ Query executed successfully")
            print(f"  Rows: {num_rows}")
            print(f"  Time: {execution_time:.2f} ms")
            
            # Check if query was routed (would need to parse logs or add instrumentation)
            # For now, we just note to check the logs
            print(f"  Expected routing: {'DuckDB' if test['type'] == 'AP' else 'PostgreSQL'}")
            
        except Exception as e:
            print(f"✗ Query failed: {e}")
        finally:
            cursor.close()
        
        print()
    
    conn.close()
    
    print("=" * 60)
    print("Test complete!")
    print()
    print("To verify routing decisions, check PostgreSQL log:")
    print("  tail -f /home/wuy/DB/pg_duckdb_postgres/data/log/*.log")
    print()
    print("Or check system log:")
    print("  sudo journalctl -u postgresql -f")
    print()
    print("Look for R-GIN routing messages to confirm the model is making decisions.")

if __name__ == "__main__":
    try:
        test_query_routing()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)