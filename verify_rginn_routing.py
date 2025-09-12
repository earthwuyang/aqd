#!/usr/bin/env python3
"""
Verify R-GIN routing is working by checking system logs and GNN plan logs.
"""

import json
import os
import sys
import glob
import psycopg2
import time
from datetime import datetime

def check_latest_plan_logs():
    """Check the latest GNN plan log files for routing information."""
    log_dir = "/tmp/pg_gnn_plans"
    
    if not os.path.exists(log_dir):
        print("⚠️  GNN plan log directory doesn't exist")
        return False
    
    # Get the most recent log files
    log_files = sorted(glob.glob(f"{log_dir}/plans_*.jsonl"), 
                      key=os.path.getmtime, reverse=True)
    
    if not log_files:
        print("⚠️  No GNN plan log files found")
        return False
    
    print(f"Found {len(log_files)} log files")
    print(f"Most recent: {os.path.basename(log_files[0])}")
    
    # Check the most recent file
    routing_found = False
    with open(log_files[0], 'r') as f:
        lines = f.readlines()
        if lines:
            # Check last few entries
            for line in lines[-5:]:
                try:
                    entry = json.loads(line)
                    if 'engine_used' in entry or 'gnn_prediction' in entry:
                        print("\n✓ Found routing information in log:")
                        print(f"  Engine used: {entry.get('engine_used', 'N/A')}")
                        print(f"  GNN prediction: {entry.get('gnn_prediction', 'N/A')}")
                        routing_found = True
                except json.JSONDecodeError:
                    continue
    
    return routing_found

def test_routing_with_explain():
    """Test routing by checking EXPLAIN output for DuckDB indicators."""
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tpch_sf1",
        user="wuy"
    )
    
    test_queries = [
        ("Simple TP", "SELECT * FROM customer WHERE c_custkey = 1"),
        ("Complex AP", """
            SELECT 
                l_returnflag, 
                l_linestatus,
                COUNT(*) as cnt
            FROM lineitem 
            WHERE l_shipdate <= date '1998-09-02'
            GROUP BY l_returnflag, l_linestatus
        """)
    ]
    
    print("\n" + "="*60)
    print("Testing Query Routing with EXPLAIN")
    print("="*60)
    
    for query_type, query in test_queries:
        print(f"\n{query_type} Query:")
        print("-" * 40)
        
        cursor = conn.cursor()
        
        try:
            # Get EXPLAIN output
            cursor.execute(f"EXPLAIN {query}")
            explain_output = cursor.fetchall()
            
            # Check if DuckDB is mentioned in the plan
            is_duckdb = any("DuckDB" in str(row) for row in explain_output)
            
            if is_duckdb:
                print("✓ Query routed to DuckDB")
            else:
                print("✓ Query routed to PostgreSQL")
            
            # Show first few lines of explain
            print("\nEXPLAIN output (first 3 lines):")
            for row in explain_output[:3]:
                print(f"  {row[0]}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        finally:
            cursor.close()
    
    conn.close()

def check_rginn_status():
    """Check R-GIN configuration status."""
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="wuy"
    )
    
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("R-GIN Configuration Status")
    print("="*60)
    
    settings = [
        "rginn.enabled",
        "rginn.model_path",
        "rginn.routing_threshold",
        "gnn_plan_logging.enabled",
        "duckdb.force_execution"
    ]
    
    for setting in settings:
        cursor.execute(f"SHOW {setting}")
        value = cursor.fetchone()[0]
        status = "✓" if value not in ["off", ""] else "✗"
        print(f"{status} {setting}: {value}")
    
    # Check if model file exists
    cursor.execute("SHOW rginn.model_path")
    model_path = cursor.fetchone()[0]
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        print(f"\n✓ Model file exists: {model_path} ({model_size} bytes)")
        
        # Show first few lines of model
        with open(model_path, 'r') as f:
            lines = f.readlines()
            print(f"  Model has {len(lines)} lines")
            if lines and lines[0].startswith("RGINN_MODEL"):
                print("  ✓ Valid R-GIN model header found")
    else:
        print(f"\n✗ Model file not found: {model_path}")
    
    cursor.close()
    conn.close()

def main():
    print("R-GIN Routing Verification")
    print("="*60)
    
    # Check configuration
    check_rginn_status()
    
    # Check plan logs
    print("\n" + "="*60)
    print("Checking GNN Plan Logs")
    print("="*60)
    if not check_latest_plan_logs():
        print("⚠️  No routing information found in recent logs")
        print("This might mean:")
        print("  1. R-GIN routing is not active yet")
        print("  2. No queries have been executed since enabling R-GIN")
        print("  3. Logging format has changed")
    
    # Test with EXPLAIN
    test_routing_with_explain()
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)
    print("\nTo see real-time routing decisions, monitor the PostgreSQL log:")
    print("  sudo journalctl -u postgresql -f")
    print("\nOr check GNN plan logs:")
    print("  tail -f /tmp/pg_gnn_plans/plans_*.jsonl | grep -E 'engine_used|gnn_prediction'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)