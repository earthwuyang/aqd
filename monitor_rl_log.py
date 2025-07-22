#!/usr/bin/env python3
"""
test_rl_logging.py - Simple test to verify RL logging is working
"""

import pymysql
import time
import sys

# Connection parameters
HOST = "127.0.0.1"
PORT = 44444
USER = "root"
PASS = ""

def test_rl_logging():
    """Run a few queries with lightgbm_dynamic mode to trigger RL logging"""
    
    print("Testing RL logging with lightgbm_dynamic mode...")
    print("-" * 60)
    
    try:
        # Connect to MySQL
        conn = pymysql.connect(
            host=HOST, port=PORT, user=USER, password=PASS,
            autocommit=True, charset="utf8mb4"
        )
        cursor = conn.cursor()
        
        # Enable lightgbm_dynamic mode
        print("Setting up lightgbm_dynamic mode...")
        setup_queries = [
            "SET use_imci_engine = ON",
            "SET cost_threshold_for_imci = 1",
            "SET hybrid_opt_dispatch_enabled = ON",
            "SET fann_model_routing_enabled = ON",
            "SET use_mm1_time = ON",  # This enables LinUCB
        ]
        
        for query in setup_queries:
            cursor.execute(query)
            print(f"  ✓ {query}")
        
        # Verify settings
        print("\nVerifying settings...")
        cursor.execute("SHOW VARIABLES LIKE 'use_mm1_time'")
        result = cursor.fetchone()
        if result:
            print(f"  use_mm1_time = {result[1]}")
        
        # Use a test database
        test_db = "tpch_sf1"  # Adjust if needed
        cursor.execute(f"USE {test_db}")
        print(f"\nUsing database: {test_db}")
        
        # Run some test queries
        test_queries = [
            # Simple query (likely row-oriented)
            "SELECT COUNT(*) FROM lineitem WHERE l_orderkey < 1000",
            
            # Aggregation query (likely column-oriented)
            "SELECT l_returnflag, l_linestatus, SUM(l_quantity) as sum_qty "
            "FROM lineitem "
            "GROUP BY l_returnflag, l_linestatus "
            "ORDER BY l_returnflag, l_linestatus",
            
            # Another simple query
            "SELECT * FROM orders WHERE o_orderkey = 1",
            
            # Complex aggregation
            "SELECT COUNT(*), AVG(l_extendedprice), MAX(l_discount) "
            "FROM lineitem "
            "WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'",
        ]
        
        print(f"\nRunning {len(test_queries)} test queries...")
        for i, query in enumerate(test_queries):
            print(f"\nQuery {i+1}:")
            print(f"  {query[:60]}...")
            
            start_time = time.time()
            cursor.execute(query)
            result = cursor.fetchall()
            elapsed = time.time() - start_time
            
            print(f"  ✓ Completed in {elapsed:.3f}s (returned {len(result)} rows)")
            
            # Small delay between queries
            time.sleep(0.1)
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*60)
        print("Test completed!")
        print("\nNow check the MySQL error log for [RL] entries:")
        print("  sudo grep '\\[RL\\]' /home/wuy/mypolardb/db/log/master-error.log | tail -20")
        print("\nOr run the diagnostic script:")
        print("  ./diagnose_rl_logs.sh")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MySQL is running on port 44444")
        print("2. Check that the test database exists")
        print("3. Verify the new C++ code is compiled and deployed")
        return False
    
    return True

if __name__ == "__main__":
    # Allow database override
    if len(sys.argv) > 1:
        test_db = sys.argv[1]
        print(f"Using database: {test_db}")
    
    test_rl_logging()