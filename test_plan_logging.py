#!/usr/bin/env python3
"""
Test the plan_logger extension by running sample queries
and displaying the logged plans
"""

import json
import psycopg2
import time
from pathlib import Path

def run_test_queries():
    """Run test queries to generate plan logs"""
    conn = psycopg2.connect(
        host="/tmp",
        port=5432,
        database="financial"
    )
    cur = conn.cursor()
    
    # Don't clear the log file - just note its current size
    log_file = Path("/tmp/postgres_plans.jsonl")
    initial_size = log_file.stat().st_size if log_file.exists() else 0
    
    print("Running test queries...")
    
    # Simple query
    print("1. Simple count query")
    cur.execute("SELECT COUNT(*) FROM trans WHERE amount > 1000")
    result = cur.fetchone()
    print(f"   Result: {result[0]} rows")
    
    # Join query
    print("2. Join query")
    cur.execute("""
        SELECT a.account_id, COUNT(t.trans_id) 
        FROM account a 
        JOIN trans t ON a.account_id = t.account_id 
        WHERE t.amount > 500 
        GROUP BY a.account_id 
        LIMIT 10
    """)
    result = cur.fetchall()
    print(f"   Result: {len(result)} groups")
    
    # Complex analytical query
    print("3. Analytical query with window function")
    cur.execute("""
        SELECT 
            account_id,
            date,
            amount,
            SUM(amount) OVER (PARTITION BY account_id ORDER BY date) as running_total
        FROM trans
        WHERE date >= '1995-01-01' AND date < '1995-02-01'
        LIMIT 20
    """)
    result = cur.fetchall()
    print(f"   Result: {len(result)} rows")
    
    conn.close()
    
    # Wait for logs to be written
    time.sleep(1)
    
    print("\nParsing logged plans...")
    
    # Read and parse the log file
    if log_file.exists():
        content = log_file.read_text()
        
        # Try to parse as JSONL (one JSON per line)
        lines = content.strip().split('\n')
        plans = []
        
        # Try different parsing strategies
        # Strategy 1: Each line is a complete JSON
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    plan = json.loads(line)
                    plans.append(plan)
                except json.JSONDecodeError:
                    pass
        
        if plans:
            print(f"\nSuccessfully parsed {len(plans)} plans")
            for i, plan in enumerate(plans[-3:], 1):  # Show last 3 plans
                print(f"\n--- Plan {i} ---")
                print(f"Database: {plan.get('database', 'N/A')}")
                print(f"Timestamp: {plan.get('timestamp', 'N/A')}")
                if 'plan' in plan and isinstance(plan['plan'], list) and plan['plan']:
                    if 'Plan' in plan['plan'][0]:
                        print(f"Node Type: {plan['plan'][0]['Plan'].get('Node Type', 'N/A')}")
                        print(f"Total Cost: {plan['plan'][0]['Plan'].get('Total Cost', 'N/A')}")
        else:
            # If JSONL parsing failed, try to extract JSON objects manually
            print("\nTrying to extract JSON objects from log...")
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches[:3]:  # Show first 3 matches
                try:
                    obj = json.loads(match)
                    if 'timestamp' in obj or 'plan' in obj:
                        print(f"\nFound plan object:")
                        print(json.dumps(obj, indent=2)[:500] + "...")
                except:
                    pass
    else:
        print("Log file not found!")

if __name__ == "__main__":
    run_test_queries()