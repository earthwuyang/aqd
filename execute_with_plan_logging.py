#!/usr/bin/env python3
"""
Execute queries with JSON plan logging
Mimics aqd.enable_feature_logging functionality
"""

import json
import psycopg2
import sys
import os
from datetime import datetime
from pathlib import Path
import hashlib
import argparse

class PlanLogger:
    def __init__(self, database, log_path="/tmp/postgres_plans.jsonl"):
        self.database = database
        self.log_path = log_path
        self.conn = None
        self.enabled = True
        
    def connect(self):
        """Connect to PostgreSQL database"""
        self.conn = psycopg2.connect(
            host="/tmp",
            port=5432,
            database=self.database
        )
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def get_query_hash(self, query):
        """Generate hash for normalized query"""
        # Simple normalization: lowercase and remove extra spaces
        normalized = ' '.join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def get_json_plan(self, query):
        """Get JSON execution plan for a query"""
        cur = self.conn.cursor()
        
        # Get EXPLAIN plan in JSON format
        explain_query = f"EXPLAIN (FORMAT JSON, VERBOSE false, COSTS true, BUFFERS false) {query}"
        cur.execute(explain_query)
        result = cur.fetchone()
        
        if result and result[0]:
            return result[0]
        return None
    
    def log_plan(self, query, plan_json, execution_time_ms=None):
        """Log plan to JSONL file"""
        if not self.enabled:
            return
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "database": self.database,
            "query_hash": self.get_query_hash(query),
            "query_text": query,
            "plan": plan_json
        }
        
        if execution_time_ms is not None:
            entry["execution_time_ms"] = execution_time_ms
        
        # Append to JSONL file
        with open(self.log_path, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
    
    def execute_with_logging(self, query):
        """Execute query and log its plan"""
        cur = self.conn.cursor()
        
        # Get the plan
        plan_json = self.get_json_plan(query)
        
        # Execute the query and measure time
        import time
        start_time = time.time()
        cur.execute(query)
        result = cur.fetchall()
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Log the plan
        if plan_json:
            self.log_plan(query, plan_json, execution_time_ms)
        
        return result
    
    def execute_file(self, sql_file):
        """Execute all queries from a SQL file with logging"""
        with open(sql_file, 'r') as f:
            content = f.read()
        
        # Split by semicolons (simple approach)
        queries = [q.strip() for q in content.split(';') if q.strip()]
        
        results = []
        for i, query in enumerate(queries, 1):
            if query.lower().startswith(('select', 'with', 'explain')):
                print(f"Executing query {i}/{len(queries)}...")
                try:
                    result = self.execute_with_logging(query)
                    results.append((query, result))
                    print(f"  ✓ Logged plan for query {i}")
                except Exception as e:
                    print(f"  ✗ Error in query {i}: {e}")
                    
        return results

def main():
    parser = argparse.ArgumentParser(description='Execute queries with JSON plan logging')
    parser.add_argument('--database', '-d', required=True, help='Database name')
    parser.add_argument('--file', '-f', help='SQL file to execute')
    parser.add_argument('--query', '-q', help='Single query to execute')
    parser.add_argument('--log-path', default='/tmp/postgres_plans.jsonl', 
                       help='Path for plan log file')
    parser.add_argument('--enable', action='store_true', default=True,
                       help='Enable plan logging (default: enabled)')
    parser.add_argument('--disable', action='store_true',
                       help='Disable plan logging')
    
    args = parser.parse_args()
    
    # Create logger
    logger = PlanLogger(args.database, args.log_path)
    logger.enabled = not args.disable
    
    try:
        logger.connect()
        
        if args.file:
            print(f"Executing queries from {args.file}...")
            print(f"Logging plans to {args.log_path}")
            results = logger.execute_file(args.file)
            print(f"\nExecuted {len(results)} queries")
            
        elif args.query:
            print(f"Executing single query...")
            result = logger.execute_with_logging(args.query)
            print(f"Result: {len(result)} rows")
            if logger.enabled:
                print(f"Plan logged to {args.log_path}")
                
        else:
            # Interactive mode - read queries from stdin
            print("Enter queries (Ctrl+D to exit):")
            while True:
                try:
                    query = input("sql> ")
                    if query.strip():
                        result = logger.execute_with_logging(query)
                        print(f"Result: {len(result)} rows")
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
    finally:
        logger.close()
    
    print(f"\nPlans logged to: {args.log_path}")
    print("To view plans: cat {} | jq .".format(args.log_path))

if __name__ == "__main__":
    main()