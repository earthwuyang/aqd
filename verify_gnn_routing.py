#!/usr/bin/env python3
"""
Verify that kernel GNN inference is working correctly.
Tests routing decisions and measures inference latency.
"""

import psycopg2
import json
import time
from typing import Dict, List, Tuple

def connect_postgres():
    """Connect to PostgreSQL with correct configuration."""
    return psycopg2.connect(
        host='localhost',
        port=5432,
        user='wuy',
        password='',
        database='financial'
    )

def test_routing_method(conn, method: int, method_name: str, queries: List[str]) -> Dict:
    """Test a specific routing method and collect metrics."""
    # Start fresh connection for each method
    conn.rollback()  # Clear any pending transaction
    cur = conn.cursor()
    
    try:
        # Set routing method
        cur.execute(f"SET aqd.routing_method = {method};")
        
        # Set model paths if needed
        if method == 2:  # LightGBM
            cur.execute("SET aqd.lightgbm_model_path = '/home/wuy/DB/pg_duckdb_postgres_2/models/lightgbm_model.txt';")
            cur.execute("SET aqd.lightgbm_library_path = '/home/wuy/DB/pg_duckdb_postgres_2/install/lib/libaqd_lgbm.so';")
        elif method == 3:  # GNN
            cur.execute("SET aqd.gnn_model_path = '/home/wuy/DB/pg_duckdb_postgres_2/models/gnn_model.pth';")
            cur.execute("SET aqd.gnn_library_path = '/home/wuy/DB/pg_duckdb_postgres_2/install/lib/libaqd_gnn.so';")
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  Error setting up method: {e}")
        cur.close()
        return {'method': method_name, 'postgres_count': 0, 'duckdb_count': 0, 
                'total_latency_us': 0, 'decisions': []}
    
    results = {
        'method': method_name,
        'postgres_count': 0,
        'duckdb_count': 0,
        'total_latency_us': 0,
        'decisions': []
    }
    
    for query in queries:
        try:
            # Execute query
            start = time.time()
            cur.execute(query)
            cur.fetchall()
            exec_time = (time.time() - start) * 1000  # ms
            conn.commit()  # Commit after successful query
            
            # Get routing decision
            cur.execute("SHOW aqd.last_routed_engine;")
            engine_str = cur.fetchone()[0]
            
            cur.execute("SHOW aqd.last_decision_us;")
            latency_us = float(cur.fetchone()[0])
            
            # Record results
            if engine_str.lower() == 'postgres':
                results['postgres_count'] += 1
                engine_name = 'PostgreSQL'
            elif engine_str.lower() == 'duckdb':
                results['duckdb_count'] += 1
                engine_name = 'DuckDB'
            else:
                engine_name = engine_str
            
            results['total_latency_us'] += latency_us
            results['decisions'].append({
                'query': query[:50] + '...' if len(query) > 50 else query,
                'engine': engine_name,
                'latency_us': latency_us,
                'exec_time_ms': exec_time
            })
            
        except Exception as e:
            conn.rollback()  # Rollback on error
            error_msg = str(e)
            print(f"    Query error: {error_msg[:100]}")
            results['decisions'].append({
                'query': query[:50] + '...' if len(query) > 50 else query,
                'error': error_msg[:100]
            })
    
    cur.close()
    return results

def main():
    """Main verification routine."""
    print("=== GNN Routing Verification ===\n")
    
    # Test queries - mix of simple and complex
    test_queries = [
        # Simple aggregations (likely DuckDB)
        "SELECT COUNT(*) FROM loan",
        "SELECT AVG(amount) FROM loan WHERE duration > 12",
        "SELECT status, COUNT(*) FROM loan GROUP BY status",
        
        # Point queries (likely PostgreSQL)
        "SELECT * FROM loan WHERE loan_id = 1000",
        "SELECT * FROM account WHERE account_id = 500",
        
        # Complex analytics (likely DuckDB)
        """SELECT 
            l.status, 
            AVG(l.amount) as avg_amount,
            COUNT(*) as count
        FROM loan l
        JOIN account a ON l.account_id = a.account_id
        GROUP BY l.status
        HAVING COUNT(*) > 10""",
        
        # Mixed workload
        "SELECT * FROM district WHERE district_id BETWEEN 10 AND 20",
        "SELECT SUM(amount), MIN(amount), MAX(amount) FROM loan",
    ]
    
    try:
        conn = connect_postgres()
        
        # Test each routing method
        methods = [
            (0, 'Default'),
            (1, 'Cost Threshold'),
            (2, 'LightGBM'),
            (3, 'GNN')
        ]
        
        all_results = []
        for method_code, method_name in methods:
            print(f"\nTesting {method_name} routing (method={method_code})...")
            results = test_routing_method(conn, method_code, method_name, test_queries)
            all_results.append(results)
            
            # Show summary
            total_queries = len(test_queries)
            successful = len([d for d in results['decisions'] if 'error' not in d])
            
            print(f"  Successful: {successful}/{total_queries}")
            print(f"  PostgreSQL: {results['postgres_count']}")
            print(f"  DuckDB: {results['duckdb_count']}")
            if successful > 0:
                avg_latency = results['total_latency_us'] / successful
                print(f"  Avg routing latency: {avg_latency:.2f} μs")
        
        # Compare routing decisions
        print("\n=== Routing Decision Comparison ===")
        print("Query | Default | Cost | LightGBM | GNN")
        print("-" * 50)
        
        for i, query in enumerate(test_queries):
            query_snippet = query[:30] + '...' if len(query) > 30 else query
            decisions = []
            for result in all_results:
                if i < len(result['decisions']) and 'error' not in result['decisions'][i]:
                    decisions.append(result['decisions'][i]['engine'][0])  # P or D
                else:
                    decisions.append('E')  # Error
            print(f"{query_snippet:<30} | {' | '.join(decisions)}")
        
        # Check if GNN makes different decisions than default
        default_decisions = [d['engine'] for d in all_results[0]['decisions'] if 'error' not in d]
        gnn_decisions = [d['engine'] for d in all_results[3]['decisions'] if 'error' not in d]
        
        if default_decisions and gnn_decisions:
            different = sum(1 for d, g in zip(default_decisions, gnn_decisions) if d != g)
            print(f"\n✓ GNN makes {different}/{len(default_decisions)} different routing decisions vs Default")
            
            if different == 0:
                print("⚠️ WARNING: GNN routing identical to Default - inference may not be working!")
            else:
                print("✓ GNN inference appears to be working correctly")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()