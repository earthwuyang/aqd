#!/usr/bin/env python3
"""
Test script to demonstrate the fixed evaluation system measures true server-side performance
"""

import psycopg2
import time
import statistics

def test_routing_overhead():
    """Test routing overhead with the new GUCs"""
    conn = psycopg2.connect(host='localhost', port=5432, user='wuy', database='postgres')
    conn.autocommit = True
    
    methods = {
        'default': 0,
        'cost_threshold': 1, 
        'lightgbm': 2,
        'gnn': 3
    }
    
    print("\n=== Testing Server-Side Routing Overhead ===\n")
    
    for method_name, method_code in methods.items():
        routing_times = []
        
        with conn.cursor() as cur:
            # Set routing method
            cur.execute(f"SET aqd.routing_method = {method_code};")
            if method_name == 'cost_threshold':
                cur.execute("SET aqd.cost_threshold = 1000.0;")
            
            # Run multiple queries to measure overhead
            for i in range(10):
                # Simple query to isolate routing overhead
                cur.execute("SELECT count(*) FROM pg_class;")
                _ = cur.fetchall()
                
                # Get routing decision time from kernel
                cur.execute("SHOW aqd.last_decision_us;")
                route_us = int(cur.fetchone()[0])
                routing_times.append(route_us)
                
                # Also check which engine was chosen
                cur.execute("SHOW aqd.last_routed_engine;")
                engine = cur.fetchone()[0]
                
                if i == 0:
                    print(f"{method_name:14s}: First routing to '{engine}'")
        
        if routing_times:
            avg_us = statistics.mean(routing_times)
            p95_us = statistics.quantiles(routing_times, n=20)[18] if len(routing_times) > 1 else routing_times[0]
            print(f"                 Avg routing overhead: {avg_us:.1f} µs, P95: {p95_us:.1f} µs")
    
    print("\n=== Key Improvements Made ===")
    print("1. Removed client-side ML prediction timing from latency measurements")
    print("2. All routing decisions now happen server-side in the kernel") 
    print("3. Added GUCs to track actual routing overhead (aqd.last_decision_us)")
    print("4. Disabled one-sided feature logging during benchmarks")
    print("5. Set proper model paths so kernel can load LightGBM/GNN models")
    print("6. Connect to correct per-dataset databases (not just 'postgres')")
    print("7. Interleave methods to avoid cache bias")
    
    conn.close()

if __name__ == "__main__":
    test_routing_overhead()