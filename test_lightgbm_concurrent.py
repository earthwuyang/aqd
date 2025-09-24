#!/usr/bin/env python3
"""
Test LightGBM routing with concurrent queries.
The model path is already set in postgresql.conf, so we don't set it per-connection.
"""

import psycopg2
import concurrent.futures
import time
import sys

def run_query(args):
    db_name, query_id, query = args
    try:
        # Connect to database
        conn = psycopg2.connect(host='/tmp', dbname=db_name, user='wuy')
        cur = conn.cursor()

        # Don't set model path - it's already in postgresql.conf
        # Just enable LightGBM if needed
        cur.execute("SET lightgbm.enabled = true")

        # Run the query
        start = time.time()
        cur.execute(query)
        result = cur.fetchone()
        elapsed = (time.time() - start) * 1000

        # Check routing decision (might not be available if GUC isn't registered)
        try:
            cur.execute("SHOW lightgbm.last_routed_engine")
            routed_to = cur.fetchone()[0]
        except Exception as e:
            routed_to = "unknown"

        cur.close()
        conn.close()

        return {
            'db': db_name,
            'query_id': query_id,
            'success': True,
            'time_ms': elapsed,
            'routed_to': routed_to,
            'result': result[0] if result else None
        }
    except Exception as e:
        return {
            'db': db_name,
            'query_id': query_id,
            'success': False,
            'error': str(e)
        }

def main():
    # Test queries
    test_queries = [
        ('postgres', 1, 'SELECT COUNT(*) FROM pg_class'),
        ('postgres', 2, 'SELECT COUNT(*) FROM pg_attribute'),
        ('postgres', 3, 'SELECT COUNT(*) FROM pg_namespace'),
        ('postgres', 4, 'SELECT relname, COUNT(*) FROM pg_class GROUP BY relname LIMIT 5'),
        ('postgres', 5, 'SELECT 1'),
    ]

    print("Testing concurrent queries with LightGBM (model pre-loaded from postgresql.conf)")
    print(f"Running {len(test_queries)} queries concurrently...")

    # Run queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_query, args) for args in test_queries]

        results = []
        for future in concurrent.futures.as_completed(futures, timeout=10):
            result = future.result()
            results.append(result)

            if result['success']:
                print(f"✓ Query {result['query_id']}: {result['time_ms']:.2f}ms (routed to {result['routed_to']})")
            else:
                print(f"✗ Query {result['query_id']}: {result['error']}")

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nResults: {successful}/{len(test_queries)} queries completed successfully")

    if successful == len(test_queries):
        print("SUCCESS: All queries completed with LightGBM enabled!")
        return 0
    else:
        print("FAILURE: Some queries failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())