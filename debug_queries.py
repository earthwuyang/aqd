#!/usr/bin/env python3
"""Debug script to test individual queries and see actual errors"""

import psycopg2
import sys

def test_single_query():
    # Test query from the benchmark
    query = '''SELECT "part"."p_retailprice", "partsupp"."ps_supplycost"
FROM "part" LEFT JOIN "partsupp" ON "part"."p_partkey" = "partsupp"."ps_partkey"
LEFT JOIN "supplier" ON "partsupp"."ps_suppkey" = "supplier"."s_suppkey"
WHERE "part"."p_type" = 'SMALL PLATED NICKEL' LIMIT 23'''

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname='tpch_sf1',
        user='wuy',
        host='localhost',
        port=5432
    )
    conn.autocommit = False

    print("Testing query with different configurations...")

    # Test 1: PostgreSQL only
    print("\n1. Testing with PostgreSQL only:")
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL duckdb.force_execution = false")
            cur.execute("SET LOCAL lightgbm.enabled = false")
            cur.execute("SET LOCAL statement_timeout = 30000")

            cur.execute(query)
            rows = cur.fetchall()
            print(f"SUCCESS: {len(rows)} rows returned")

    except Exception as e:
        print(f"ERROR: {e}")

    conn.rollback()

    # Test 2: DuckDB only
    print("\n2. Testing with DuckDB only:")
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL duckdb.force_execution = true")
            cur.execute("SET LOCAL lightgbm.enabled = false")
            cur.execute("SET LOCAL statement_timeout = 30000")

            cur.execute(query)
            rows = cur.fetchall()
            print(f"SUCCESS: {len(rows)} rows returned")

    except Exception as e:
        print(f"ERROR: {e}")

    conn.rollback()

    # Test 3: LightGBM routing
    print("\n3. Testing with LightGBM routing:")
    try:
        with conn.cursor() as cur:
            # Check if LightGBM GUCs exist
            try:
                cur.execute("SHOW lightgbm.enabled")
                lgbm_enabled = cur.fetchone()[0]
                print(f"lightgbm.enabled = {lgbm_enabled}")
            except Exception as e:
                print(f"lightgbm.enabled GUC not found: {e}")
                return

            cur.execute("SET LOCAL lightgbm.enabled = true")
            cur.execute("SET LOCAL statement_timeout = 30000")

            cur.execute(query)
            rows = cur.fetchall()
            print(f"SUCCESS: {len(rows)} rows returned")

            # Try to get routing decision
            try:
                cur.execute("SHOW lightgbm.last_routed_engine")
                decision = cur.fetchone()[0]
                print(f"Routed to: {decision}")
            except Exception as e:
                print(f"Could not get routing decision: {e}")

    except Exception as e:
        print(f"ERROR: {e}")

    conn.rollback()
    conn.close()

if __name__ == "__main__":
    test_single_query()