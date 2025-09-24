import psycopg2
import time

# Connect to database
conn = psycopg2.connect(
    dbname='tpch_sf1',
    user='wuy',
    host='localhost',
    port=5432
)

with conn.cursor() as cur:
    # Enable LightGBM
    cur.execute("SET lightgbm.enabled = true")
    cur.execute("SET lightgbm.model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model.txt'")
    
    # Run a query
    print("Running query...")
    cur.execute("SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '1998-01-01'")
    result = cur.fetchone()
    print(f"Query result: {result}")
    
    # Check routing decision immediately
    cur.execute("SHOW lightgbm.last_decision")
    decision = cur.fetchone()
    print(f"Routing decision: {decision}")
    
    # Check decision time
    cur.execute("SHOW lightgbm.last_decision_us")
    decision_time = cur.fetchone()
    print(f"Decision time: {decision_time} us")

conn.close()
