-- Simple test to check routing
SET lightgbm.enabled = true;
SET lightgbm.model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model.txt';

-- Test AP query (should route to DuckDB)
SELECT 'AP Query Test' as test_type;
SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= '1998-01-01';
SHOW lightgbm.last_decision;

-- Test TP query (should route to PostgreSQL)
SELECT 'TP Query Test' as test_type;
SELECT * FROM lineitem WHERE l_orderkey = 1;
SHOW lightgbm.last_decision;
