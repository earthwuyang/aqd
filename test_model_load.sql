-- First ensure LightGBM is disabled
SET lightgbm.enabled = false;
SELECT 'Testing with LightGBM disabled' as test_status;

-- Test simple query
SELECT 1 as simple_test;

-- Now enable and check model path
SET lightgbm.enabled = true;
SET lightgbm.model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model.txt';
SELECT 'Model path set' as status;

-- Try to check if model loads
SHOW lightgbm.model_path;
