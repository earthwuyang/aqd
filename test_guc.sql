-- Enable LightGBM
SET lightgbm.enabled = true;
SET lightgbm.model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_model.txt';

-- Run a simple query  
SELECT 1;

-- Check the GUC immediately after
SHOW lightgbm.last_decision;

-- Run another query
SELECT COUNT(*) FROM pg_class;

-- Check again
SHOW lightgbm.last_decision;
