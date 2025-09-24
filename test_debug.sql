SET log_min_messages = DEBUG2;
SET client_min_messages = DEBUG2;
SET lightgbm.enabled = true;
SET lightgbm.model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_model.txt';
SELECT 1;
SHOW lightgbm.last_decision;
