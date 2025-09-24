-- Enable LightGBM globally
ALTER SYSTEM SET lightgbm.enabled = true;
ALTER SYSTEM SET lightgbm.model_path = '/home/wuy/DB/pg_duckdb_postgres/lightgbm_models/lightgbm_model.txt';
ALTER SYSTEM SET lightgbm.routing_threshold = 0.5;
ALTER SYSTEM SET lightgbm.enable_preopt_feature_extraction = true;

SELECT pg_reload_conf();
