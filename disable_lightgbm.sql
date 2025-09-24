ALTER SYSTEM SET lightgbm.enabled = false;
ALTER SYSTEM SET lightgbm.model_path = '';
SELECT pg_reload_conf();
