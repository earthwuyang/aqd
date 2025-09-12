-- Configure R-GIN model for query routing
-- Run this after restarting PostgreSQL with the new build

-- Enable R-GIN model
ALTER SYSTEM SET rginn.enabled = on;

-- Set the path to the trained model
ALTER SYSTEM SET rginn.model_path = '/home/wuy/DB/pg_duckdb_postgres/models/rginn_model.txt';

-- Set routing threshold (0.0 means use model's raw prediction)
-- Positive values favor DuckDB, negative favor PostgreSQL
ALTER SYSTEM SET rginn.routing_threshold = 0.0;

-- Enable GNN plan logging for debugging
ALTER SYSTEM SET gnn_plan_logging.enabled = on;
ALTER SYSTEM SET gnn_plan_logging.directory = '/tmp/pg_gnn_plans';

-- Disable forced DuckDB execution so R-GIN can decide
ALTER SYSTEM SET duckdb.force_execution = off;

-- Apply changes
SELECT pg_reload_conf();