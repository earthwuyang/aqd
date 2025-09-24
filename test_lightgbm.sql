-- Test LightGBM routing with a simple query
SET client_min_messages = DEBUG1;
SET lightgbm.enable_plan_logging = true;

-- Simple aggregation query that should route to DuckDB
EXPLAIN (ANALYZE, VERBOSE, BUFFERS) 
SELECT COUNT(*), AVG(l_quantity), SUM(l_extendedprice)
FROM lineitem
WHERE l_shipdate >= '1998-01-01'
GROUP BY l_returnflag;

-- Check what happened
SHOW lightgbm.last_decision;
SHOW lightgbm.last_decision_us;
