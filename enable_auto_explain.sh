#!/bin/bash

# Enable PostgreSQL's auto_explain module for JSON plan logging
# This is a simpler alternative using PostgreSQL's built-in functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SCRIPT_DIR/pgsql/bin:$PATH"
export PGDATA="$SCRIPT_DIR/data"

echo "Configuring auto_explain for JSON plan logging..."

# Add auto_explain configuration to postgresql.conf
cat >> "$PGDATA/postgresql.conf" << 'EOF'

# Auto Explain Configuration for AQD-style logging
shared_preload_libraries = 'pg_duckdb,auto_explain'
auto_explain.log_min_duration = 0          # Log all queries
auto_explain.log_analyze = false           # Don't run ANALYZE
auto_explain.log_buffers = false           # Don't log buffer usage
auto_explain.log_timing = false            # Don't log timing
auto_explain.log_triggers = false          # Don't log trigger stats
auto_explain.log_verbose = false           # Not verbose
auto_explain.log_format = 'json'           # JSON format
auto_explain.log_nested_statements = true  # Log nested statements
auto_explain.sample_rate = 1.0             # Log all queries (100%)

# Custom GUC for enabling/disabling feature logging
# (This mimics aqd.enable_feature_logging)
log_statement = 'none'                     # Don't log statements separately
log_duration = off                          # Don't log duration separately
EOF

echo "Restarting PostgreSQL to apply changes..."
pg_ctl -D "$PGDATA" restart

echo "Waiting for PostgreSQL to start..."
sleep 3

echo ""
echo "Testing auto_explain..."
psql -h /tmp -p 5432 -d financial << 'EOF'
-- Test query
EXPLAIN (FORMAT JSON) 
SELECT COUNT(*) FROM trans WHERE amount > 1000;

-- Run actual query (will be logged by auto_explain)
SELECT COUNT(*) FROM trans WHERE amount > 1000;
EOF

echo ""
echo "Auto_explain is now configured!"
echo ""
echo "JSON plans are being logged to the PostgreSQL log file."
echo "To view the plans, use:"
echo "  grep 'auto_explain' $PGDATA/log/*.log | grep -o '{.*}' | jq ."
echo ""
echo "To disable auto_explain temporarily:"
echo "  psql -c \"ALTER SYSTEM SET auto_explain.log_min_duration = -1;\""
echo "  psql -c \"SELECT pg_reload_conf();\""
echo ""
echo "To re-enable:"
echo "  psql -c \"ALTER SYSTEM SET auto_explain.log_min_duration = 0;\""
echo "  psql -c \"SELECT pg_reload_conf();\""