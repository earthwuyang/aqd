#!/bin/bash

# Enable plan logging in PostgreSQL
# This script configures the plan_logger extension to log query plans

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SCRIPT_DIR/pgsql/bin:$PATH"
export PGDATA="$SCRIPT_DIR/data"

echo "Enabling plan logging extension..."

# Add extension to preload libraries in postgresql.conf
echo "Configuring PostgreSQL to preload plan_logger..."
if ! grep -q "shared_preload_libraries.*plan_logger" "$PGDATA/postgresql.conf"; then
    # Check if shared_preload_libraries exists
    if grep -q "^shared_preload_libraries" "$PGDATA/postgresql.conf"; then
        # Update existing line
        sed -i "s/^shared_preload_libraries = '\(.*\)'/shared_preload_libraries = '\1,plan_logger'/" "$PGDATA/postgresql.conf"
        # Clean up double commas if any
        sed -i "s/,,/,/g" "$PGDATA/postgresql.conf"
        sed -i "s/= ',/= '/g" "$PGDATA/postgresql.conf"
    else
        # Add new line
        echo "shared_preload_libraries = 'plan_logger'" >> "$PGDATA/postgresql.conf"
    fi
fi

# Add GUC settings for plan_logger
echo "Configuring plan_logger settings..."
cat >> "$PGDATA/postgresql.conf" << EOF

# Plan Logger Settings
plan_logger.enable = on
plan_logger.path = '/tmp/postgres_plans.jsonl'
plan_logger.format = 'jsonl'
EOF

echo "Restarting PostgreSQL to apply changes..."
pg_ctl -D "$PGDATA" restart

echo "Waiting for PostgreSQL to start..."
sleep 3

# Create extension in all databases
echo "Creating plan_logger extension in all databases..."
for db in postgres Airline Credit Carcinogenesis employee financial geneea Hepatitis_std tpch_sf1 tpcds_sf1; do
    if psql -h /tmp -p 5432 -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$db'" | grep -q 1; then
        echo "  Creating extension in $db..."
        psql -h /tmp -p 5432 -d "$db" -c "CREATE EXTENSION IF NOT EXISTS plan_logger;" 2>/dev/null || true
    fi
done

# Test the extension
echo ""
echo "Testing plan_logger..."
psql -h /tmp -p 5432 -d postgres << EOF
-- Check GUC settings
SHOW plan_logger.enable;
SHOW plan_logger.path;
SHOW plan_logger.format;

-- Check extension info
SELECT * FROM plan_logger_info();
EOF

echo ""
echo "Plan logging is now enabled!"
echo "Plans will be logged to: /tmp/postgres_plans.jsonl"
echo ""
echo "To disable plan logging, run:"
echo "  psql -c \"SET plan_logger.enable = off;\""
echo ""
echo "To view logged plans:"
echo "  tail -f /tmp/postgres_plans.jsonl | jq ."