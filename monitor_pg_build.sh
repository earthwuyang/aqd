#!/bin/bash
# PostgreSQL Build Monitor

while true; do
    clear
    echo "=== PostgreSQL Build Status Monitor - $(date) ==="
    echo
    
    echo "Screen Sessions:"
    screen -ls | grep -E "(pg_build|lightgbm_compile|postgres_compile)" || echo "No build sessions found"
    echo
    
    echo "Current Build Logs (last 10 lines):"
    echo "--- PostgreSQL Build ---"
    if [ -f "/home/wuy/DB/pg_duckdb_postgres/postgres_build.log" ]; then
        tail -10 /home/wuy/DB/pg_duckdb_postgres/postgres_build.log
    else
        echo "No PostgreSQL build log found yet"
    fi
    echo
    
    echo "--- LightGBM Build ---"
    if [ -f "/home/wuy/DB/pg_duckdb_postgres/lightgbm_build.log" ]; then
        tail -10 /home/wuy/DB/pg_duckdb_postgres/lightgbm_build.log
    else
        echo "No LightGBM build log found yet"
    fi
    echo
    
    echo "Install Directory Contents:"
    if [ -d "/home/wuy/DB/pg_duckdb_postgres/install" ]; then
        ls -la /home/wuy/DB/pg_duckdb_postgres/install/
    else
        echo "Install directory not created yet"
    fi
    
    echo
    echo "Waiting 60 seconds before next check..."
    sleep 60
done