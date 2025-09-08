#!/bin/bash

while true; do
  clear
  echo "=== Build Status Monitor - $(date) ==="
  echo
  
  echo "PostgreSQL Build Status:"
  if ps aux | grep -q "[m]ake.*postgres"; then
    echo "  ✅ PostgreSQL build ACTIVE"
    echo "  Recent progress:"
    tail -3 compile2.log 2>/dev/null | sed 's/^/    /' || echo "    No log available yet"
  else
    echo "  ⏸️  PostgreSQL build not running"
    if [ -f compile2.log ]; then
      echo "  Last status:"
      tail -2 compile2.log | sed 's/^/    /'
    fi
  fi
  
  echo
  echo "LightGBM Build Status:"
  if ps aux | grep -E "(make|cmake)" | grep -i light | grep -q -v grep; then
    echo "  ✅ LightGBM build ACTIVE"
    echo "  Recent progress:"
    tail -3 /home/wuy/DB/pg_duckdb_postgres/LightGBM/build_log.txt 2>/dev/null | sed 's/^/    /' || echo "    Building LightGBM C++ library..."
  else
    echo "  ⏸️  LightGBM build not running"
    if [ -d "/home/wuy/DB/pg_duckdb_postgres/LightGBM/build" ]; then
      echo "  Last status:"
      echo "    LightGBM configured and ready"
    fi
  fi
  
  echo
  echo "Active compilation processes: $(ps aux | grep -E '(gcc|g\+\+|make|cmake)' | grep -v grep | wc -l)"
  
  echo
  echo "Screen sessions:"
  screen -ls 2>/dev/null | grep -E "(postgres|lightgbm)" | sed 's/^/  /' || echo "  No screen sessions found"
  
  echo "============================================"
  sleep 60
done