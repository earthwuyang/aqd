# PostgreSQL Query Plan Logging

This directory includes tools for logging PostgreSQL optimizer JSON plans, similar to the AQD (Adaptive Query Dispatcher) feature logging functionality.

## Features

- **JSON Plan Logging**: Captures query execution plans in JSON format
- **Query Hashing**: Generates unique hashes for normalized queries
- **Execution Time Tracking**: Records actual query execution times
- **JSONL Format**: One JSON object per line for easy streaming and processing
- **Selective Logging**: Can be enabled/disabled per session or globally

## Components

### 1. Python Script: `execute_with_plan_logging.py`

A standalone Python script that executes queries and logs their plans.

**Usage:**

```bash
# Execute a single query
python3 execute_with_plan_logging.py -d financial -q "SELECT COUNT(*) FROM trans"

# Execute queries from a file
python3 execute_with_plan_logging.py -d financial -f benchmark_queries/financial/workload_ap_queries.sql

# Interactive mode
python3 execute_with_plan_logging.py -d financial

# Disable logging
python3 execute_with_plan_logging.py -d financial --disable -q "SELECT 1"

# Custom log path
python3 execute_with_plan_logging.py -d financial --log-path /tmp/my_plans.jsonl -q "SELECT 1"
```

### 2. PostgreSQL Extension: `plan_logger` (Optional)

A custom PostgreSQL extension that hooks into the executor to log plans automatically.

**Installation:**

```bash
cd plan_logger
make && make install
```

**Configuration:**

Add to `postgresql.conf`:
```conf
shared_preload_libraries = 'pg_duckdb,plan_logger'
plan_logger.enable = on
plan_logger.path = '/tmp/postgres_plans.jsonl'
```

**Usage:**

```sql
-- Check settings
SHOW plan_logger.enable;
SELECT * FROM plan_logger_info();

-- Disable for session
SET plan_logger.enable = off;
```

## Log Format

Plans are logged in JSONL format (one JSON object per line):

```json
{
  "timestamp": "2025-09-11T18:41:31.135818",
  "database": "financial",
  "query_hash": "8968b4d3b22ebe2b",
  "query_text": "SELECT COUNT(*) FROM trans WHERE amount > 1000",
  "plan": [{
    "Plan": {
      "Node Type": "Aggregate",
      "Total Cost": 17352.44,
      "Plan Rows": 1,
      ...
    }
  }],
  "execution_time_ms": 68.318
}
```

## Analyzing Logged Plans

### View plans with jq

```bash
# View all plans
cat /tmp/postgres_plans.jsonl | jq .

# Filter by database
cat /tmp/postgres_plans.jsonl | jq 'select(.database == "financial")'

# Extract specific fields
cat /tmp/postgres_plans.jsonl | jq '{query: .query_text, cost: .plan[0].Plan."Total Cost", time: .execution_time_ms}'

# Find slowest queries
cat /tmp/postgres_plans.jsonl | jq -s 'sort_by(.execution_time_ms) | reverse | .[0:10] | .[] | {query: .query_text, time: .execution_time_ms}'
```

### Convert to CSV

```bash
cat /tmp/postgres_plans.jsonl | jq -r '[.timestamp, .database, .query_hash, .execution_time_ms] | @csv' > plans.csv
```

### Python analysis

```python
import json
import pandas as pd

# Read JSONL file
plans = []
with open('/tmp/postgres_plans.jsonl', 'r') as f:
    for line in f:
        plans.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(plans)

# Analyze
print(f"Total queries: {len(df)}")
print(f"Average execution time: {df['execution_time_ms'].mean():.2f} ms")
print(f"Databases: {df['database'].unique()}")
```

## Batch Processing

Process all benchmark queries with plan logging:

```bash
#!/bin/bash
for db in financial employee Airline Credit; do
    echo "Processing $db AP queries..."
    python3 execute_with_plan_logging.py \
        -d "$db" \
        -f "benchmark_queries/$db/workload_ap_queries.sql" \
        --log-path "/tmp/${db}_ap_plans.jsonl"
        
    echo "Processing $db TP queries..."
    python3 execute_with_plan_logging.py \
        -d "$db" \
        -f "benchmark_queries/$db/workload_tp_queries.sql" \
        --log-path "/tmp/${db}_tp_plans.jsonl"
done
```

## Integration with AQD

This plan logging mechanism provides similar functionality to the AQD feature logger:

- **Query Hashing**: Like AQD, we generate unique hashes for normalized queries
- **JSON Plans**: Plans are stored in JSON format for easy parsing
- **Execution Metrics**: Captures execution time alongside plans
- **JSONL Format**: Supports streaming and incremental processing

The logged plans can be used for:
- Training ML models for query optimization
- Analyzing query patterns and performance
- Building adaptive query dispatchers
- Performance regression testing

## Troubleshooting

**Plans not being logged:**
- Check PostgreSQL is running: `pg_ctl -D data status`
- Verify database exists: `psql -l`
- Check file permissions: `ls -la /tmp/postgres_plans.jsonl`

**JSON parsing errors:**
- Ensure JSONL format (one JSON per line)
- Use `jq` to validate: `cat file.jsonl | jq -s .`

**Extension not loading:**
- Check shared_preload_libraries: `SHOW shared_preload_libraries;`
- Restart PostgreSQL after config changes
- Check logs: `tail -f data/log/*.log`

## Related Files

- `execute_with_plan_logging.py` - Main plan logging script
- `plan_logger/` - PostgreSQL extension source
- `enable_plan_logging.sh` - Extension setup script
- `test_plan_logging.py` - Test script for validation
- `/tmp/postgres_plans.jsonl` - Default log location