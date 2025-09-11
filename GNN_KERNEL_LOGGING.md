# PostgreSQL Kernel-Level GNN Plan Logging

This document describes the kernel-level modifications to PostgreSQL 17 that enable automatic query plan logging for GNN (Graph Neural Network) training and routing decisions.

## Overview

The GNN plan logging system is integrated directly into the PostgreSQL kernel, providing:
- **Automatic plan logging** in JSON format to disk
- **Query feature extraction** for GNN model training
- **Routing hooks** for dispatching queries between PostgreSQL and DuckDB
- **GUC variables** for runtime configuration

## Architecture

### Core Components

1. **`gnn_plan_logger.h`** - Header file defining structures and interfaces
   - Location: `postgres/src/include/utils/gnn_plan_logger.h`
   - Defines `GNNQueryFeatures` structure for feature extraction
   - Declares routing methods and execution engines

2. **`gnn_plan_logger.c`** - Implementation of plan logging and routing
   - Location: `postgres/src/backend/utils/misc/gnn_plan_logger.c`
   - Implements plan serialization to JSON
   - Extracts query features (joins, scans, aggregates, etc.)
   - Provides routing decision logic

3. **Modified Files**:
   - `execMain.c` - Hooks for plan logging in executor
   - `guc.c` - GUC variable registration
   - `Makefile` - Build integration

## GUC Variables

Configure the GNN plan logger using these PostgreSQL configuration parameters:

```sql
-- Enable/disable plan logging
SET gnn.plan_logging_enabled = on;

-- Set log directory
SET gnn.plan_log_directory = '/tmp/pg_gnn_plans';

-- Set GNN model path (for routing)
SET gnn.model_path = '/path/to/model.onnx';

-- Routing method: none, heuristic, gnn, hybrid
SET gnn.routing_method = 'heuristic';

-- Routing confidence threshold
SET gnn.routing_threshold = 0.5;
```

## Plan Logging Format

Plans are logged as JSONL (one JSON object per line):

```json
{
  "timestamp": "2025-09-11 19:00:00.123456+08",
  "query_hash": "8968b4d3",
  "query_text": "SELECT COUNT(*) FROM trans WHERE amount > 1000",
  "database": "financial",
  "features": {
    "num_nodes": 5,
    "max_depth": 3,
    "total_cost": 17352.44,
    "total_rows": 619665,
    "num_seq_scans": 1,
    "num_index_scans": 0,
    "num_joins": 0,
    "num_aggregates": 1,
    "num_sorts": 0
  },
  "plan": [
    {
      "Plan": {
        "Node Type": "Aggregate",
        "Strategy": "Plain",
        "Total Cost": 17352.44,
        ...
      }
    }
  ]
}
```

## Feature Extraction

The system automatically extracts these features from query plans:

### Structural Features
- `num_nodes` - Total number of plan nodes
- `max_depth` - Maximum depth of plan tree
- `total_cost` - Estimated total cost
- `total_rows` - Estimated row count

### Operation Counts
- `num_seq_scans` - Sequential scan operations
- `num_index_scans` - Index scan operations  
- `num_joins` - Join operations (NestLoop, Hash, Merge)
- `num_aggregates` - Aggregation operations
- `num_sorts` - Sort operations

## Routing Decision Logic

The system supports multiple routing methods:

### 1. Heuristic Routing
Simple rules based on query characteristics:
- Large aggregations → DuckDB
- Many joins with large results → DuckDB
- Heavy sorting → DuckDB
- Point queries → PostgreSQL

### 2. GNN Model Routing
Uses trained GNN model to predict optimal engine:
- Converts plan features to tensor format
- Runs inference through GNN
- Returns engine with highest confidence score

### 3. Hybrid Routing
Combines GNN predictions with heuristics:
- Uses GNN prediction when confidence > threshold
- Falls back to heuristics for low-confidence predictions

## Building PostgreSQL with GNN Support

1. **Apply the modifications**:
   ```bash
   cd postgres
   # Files are already modified in place
   ```

2. **Configure PostgreSQL**:
   ```bash
   ./configure --prefix=$PWD/../pgsql --enable-debug --enable-cassert
   ```

3. **Compile**:
   ```bash
   make -j$(nproc)
   make install
   ```

4. **Initialize database**:
   ```bash
   initdb -D ../data
   ```

5. **Configure GNN settings in postgresql.conf**:
   ```conf
   gnn.plan_logging_enabled = on
   gnn.plan_log_directory = '/tmp/pg_gnn_plans'
   gnn.routing_method = 'heuristic'
   ```

6. **Start PostgreSQL**:
   ```bash
   pg_ctl -D ../data start
   ```

## Using the Plan Logger

### Enable logging for a session:
```sql
SET gnn.plan_logging_enabled = on;
```

### Check logging status:
```sql
SHOW gnn.plan_logging_enabled;
SHOW gnn.plan_log_directory;
```

### View logged plans:
```bash
# List log files
ls /tmp/pg_gnn_plans/

# View latest plans
tail -f /tmp/pg_gnn_plans/plans_*.jsonl | jq .
```

### Analyze logged plans:
```python
import json
import pandas as pd

# Read JSONL file
plans = []
with open('/tmp/pg_gnn_plans/plans_20250911_190000.jsonl', 'r') as f:
    for line in f:
        plans.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame([p['features'] for p in plans])
print(df.describe())
```

## Training GNN Models

The logged plans can be used to train GNN models for routing:

1. **Collect training data**: Run workloads with logging enabled
2. **Label data**: Mark which engine performed better for each query
3. **Extract features**: Use the logged features as input
4. **Train GNN**: Use frameworks like PyTorch Geometric or DGL
5. **Export model**: Save as ONNX or TorchScript
6. **Load in PostgreSQL**: Set `gnn.model_path`

## Integration with pg_duckdb

When pg_duckdb is installed, the routing decision can direct queries:
- `ENGINE_POSTGRES` - Execute in PostgreSQL
- `ENGINE_DUCKDB` - Execute in DuckDB via pg_duckdb

The routing hook in `execMain.c` makes this decision before query execution begins.

## Performance Considerations

- **Logging overhead**: Minimal (<1ms per query)
- **Feature extraction**: Fast, uses existing plan tree
- **JSON serialization**: Efficient using PostgreSQL's EXPLAIN infrastructure
- **File I/O**: Asynchronous writes with buffering
- **GNN inference**: ~5-10ms with optimized models

## Troubleshooting

### Plans not being logged:
```sql
-- Check if enabled
SHOW gnn.plan_logging_enabled;

-- Check log directory exists
\! ls -la /tmp/pg_gnn_plans/

-- Check PostgreSQL logs
\! tail -f data/log/*.log | grep GNN
```

### Invalid JSON in log files:
```bash
# Validate JSONL format
cat plans_*.jsonl | jq -s . > /dev/null
```

### GNN model not loading:
```sql
-- Check model path
SHOW gnn.model_path;

-- Check PostgreSQL logs for errors
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

## Future Enhancements

1. **Real-time routing**: Adaptive routing based on system load
2. **Distributed logging**: Send plans to central training server
3. **Online learning**: Update models based on execution feedback
4. **Cost model calibration**: Use actual vs estimated costs
5. **Query rewriting**: Optimize queries before routing

## Related Files

- `postgres/src/include/utils/gnn_plan_logger.h` - Header definitions
- `postgres/src/backend/utils/misc/gnn_plan_logger.c` - Implementation
- `postgres/src/backend/executor/execMain.c` - Executor hooks
- `postgres/src/backend/utils/misc/guc.c` - GUC registration
- `/tmp/pg_gnn_plans/` - Default log directory