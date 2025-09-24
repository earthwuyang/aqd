# Known Issues with LightGBM Query Routing

## ~~Issue: Concurrent Query Execution Hangs~~ FIXED

### Solution Implemented
The issue was caused by OpenMP parallelization in LightGBM conflicting with PostgreSQL's multi-process architecture. Fixed by:
1. Setting `OMP_NUM_THREADS=1` environment variable before loading LightGBM library
2. Passing `num_threads=1` parameter to `LGBM_BoosterPredictForMat` function
3. Checking for parallel workers with `IsParallelWorker()` to skip routing

### Original Issue: Concurrent Query Execution Hangs (Even with Small Model)

### Description
When LightGBM routing is enabled (even with a small 278KB model), concurrent query execution can hang indefinitely. This particularly affects:
- Parallel query workers
- Concurrent connections executing queries simultaneously
- Performance testing with multiple concurrent queries

### Symptoms
- PostgreSQL backends stuck in SELECT state
- Parallel workers spawned but not completing
- Performance tests hanging at ~40/45 queries
- Process list shows multiple stuck postgres processes with parallel workers

### Root Cause
The issue occurs even with a small model (278KB, 50 features, 10 trees) and appears to be related to:
1. **GUC Assignment Hooks**: Setting `lightgbm.model_path` per-connection triggers model loading
2. **LightGBM C Library**: The LGBM_BoosterPredictForMat function may not be thread-safe
3. **GUC Access**: Even reading GUCs like `lightgbm.last_routed_engine` can cause issues
4. **Parallel Workers**: Despite IsParallelWorker() check, issues persist

### Temporary Solution
Disable LightGBM routing for concurrent workloads:

```sql
-- Option 1: Disable at session level
SET lightgbm.enabled = false;

-- Option 2: Disable in postgresql.conf
# lightgbm.enabled = false

-- Option 3: Disable parallel queries
SET max_parallel_workers_per_gather = 0;
```

### Permanent Solutions (TODO)

1. **Implement Proper Synchronization**
   - Add mutex/spinlock protection around model loading
   - Ensure only one backend loads the model at a time
   - Use shared memory for model data

2. **Use Smaller Model**
   - Reduce number of trees (currently 10,000)
   - Reduce feature count if possible
   - Use model compression techniques

3. **Lazy Loading Strategy**
   - Load model only when first query needs routing
   - Cache model in shared memory after first load
   - Implement reference counting for safe cleanup

4. **Disable for Parallel Workers**
   - Detect if current process is a parallel worker
   - Skip LightGBM routing for parallel workers
   - Only use routing in main backend process

### Workaround for Testing
For performance testing or concurrent workloads:

```bash
# 1. Temporarily disable LightGBM in configuration
sed -i 's/lightgbm.enabled = true/# lightgbm.enabled = false/' data/postgresql.conf

# 2. Restart PostgreSQL
pg_ctl -D data restart

# 3. Run your tests
python performance_test.py

# 4. Re-enable after testing if needed
sed -i 's/# lightgbm.enabled = false/lightgbm.enabled = true/' data/postgresql.conf
pg_ctl -D data restart
```

### Impact
- Single-connection workloads: Works fine
- Multi-connection workloads: May experience hangs
- Performance testing: Requires LightGBM disabled
- Production use: Not recommended until fixed

### Monitoring
Check for stuck queries:
```sql
SELECT pid, state, query
FROM pg_stat_activity
WHERE state != 'idle'
AND query NOT LIKE '%pg_stat_activity%';
```

Check for parallel workers:
```bash
ps aux | grep "parallel worker" | wc -l
```

### Future Improvements
1. Implement shared memory model storage
2. Add connection pooling awareness
3. Optimize model size and loading time
4. Add timeout mechanism for model operations
5. Implement graceful fallback to PostgreSQL on model issues