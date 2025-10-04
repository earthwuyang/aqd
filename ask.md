# LightGBM Concurrent Prediction Segfault in PostgreSQL Multi-Process Environment

## Problem Summary

I'm integrating LightGBM into PostgreSQL for query routing. The model loads successfully at startup, but **concurrent predictions cause segfaults** when multiple PostgreSQL backend processes call `LGBM_BoosterPredictForMat()` simultaneously on the same booster handle.

## Architecture

- **PostgreSQL 17.6** with custom routing extension (multi-process, not multi-threaded)
- **LightGBM C API** (lib_lightgbm.so) from latest GitHub
- **Model loading**: Once at PostgreSQL startup in `TopMemoryContext` (shared memory)
- **Model usage**: Multiple backend processes call prediction API concurrently
- **OS**: Linux 6.8.0-59-generic

## Current Implementation

### Model Loading (at startup - works fine)
```c
// Global variable in TopMemoryContext, loaded once by postmaster
LightGBMModel *lightgbm_global_model = NULL;

bool LoadGlobalLightGBMModel(const char *model_path) {
    // Called once at PostgreSQL startup before forking backends
    MemoryContext oldcontext = MemoryContextSwitchTo(TopMemoryContext);

    lightgbm_global_model = (LightGBMModel *) palloc0(sizeof(LightGBMModel));

    ret = LGBM_BoosterCreateFromModelfile(
        model_path,
        &num_iterations,
        &lightgbm_global_model->booster
    );

    // Model loaded successfully: 85 features, 3200 trees (60 iterations)
    // File: /path/to/lightgbm_model.txt (350KB)
}
```

### Prediction (concurrent - causes segfaults)
```c
// Called by multiple backend PROCESSES (not threads) concurrently
// All backends inherit the same lightgbm_global_model pointer after fork()
static double PredictWithLightGBM(LightGBMModel *model, double *features) {
    if (!model || !model->loaded || !model->booster)
        return 0.5;

    // THE PROBLEMATIC CALL - crashes when multiple processes call simultaneously
    ret = LGBM_BoosterPredictForMat(
        model->booster,        // SAME booster handle shared by all backends
        features,              // Per-backend feature array (stack-allocated)
        1,                     // data_type: double
        1,                     // nrow: 1 (single prediction)
        85,                    // ncol: 85 features
        1,                     // is_row_major
        0,                     // predict_type: normal (C_API_PREDICT_NORMAL)
        0,                     // start_iteration
        -1,                    // num_iteration: use all
        "num_threads=1",       // Force single-threaded
        &out_len,
        &prediction
    );

    return prediction;
}
```

## Test Results

**Scenario**: 9 concurrent queries starting simultaneously
- **Single query**: ✅ Works perfectly
- **9 concurrent queries**: ❌ 5/9 crash with segfault
- **Error**: "LightGBM routing failed with segfault, falling back to PostgreSQL"
- **Success pattern**: First ~4 queries succeed, remaining 5 crash

## PostgreSQL Multi-Process Architecture (Critical Context)

PostgreSQL uses **processes, not threads**:

1. **Postmaster** (main process) loads model at startup
2. **fork()** creates child backend processes for each connection
3. Each backend inherits **copy-on-write memory** including the booster pointer
4. **No shared memory locking** - processes have separate address spaces
5. Traditional **spinlocks/mutexes don't work** across processes in PostgreSQL

After fork:
```
Postmaster (PID 1000)
  └─ lightgbm_global_model->booster = 0x7f1234567890
      ├─ Backend 1 (PID 1001) sees same pointer 0x7f1234567890
      ├─ Backend 2 (PID 1002) sees same pointer 0x7f1234567890
      └─ Backend 3 (PID 1003) sees same pointer 0x7f1234567890
```

## What I've Tried

1. ✅ **Model loaded once at startup** - eliminates concurrent file access
2. ✅ **`num_threads=1` parameter** - forces single-threaded prediction
3. ❌ **PostgreSQL spinlock** - doesn't work across processes (only threads)
4. ❌ **Random delays before prediction** - doesn't help
5. ❌ **Per-backend model loading** - causes concurrent file access crashes
6. ❌ **PG_TRY/PG_CATCH** - can't catch SIGSEGV signals

## Root Cause Hypothesis

I suspect the segfault is caused by one of:

1. **LightGBM internal state mutation**: The booster handle contains mutable buffers/caches that get corrupted when multiple processes modify them simultaneously (even for read-only prediction)

2. **Process-local state after fork()**: LightGBM might use thread-local storage, mutexes, or other process-local resources that become invalid after PostgreSQL fork()s the backends

3. **Memory layout assumptions**: LightGBM might assume single-process access and use non-atomic operations on shared internal structures

## Key Questions

1. **Is `LGBM_BoosterPredictForMat()` safe for multi-process concurrent read-only prediction?**
   - Thread-safe ≠ Process-safe
   - Does it maintain internal mutable state?

2. **Does LightGBM support shared memory scenarios?**
   - Can a booster loaded in parent process be used by fork()ed children?
   - Are there any known limitations or requirements?

3. **What's the recommended architecture for this use case?**
   - Option A: Each process loads its own model (wastes ~35MB for 100 backends)
   - Option B: Use shared memory with IPC locking (how?)
   - Option C: Single prediction server process with IPC (too slow?)

4. **Are there LightGBM build flags or API calls** to make prediction process-safe?
   - Copy-on-write friendly modes?
   - Prediction with no internal state mutation?

## Debugging Information

No actual stack traces available (segfault happens in C library, caught by Python test harness), but the pattern is:
- Crashes happen during `LGBM_BoosterPredictForMat()` call
- Always affects queries 5-9 (the later ones in concurrent batch)
- Never crashes with sequential queries
- Model file is valid (works fine in single-process mode)

## Workarounds Considered

1. **Copy model for each backend** - Works but wastes memory
2. **Serialize predictions with file locks** - Would kill performance
3. **Pre-compute predictions** - Not feasible, queries are dynamic
4. **Use Python LightGBM via separate process** - Too slow, defeats purpose

## What Would Help

- Confirmation if LightGBM C API is designed for multi-process scenarios
- Recommended synchronization approach (if any)
- Whether fork() invalidates booster handles
- Any experience with LightGBM in Apache/nginx (multi-process servers)
- Alternative APIs or methods that are process-safe

Thank you for any insights!
