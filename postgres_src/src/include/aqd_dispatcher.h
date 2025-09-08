/*
 * aqd_dispatcher.h
 *
 * Adaptive Query Dispatcher (AQD) - Query Routing Engine
 * Routes queries between PostgreSQL and DuckDB based on ML predictions
 * 
 * Based on AQD paper: https://github.com/thuml/AQD
 */

#ifndef AQD_DISPATCHER_H
#define AQD_DISPATCHER_H

#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "executor/instrument.h"
#include "aqd_feature_logger.h"

/* Execution engines */
typedef enum AQDExecutionEngine
{
    AQD_ENGINE_POSTGRES = 0,
    AQD_ENGINE_DUCKDB = 1
} AQDExecutionEngine;

/* Query dispatch methods */
typedef enum AQDDispatchMethod
{
    AQD_METHOD_DEFAULT = 0,        /* pg_duckdb default heuristics */
    AQD_METHOD_COST_THRESHOLD = 1, /* Cost-based threshold */
    AQD_METHOD_LIGHTGBM = 2,       /* LightGBM ML model */
    AQD_METHOD_GNN = 3             /* Graph Neural Network (future) */
} AQDDispatchMethod;

/* Dispatcher configuration */
typedef struct AQDDispatcherConfig
{
    AQDDispatchMethod method;       /* Current dispatch method */
    double cost_threshold;          /* Cost threshold for method 1 */
    char model_path[1024];          /* Path to LightGBM model */
    bool enable_fallback;           /* Enable fallback to PostgreSQL on error */
    bool enable_logging;            /* Enable dispatch decision logging */
    char log_file_path[1024];       /* Path to dispatch log file */
} AQDDispatcherConfig;

/* Dispatch decision result */
typedef struct AQDDispatchDecision
{
    AQDExecutionEngine engine;      /* Selected execution engine */
    AQDDispatchMethod method;       /* Method used for decision */
    double confidence;              /* Confidence score (0.0-1.0) */
    double predicted_time_ratio;    /* Predicted postgres/duckdb time ratio */
    double estimated_cost;          /* PostgreSQL optimizer cost estimate */
    char reason[256];               /* Human-readable reason */
    TimestampTz decision_time;      /* When decision was made */
} AQDDispatchDecision;

/* Global dispatcher state */
typedef struct AQDDispatcher
{
    bool initialized;               /* Whether dispatcher is initialized */
    AQDDispatcherConfig config;     /* Current configuration */
    void *lightgbm_predictor;       /* LightGBM predictor instance */
    FILE *log_file;                 /* Log file handle */
    
    /* Statistics */
    int total_dispatches;           /* Total dispatch decisions */
    int postgres_dispatches;        /* Queries sent to PostgreSQL */
    int duckdb_dispatches;          /* Queries sent to DuckDB */
    int fallback_count;            /* Fallback occurrences */
} AQDDispatcher;

/* Global dispatcher instance */
extern AQDDispatcher aqd_dispatcher;

/* Function declarations */

/* Initialize the dispatcher */
extern void aqd_init_dispatcher(void);

/* Cleanup the dispatcher */
extern void aqd_cleanup_dispatcher(void);

/* Make dispatch decision for a query */
extern AQDDispatchDecision aqd_make_dispatch_decision(const char *query_text,
                                                     PlannedStmt *planned_stmt,
                                                     QueryDesc *query_desc);

/* Execute query on selected engine */
extern void aqd_execute_query(QueryDesc *query_desc, 
                             const AQDDispatchDecision *decision,
                             ScanDirection direction, 
                             uint64 count, 
                             bool execute_once);

/* Configuration functions */
extern void aqd_set_dispatch_method(AQDDispatchMethod method);
extern AQDDispatchMethod aqd_get_dispatch_method(void);
extern void aqd_set_cost_threshold(double threshold);
extern double aqd_get_cost_threshold(void);
extern void aqd_set_model_path(const char *path);
extern const char *aqd_get_model_path(void);

/* Dispatch method implementations */
extern AQDDispatchDecision aqd_dispatch_default(const char *query_text,
                                               PlannedStmt *planned_stmt,
                                               QueryDesc *query_desc);

extern AQDDispatchDecision aqd_dispatch_cost_threshold(const char *query_text,
                                                      PlannedStmt *planned_stmt,
                                                      QueryDesc *query_desc,
                                                      double threshold);

extern AQDDispatchDecision aqd_dispatch_lightgbm(const char *query_text,
                                                 PlannedStmt *planned_stmt,
                                                 QueryDesc *query_desc);

extern AQDDispatchDecision aqd_dispatch_gnn(const char *query_text,
                                           PlannedStmt *planned_stmt,
                                           QueryDesc *query_desc);

/* Utility functions */
extern double aqd_estimate_query_cost(PlannedStmt *planned_stmt);
extern void aqd_log_dispatch_decision(const AQDDispatchDecision *decision,
                                     const char *query_text);
extern const char *aqd_engine_name(AQDExecutionEngine engine);
extern const char *aqd_method_name(AQDDispatchMethod method);

/* Configuration variables */
extern int aqd_dispatch_method;
extern double aqd_cost_threshold;
extern char *aqd_lightgbm_model_path;
extern bool aqd_enable_dispatch_logging;
extern char *aqd_dispatch_log_path;
extern bool aqd_apply_decision;

/* GUC setup function */
extern void aqd_define_dispatch_guc_variables(void);

#endif /* AQD_DISPATCHER_H */
