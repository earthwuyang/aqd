/*
 * aqd_query_router.h
 *
 * Adaptive Query Dispatcher (AQD) Query Router
 * Implements multiple routing strategies for PostgreSQL/DuckDB query dispatching
 */

#ifndef AQD_QUERY_ROUTER_H
#define AQD_QUERY_ROUTER_H

#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "aqd_feature_logger.h"

/* Routing methods */
typedef enum AQDRoutingMethod
{
    AQD_ROUTE_DEFAULT = 0,          /* pg_duckdb default heuristic */
    AQD_ROUTE_COST_THRESHOLD = 1,   /* Cost-based threshold routing */
    AQD_ROUTE_LIGHTGBM = 2,         /* LightGBM ML-based routing */
    AQD_ROUTE_GNN = 3               /* Graph Neural Network routing */
} AQDRoutingMethod;

/* Query execution engine */
typedef enum AQDExecutionEngine
{
    AQD_ENGINE_POSTGRES = 0,        /* Route to PostgreSQL */
    AQD_ENGINE_DUCKDB = 1,          /* Route to DuckDB */
    AQD_ENGINE_AUTO = 2             /* Let router decide */
} AQDExecutionEngine;

/* Routing decision */
typedef struct AQDRoutingDecision
{
    AQDExecutionEngine engine;      /* Selected engine */
    double confidence;              /* Confidence in decision (0.0-1.0) */
    double predicted_pg_time;       /* Predicted PostgreSQL time */
    double predicted_duck_time;     /* Predicted DuckDB time */
    char reason[256];               /* Reasoning for decision */
    TimestampTz decision_time;      /* When decision was made */
    double decision_latency_us;     /* Time taken to make decision (microseconds) */
} AQDRoutingDecision;

/* Router configuration */
typedef struct AQDRouterConfig
{
    AQDRoutingMethod method;        /* Active routing method */
    double cost_threshold;          /* Cost threshold for cost-based routing */
    char lightgbm_model_path[1024]; /* Path to LightGBM model */
    char gnn_model_path[1024];      /* Path to GNN model */
    bool enable_feedback;           /* Enable online feedback learning */
    bool enable_resource_regulation; /* Enable Mahalanobis resource regulation */
    double resource_balance_factor; /* Resource balancing factor (0.0-1.0) */
} AQDRouterConfig;

/* Thompson Sampling Bandit state */
typedef struct AQDThompsonBandit
{
    /* For PostgreSQL engine */
    double pg_alpha;               /* Success parameter */
    double pg_beta;                /* Failure parameter */
    
    /* For DuckDB engine */
    double duck_alpha;             /* Success parameter */
    double duck_beta;              /* Failure parameter */
    
    /* Recent performance history */
    double pg_recent_performance[100];   /* Recent PostgreSQL performance */
    double duck_recent_performance[100]; /* Recent DuckDB performance */
    int history_index;             /* Current index in history arrays */
    int history_count;             /* Number of entries in history */
    
    /* Exploration parameters */
    double exploration_factor;     /* Exploration vs exploitation balance */
    double decay_factor;           /* Decay factor for historical performance */
} AQDThompsonBandit;

/* Resource regulation state */
typedef struct AQDResourceRegulator
{
    /* CPU usage tracking */
    double pg_cpu_usage;           /* PostgreSQL CPU usage (0.0-1.0) */
    double duck_cpu_usage;         /* DuckDB CPU usage (0.0-1.0) */
    
    /* Memory usage tracking */
    double pg_memory_usage;        /* PostgreSQL memory usage (MB) */
    double duck_memory_usage;      /* DuckDB memory usage (MB) */
    
    /* Mahalanobis distance components */
    double resource_mean[4];       /* Mean resource usage [pg_cpu, duck_cpu, pg_mem, duck_mem] */
    double resource_cov[4][4];     /* Covariance matrix */
    double resource_inv_cov[4][4]; /* Inverse covariance matrix */
    
    /* Load balancing */
    int pg_active_queries;         /* Active PostgreSQL queries */
    int duck_active_queries;       /* Active DuckDB queries */
    double load_balance_threshold; /* Threshold for load balancing */
} AQDResourceRegulator;

/* Main router state */
typedef struct AQDQueryRouter
{
    bool initialized;              /* Whether router is initialized */
    AQDRouterConfig config;        /* Router configuration */
    AQDThompsonBandit bandit;      /* Thompson sampling bandit */
    AQDResourceRegulator regulator; /* Resource regulation system */
    
    /* Statistics */
    int total_queries;             /* Total queries routed */
    int pg_routed_queries;         /* Queries routed to PostgreSQL */
    int duck_routed_queries;       /* Queries routed to DuckDB */
    double total_routing_time_us;  /* Total time spent on routing decisions */
    
    /* Performance tracking */
    double avg_pg_time_ms;         /* Average PostgreSQL execution time */
    double avg_duck_time_ms;       /* Average DuckDB execution time */
    double routing_accuracy;       /* Accuracy of routing decisions */
    
    /* Model handles */
    void *lightgbm_model;          /* LightGBM model handle */
    void *gnn_model;               /* GNN model handle */
} AQDQueryRouter;

/* Global router instance */
extern AQDQueryRouter aqd_query_router;

/* Configuration variables */
extern int aqd_routing_method;
extern double aqd_cost_threshold;
extern char *aqd_lightgbm_model_path;
extern char *aqd_lightgbm_library_path;
extern char *aqd_gnn_model_path;
extern char *aqd_gnn_library_path;
extern bool aqd_enable_thompson_sampling;
extern bool aqd_enable_resource_regulation;
extern double aqd_resource_balance_factor;

/* Tracking (read-only) GUC variables for observability */
extern int aqd_last_decision_engine_code;     /* 0=PostgreSQL, 1=DuckDB */
extern double aqd_last_decision_latency_us;   /* microseconds */

/* Function declarations */

/* Router initialization and cleanup */
extern void aqd_init_query_router(void);
extern void aqd_cleanup_query_router(void);
extern void aqd_reset_router_stats(void);

/* Main routing function */
extern AQDRoutingDecision aqd_route_query(const char *query_text,
                                          PlannedStmt *planned_stmt,
                                          AQDQueryFeatures *features);

/* Routing method implementations */
extern AQDExecutionEngine aqd_route_default(const char *query_text, PlannedStmt *planned_stmt);
extern AQDExecutionEngine aqd_route_cost_threshold(PlannedStmt *planned_stmt, double threshold);
extern AQDExecutionEngine aqd_route_lightgbm(AQDQueryFeatures *features);
extern AQDExecutionEngine aqd_route_gnn(PlannedStmt *planned_stmt, AQDQueryFeatures *features);

/* Thompson sampling bandit functions */
extern void aqd_init_thompson_bandit(AQDThompsonBandit *bandit);
extern AQDExecutionEngine aqd_thompson_sample(AQDThompsonBandit *bandit);
extern void aqd_update_thompson_bandit(AQDThompsonBandit *bandit, 
                                       AQDExecutionEngine engine,
                                       double performance);

/* Resource regulation functions */
extern void aqd_init_resource_regulator(AQDResourceRegulator *regulator);
extern void aqd_update_resource_usage(AQDResourceRegulator *regulator);
extern double aqd_compute_mahalanobis_distance(AQDResourceRegulator *regulator,
                                              double resource_usage[4]);
extern bool aqd_should_balance_load(AQDResourceRegulator *regulator);
extern AQDExecutionEngine aqd_apply_resource_regulation(AQDExecutionEngine preferred_engine,
                                                       AQDResourceRegulator *regulator);

/* Performance feedback functions */
extern void aqd_record_execution_feedback(AQDRoutingDecision *decision,
                                         double actual_time_ms,
                                         AQDExecutionEngine actual_engine);

/* Model loading functions */
extern bool aqd_load_lightgbm_model(const char *model_path);
extern bool aqd_load_gnn_model(const char *model_path);
extern void aqd_unload_models(void);

/* Utility functions */
extern double aqd_predict_execution_time(AQDQueryFeatures *features, 
                                        AQDExecutionEngine engine);
extern double aqd_compute_routing_confidence(AQDQueryFeatures *features,
                                            AQDExecutionEngine selected_engine);
extern void aqd_log_routing_decision(const AQDRoutingDecision *decision);

/* Configuration and GUC setup */
extern void aqd_define_routing_guc_variables(void);
extern void aqd_set_routing_method(AQDRoutingMethod method);
extern AQDRoutingMethod aqd_get_routing_method(void);

/* Statistics and monitoring */
extern void aqd_print_router_stats(void);
extern void aqd_export_router_stats(const char *filename);

#endif /* AQD_QUERY_ROUTER_H */
