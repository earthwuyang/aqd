#include "postgres.h"
#include "aqd_dispatcher.h"
#include "aqd_feature_logger.h"
#include "lightgbm_inference.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"
#include "miscadmin.h"
#include <math.h>

/* Global dispatcher instance */
AQDDispatcher aqd_dispatcher = {0};

/* Configuration variables */
int aqd_dispatch_method = AQD_METHOD_DEFAULT;
double aqd_cost_threshold = 1000.0;
char *aqd_lightgbm_model_path = NULL;
bool aqd_enable_dispatch_logging = false;
char *aqd_dispatch_log_path = NULL;
bool aqd_apply_decision = false;

/* Forward declarations for GUC assign hooks */
static void aqd_assign_dispatch_method(int newval, void *extra);
static void aqd_assign_lightgbm_model_path(const char *newval, void *extra);
static void aqd_assign_enable_dispatch_logging(bool newval, void *extra);
static void aqd_assign_dispatch_log_path(const char *newval, void *extra);

/*
 * Initialize the AQD dispatcher
 */
void
aqd_init_dispatcher(void)
{
    if (aqd_dispatcher.initialized)
        return;
        
    memset(&aqd_dispatcher, 0, sizeof(AQDDispatcher));
    
    /* Set default configuration */
    aqd_dispatcher.config.method = AQD_METHOD_DEFAULT;
    aqd_dispatcher.config.cost_threshold = 1000.0;
    aqd_dispatcher.config.enable_fallback = true;
    aqd_dispatcher.config.enable_logging = false;
    
    /* Initialize LightGBM predictor if model path is provided */
    if (aqd_lightgbm_model_path && strlen(aqd_lightgbm_model_path) > 0)
    {
        LightGBMPredictor *predictor = lightgbm_create_predictor();
        if (lightgbm_load_model(predictor, aqd_lightgbm_model_path))
        {
            aqd_dispatcher.lightgbm_predictor = predictor;
            ereport(LOG,
                    (errmsg("AQD: Successfully loaded LightGBM model from %s", aqd_lightgbm_model_path)));
        }
        else
        {
            lightgbm_free_predictor(predictor);
            ereport(WARNING,
                    (errmsg("AQD: Failed to load LightGBM model from %s", aqd_lightgbm_model_path)));
        }
    }
    
    /* Initialize log file if enabled */
    if (aqd_enable_dispatch_logging && aqd_dispatch_log_path)
    {
        aqd_dispatcher.log_file = fopen(aqd_dispatch_log_path, "a");
        if (aqd_dispatcher.log_file == NULL)
        {
            ereport(WARNING,
                    (errmsg("AQD: Failed to open dispatch log file: %s", aqd_dispatch_log_path)));
        }
        aqd_dispatcher.config.enable_logging = (aqd_dispatcher.log_file != NULL);
    }
    
    aqd_dispatcher.initialized = true;
    
    ereport(LOG,
            (errmsg("AQD: Dispatcher initialized with method=%s", 
                   aqd_method_name(aqd_dispatch_method))));
}

/*
 * Cleanup the AQD dispatcher
 */
void
aqd_cleanup_dispatcher(void)
{
    if (!aqd_dispatcher.initialized)
        return;
        
    /* Free LightGBM predictor */
    if (aqd_dispatcher.lightgbm_predictor)
    {
        lightgbm_free_predictor((LightGBMPredictor *) aqd_dispatcher.lightgbm_predictor);
        aqd_dispatcher.lightgbm_predictor = NULL;
    }
    
    /* Close log file */
    if (aqd_dispatcher.log_file)
    {
        fclose(aqd_dispatcher.log_file);
        aqd_dispatcher.log_file = NULL;
    }
    
    aqd_dispatcher.initialized = false;
}

/* --- GUC assign hooks -------------------------------------------------- */

static void
aqd_assign_dispatch_method(int newval, void *extra)
{
    aqd_dispatch_method = newval;
    if (aqd_dispatcher.initialized)
        aqd_dispatcher.config.method = (AQDDispatchMethod) newval;
}

static void
aqd_assign_lightgbm_model_path(const char *newval, void *extra)
{
    /* Reload LightGBM model when path changes */
    if (!newval)
        return;

    if (aqd_dispatcher.lightgbm_predictor)
    {
        lightgbm_free_predictor((LightGBMPredictor *) aqd_dispatcher.lightgbm_predictor);
        aqd_dispatcher.lightgbm_predictor = NULL;
    }

    if (strlen(newval) > 0)
    {
        LightGBMPredictor *predictor = lightgbm_create_predictor();
        if (lightgbm_load_model(predictor, newval))
        {
            aqd_dispatcher.lightgbm_predictor = predictor;
            ereport(LOG,
                    (errmsg("AQD: Successfully loaded LightGBM model from %s", newval)));
        }
        else
        {
            lightgbm_free_predictor(predictor);
            ereport(WARNING,
                    (errmsg("AQD: Failed to load LightGBM model from %s", newval)));
        }
    }
}

static void
aqd_assign_enable_dispatch_logging(bool newval, void *extra)
{
    aqd_enable_dispatch_logging = newval;
    if (!aqd_dispatcher.initialized)
        return;

    if (aqd_dispatcher.log_file)
    {
        fclose(aqd_dispatcher.log_file);
        aqd_dispatcher.log_file = NULL;
    }

    if (newval && aqd_dispatch_log_path && strlen(aqd_dispatch_log_path) > 0)
    {
        aqd_dispatcher.log_file = fopen(aqd_dispatch_log_path, "a");
        if (!aqd_dispatcher.log_file)
            ereport(WARNING, (errmsg("AQD: Failed to open dispatch log file: %s", aqd_dispatch_log_path)));
    }
    aqd_dispatcher.config.enable_logging = (aqd_dispatcher.log_file != NULL);
}

static void
aqd_assign_dispatch_log_path(const char *newval, void *extra)
{
    if (!aqd_dispatcher.initialized)
        return;

    if (aqd_dispatcher.log_file)
    {
        fclose(aqd_dispatcher.log_file);
        aqd_dispatcher.log_file = NULL;
    }
    if (newval && strlen(newval) > 0 && aqd_enable_dispatch_logging)
    {
        aqd_dispatcher.log_file = fopen(newval, "a");
        if (!aqd_dispatcher.log_file)
            ereport(WARNING, (errmsg("AQD: Failed to open dispatch log file: %s", newval)));
    }
    aqd_dispatcher.config.enable_logging = (aqd_dispatcher.log_file != NULL);
}

/*
 * Make dispatch decision for a query
 */
AQDDispatchDecision
aqd_make_dispatch_decision(const char *query_text,
                          PlannedStmt *planned_stmt,
                          QueryDesc *query_desc)
{
    AQDDispatchDecision decision;
    
    /* Initialize decision structure */
    memset(&decision, 0, sizeof(AQDDispatchDecision));
    decision.engine = AQD_ENGINE_POSTGRES;  /* Default fallback */
    decision.method = (AQDDispatchMethod) aqd_dispatch_method;
    decision.confidence = 0.5;
    decision.predicted_time_ratio = 1.0;
    decision.decision_time = GetCurrentTimestamp();
    strcpy(decision.reason, "Default routing");
    
    /* Ensure dispatcher is initialized */
    if (!aqd_dispatcher.initialized)
        aqd_init_dispatcher();
    
    /* Route based on configured method */
    switch (aqd_dispatch_method)
    {
        case AQD_METHOD_DEFAULT:
            decision = aqd_dispatch_default(query_text, planned_stmt, query_desc);
            break;
            
        case AQD_METHOD_COST_THRESHOLD:
            decision = aqd_dispatch_cost_threshold(query_text, planned_stmt, query_desc, aqd_cost_threshold);
            break;
            
        case AQD_METHOD_LIGHTGBM:
            decision = aqd_dispatch_lightgbm(query_text, planned_stmt, query_desc);
            break;
            
        case AQD_METHOD_GNN:
            decision = aqd_dispatch_gnn(query_text, planned_stmt, query_desc);
            break;
            
        default:
            decision = aqd_dispatch_default(query_text, planned_stmt, query_desc);
            break;
    }
    
    /* Update statistics */
    aqd_dispatcher.total_dispatches++;
    if (decision.engine == AQD_ENGINE_POSTGRES)
        aqd_dispatcher.postgres_dispatches++;
    else
        aqd_dispatcher.duckdb_dispatches++;
    
    /* Log decision if enabled */
    if (aqd_dispatcher.config.enable_logging)
    {
        aqd_log_dispatch_decision(&decision, query_text);
    }
    
    return decision;
}

/*
 * Default dispatch method (pg_duckdb heuristics)
 */
AQDDispatchDecision
aqd_dispatch_default(const char *query_text,
                    PlannedStmt *planned_stmt,
                    QueryDesc *query_desc)
{
    AQDDispatchDecision decision;
    
    memset(&decision, 0, sizeof(AQDDispatchDecision));
    decision.engine = AQD_ENGINE_POSTGRES;
    decision.method = AQD_METHOD_DEFAULT;
    decision.confidence = 0.5;
    decision.predicted_time_ratio = 1.0;
    decision.decision_time = GetCurrentTimestamp();
    
    /* Simple heuristic: use DuckDB for analytical queries */
    bool has_aggregation = (planned_stmt && 
                           aqd_count_aggregates(planned_stmt->planTree) > 0);
    bool has_join = (planned_stmt && 
                    aqd_count_joins(planned_stmt->planTree) > 0);
    
    if (has_aggregation || has_join)
    {
        decision.engine = AQD_ENGINE_DUCKDB;
        strcpy(decision.reason, "Analytical query with aggregation/joins");
    }
    else
    {
        strcpy(decision.reason, "Simple OLTP query");
    }
    
    return decision;
}

/*
 * Cost threshold dispatch method
 */
AQDDispatchDecision
aqd_dispatch_cost_threshold(const char *query_text,
                           PlannedStmt *planned_stmt,
                           QueryDesc *query_desc,
                           double threshold)
{
    AQDDispatchDecision decision;
    
    memset(&decision, 0, sizeof(AQDDispatchDecision));
    decision.engine = AQD_ENGINE_POSTGRES;
    decision.method = AQD_METHOD_COST_THRESHOLD;
    decision.confidence = 0.7;
    decision.predicted_time_ratio = 1.0;
    decision.decision_time = GetCurrentTimestamp();
    
    if (planned_stmt)
    {
        decision.estimated_cost = aqd_estimate_query_cost(planned_stmt);
        
        if (decision.estimated_cost > threshold)
        {
            decision.engine = AQD_ENGINE_DUCKDB;
            snprintf(decision.reason, sizeof(decision.reason),
                    "High cost query (%.2f > %.2f)", 
                    decision.estimated_cost, threshold);
        }
        else
        {
            snprintf(decision.reason, sizeof(decision.reason),
                    "Low cost query (%.2f <= %.2f)", 
                    decision.estimated_cost, threshold);
        }
    }
    else
    {
        strcpy(decision.reason, "No plan available, using PostgreSQL");
    }
    
    return decision;
}

/*
 * LightGBM ML dispatch method
 */
AQDDispatchDecision
aqd_dispatch_lightgbm(const char *query_text,
                     PlannedStmt *planned_stmt,
                     QueryDesc *query_desc)
{
    AQDDispatchDecision decision;
    LightGBMPredictor *predictor;
    
    memset(&decision, 0, sizeof(AQDDispatchDecision));
    decision.engine = AQD_ENGINE_POSTGRES;
    decision.method = AQD_METHOD_LIGHTGBM;
    decision.confidence = 0.9;
    decision.predicted_time_ratio = 1.0;
    decision.decision_time = GetCurrentTimestamp();
    
    predictor = (LightGBMPredictor *) aqd_dispatcher.lightgbm_predictor;
    
    if (predictor && lightgbm_is_loaded(predictor))
    {
        /* Extract features for prediction */
        AQDQueryFeatures features;
        memset(&features, 0, sizeof(AQDQueryFeatures));
        
        aqd_extract_query_features(&features, query_text, planned_stmt, query_desc);
        
        /* Create feature vector for LightGBM */
        double feature_vector[14] = {0.0};  /* 14 features from our training */
        
        /* Map AQD features to LightGBM feature vector */
        for (int i = 0; i < features.num_features; i++)
        {
            const char *name = features.features[i].name;
            double value = features.features[i].value;
            
            int feature_idx = lightgbm_get_feature_index(predictor, name);
            if (feature_idx >= 0 && feature_idx < 14)
            {
                feature_vector[feature_idx] = value;
            }
        }
        
        /* Make prediction */
        double log_time_diff = lightgbm_predict(predictor, feature_vector);
        decision.predicted_time_ratio = exp(log_time_diff);
        
        /* Route based on prediction */
        if (decision.predicted_time_ratio > 1.0)
        {
            /* PostgreSQL predicted to be slower, use DuckDB */
            decision.engine = AQD_ENGINE_DUCKDB;
            snprintf(decision.reason, sizeof(decision.reason),
                    "LightGBM: PostgreSQL %.2fx slower", decision.predicted_time_ratio);
        }
        else
        {
            /* PostgreSQL predicted to be faster */
            decision.engine = AQD_ENGINE_POSTGRES;
            snprintf(decision.reason, sizeof(decision.reason),
                    "LightGBM: PostgreSQL %.2fx faster", 1.0 / decision.predicted_time_ratio);
        }
    }
    else
    {
        /* Fallback if no model available */
        strcpy(decision.reason, "LightGBM model not loaded, using PostgreSQL");
        decision.confidence = 0.1;
        aqd_dispatcher.fallback_count++;
    }
    
    return decision;
}

/*
 * GNN dispatch method (placeholder for future implementation)
 */
AQDDispatchDecision
aqd_dispatch_gnn(const char *query_text,
                PlannedStmt *planned_stmt,
                QueryDesc *query_desc)
{
    AQDDispatchDecision decision;
    
    memset(&decision, 0, sizeof(AQDDispatchDecision));
    decision.engine = AQD_ENGINE_POSTGRES;
    decision.method = AQD_METHOD_GNN;
    decision.confidence = 0.1;
    decision.predicted_time_ratio = 1.0;
    decision.decision_time = GetCurrentTimestamp();
    strcpy(decision.reason, "GNN not implemented, using PostgreSQL");
    
    aqd_dispatcher.fallback_count++;
    
    return decision;
}

/*
 * Estimate query cost from plan
 */
double
aqd_estimate_query_cost(PlannedStmt *planned_stmt)
{
    if (planned_stmt == NULL || planned_stmt->planTree == NULL)
        return 0.0;
        
    return planned_stmt->planTree->total_cost;
}

/*
 * Log dispatch decision
 */
void
aqd_log_dispatch_decision(const AQDDispatchDecision *decision,
                         const char *query_text)
{
    char timestamp_str[64];
    
    if (aqd_dispatcher.log_file == NULL || decision == NULL)
        return;
    
    /* Format timestamp */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d %H:%M:%S", 
             localtime(&tv.tv_sec));
    
    /* Log in JSON format */
    fprintf(aqd_dispatcher.log_file,
            "{\"timestamp\":\"%s\",\"engine\":\"%s\",\"method\":\"%s\","
            "\"confidence\":%.3f,\"predicted_ratio\":%.3f,\"estimated_cost\":%.2f,"
            "\"reason\":\"%s\"}\n",
            timestamp_str,
            aqd_engine_name(decision->engine),
            aqd_method_name(decision->method),
            decision->confidence,
            decision->predicted_time_ratio,
            decision->estimated_cost,
            decision->reason);
    
    fflush(aqd_dispatcher.log_file);
}

/*
 * Utility functions
 */
const char *
aqd_engine_name(AQDExecutionEngine engine)
{
    switch (engine)
    {
        case AQD_ENGINE_POSTGRES:
            return "PostgreSQL";
        case AQD_ENGINE_DUCKDB:
            return "DuckDB";
        default:
            return "Unknown";
    }
}

const char *
aqd_method_name(AQDDispatchMethod method)
{
    switch (method)
    {
        case AQD_METHOD_DEFAULT:
            return "Default";
        case AQD_METHOD_COST_THRESHOLD:
            return "CostThreshold";
        case AQD_METHOD_LIGHTGBM:
            return "LightGBM";
        case AQD_METHOD_GNN:
            return "GNN";
        default:
            return "Unknown";
    }
}

/*
 * Configuration setters/getters
 */
void
aqd_set_dispatch_method(AQDDispatchMethod method)
{
    aqd_dispatch_method = method;
    if (aqd_dispatcher.initialized)
        aqd_dispatcher.config.method = method;
}

AQDDispatchMethod
aqd_get_dispatch_method(void)
{
    return (AQDDispatchMethod) aqd_dispatch_method;
}

void
aqd_set_cost_threshold(double threshold)
{
    aqd_cost_threshold = threshold;
    if (aqd_dispatcher.initialized)
        aqd_dispatcher.config.cost_threshold = threshold;
}

double
aqd_get_cost_threshold(void)
{
    return aqd_cost_threshold;
}

void
aqd_set_model_path(const char *path)
{
    if (aqd_lightgbm_model_path)
        pfree(aqd_lightgbm_model_path);
        
    aqd_lightgbm_model_path = pstrdup(path);
}

const char *
aqd_get_model_path(void)
{
    return aqd_lightgbm_model_path ? aqd_lightgbm_model_path : "";
}

/*
 * Define GUC variables for AQD dispatcher
 */
void
aqd_define_dispatch_guc_variables(void)
{
    DefineCustomIntVariable("aqd.dispatch_method",
                           "AQD query dispatch method (0=default, 1=cost_threshold, 2=lightgbm, 3=gnn)",
                           "Controls which method AQD uses to route queries between PostgreSQL and DuckDB",
                           &aqd_dispatch_method,
                           AQD_METHOD_DEFAULT,
                           AQD_METHOD_DEFAULT,
                           AQD_METHOD_GNN,
                           PGC_SUSET,
                           0,
                           NULL,
                           aqd_assign_dispatch_method,
                           NULL);
    
    DefineCustomRealVariable("aqd.cost_threshold",
                            "Cost threshold for cost-based routing",
                            "Queries with cost above this threshold will be routed to DuckDB",
                            &aqd_cost_threshold,
                            1000.0,
                            0.0,
                            1e10,
                            PGC_SUSET,
                            0,
                            NULL,
                            NULL,
                            NULL);
    
    DefineCustomStringVariable("aqd.lightgbm_model_path",
                              "Path to LightGBM model file",
                              "File path to the trained LightGBM model for query routing",
                              &aqd_lightgbm_model_path,
                              "",
                              PGC_SUSET,
                              0,
                              NULL,
                              aqd_assign_lightgbm_model_path,
                              NULL);
    
    DefineCustomBoolVariable("aqd.enable_dispatch_logging",
                            "Enable AQD dispatch decision logging",
                            "Log all AQD dispatch decisions to file for analysis",
                            &aqd_enable_dispatch_logging,
                            false,
                            PGC_SUSET,
                            0,
                            NULL,
                            aqd_assign_enable_dispatch_logging,
                            NULL);
    
    DefineCustomStringVariable("aqd.dispatch_log_path",
                              "Path to AQD dispatch log file",
                              "File path where AQD dispatch decisions will be logged",
                              &aqd_dispatch_log_path,
                              "",
                              PGC_SUSET,
                              0,
                              NULL,
                              aqd_assign_dispatch_log_path,
                              NULL);

    DefineCustomBoolVariable("aqd.apply_decision",
                            "Apply AQD decision by toggling duckdb.force_execution",
                            "If enabled, AQD sets duckdb.force_execution based on its decision. Disable to avoid GUC changes during execution.",
                            &aqd_apply_decision,
                            false,
                            PGC_SUSET,
                            0,
                            NULL,
                            NULL,
                            NULL);
}
