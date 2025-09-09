/*
 * aqd_query_router.c
 *
 * Implementation of Adaptive Query Dispatcher Query Router
 * Supports multiple routing strategies including cost-threshold, LightGBM, and GNN
 */

#include "postgres.h"
#include "aqd_query_router.h"

#include "optimizer/cost.h"
#include "optimizer/planner.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"
#include "catalog/pg_type.h"
#include "lightgbm_inference.h"

#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

/* Global router instance */
AQDQueryRouter aqd_query_router = {
    .initialized = false,
    .total_queries = 0
};

/* Configuration variables */
int aqd_routing_method = AQD_ROUTE_DEFAULT;
double aqd_cost_threshold = 10000.0;
char *aqd_lightgbm_model_path = NULL;
char *aqd_gnn_model_path = NULL;
bool aqd_enable_thompson_sampling = false;
bool aqd_enable_resource_regulation = false;
double aqd_resource_balance_factor = 0.5;

/* Initialize the query router */
void
aqd_init_query_router(void)
{
    if (aqd_query_router.initialized)
        return;
        
    memset(&aqd_query_router, 0, sizeof(AQDQueryRouter));
    
    /* Set configuration */
    aqd_query_router.config.method = (AQDRoutingMethod)aqd_routing_method;
    aqd_query_router.config.cost_threshold = aqd_cost_threshold;
    aqd_query_router.config.enable_feedback = aqd_enable_thompson_sampling;
    aqd_query_router.config.enable_resource_regulation = aqd_enable_resource_regulation;
    aqd_query_router.config.resource_balance_factor = aqd_resource_balance_factor;
    
    if (aqd_lightgbm_model_path)
    {
        strncpy(aqd_query_router.config.lightgbm_model_path, 
                aqd_lightgbm_model_path, 
                sizeof(aqd_query_router.config.lightgbm_model_path) - 1);
    }
    
    if (aqd_gnn_model_path)
    {
        strncpy(aqd_query_router.config.gnn_model_path, 
                aqd_gnn_model_path, 
                sizeof(aqd_query_router.config.gnn_model_path) - 1);
    }
    
    /* Initialize Thompson sampling bandit */
    if (aqd_query_router.config.enable_feedback)
    {
        aqd_init_thompson_bandit(&aqd_query_router.bandit);
    }
    
    /* Initialize resource regulator */
    if (aqd_query_router.config.enable_resource_regulation)
    {
        aqd_init_resource_regulator(&aqd_query_router.regulator);
    }
    
    /* Load models if specified */
    if (aqd_query_router.config.method == AQD_ROUTE_LIGHTGBM && 
        strlen(aqd_query_router.config.lightgbm_model_path) > 0)
    {
        aqd_load_lightgbm_model(aqd_query_router.config.lightgbm_model_path);
    }
    
    if (aqd_query_router.config.method == AQD_ROUTE_GNN && 
        strlen(aqd_query_router.config.gnn_model_path) > 0)
    {
        aqd_load_gnn_model(aqd_query_router.config.gnn_model_path);
    }
    
    aqd_query_router.initialized = true;
    
    elog(LOG, "AQD Query Router initialized with method: %d", aqd_query_router.config.method);
}

/* Cleanup the query router */
void
aqd_cleanup_query_router(void)
{
    if (!aqd_query_router.initialized)
        return;
        
    /* Unload models */
    aqd_unload_models();
    
    /* Print final statistics */
    aqd_print_router_stats();
    
    aqd_query_router.initialized = false;
    
    elog(LOG, "AQD Query Router cleaned up");
}

/* Main routing function */
AQDRoutingDecision
aqd_route_query(const char *query_text,
                PlannedStmt *planned_stmt,
                AQDQueryFeatures *features)
{
    AQDRoutingDecision decision;
    TimestampTz start_time, end_time;
    AQDExecutionEngine preferred_engine;
    
    if (!aqd_query_router.initialized)
        aqd_init_query_router();
    
    start_time = GetCurrentTimestamp();
    
    /* Initialize decision structure */
    memset(&decision, 0, sizeof(AQDRoutingDecision));
    decision.decision_time = start_time;
    decision.confidence = 1.0;
    
    /* Route based on configured method */
    switch (aqd_query_router.config.method)
    {
        case AQD_ROUTE_DEFAULT:
            preferred_engine = aqd_route_default(query_text, planned_stmt);
            strcpy(decision.reason, "Default pg_duckdb heuristic");
            break;
            
        case AQD_ROUTE_COST_THRESHOLD:
            preferred_engine = aqd_route_cost_threshold(planned_stmt, 
                                                       aqd_query_router.config.cost_threshold);
            snprintf(decision.reason, sizeof(decision.reason), 
                    "Cost threshold (%.1f)", aqd_query_router.config.cost_threshold);
            break;
            
        case AQD_ROUTE_LIGHTGBM:
            if (features && aqd_query_router.lightgbm_model)
            {
                preferred_engine = aqd_route_lightgbm(features);
                strcpy(decision.reason, "LightGBM ML prediction");
                decision.confidence = aqd_compute_routing_confidence(features, preferred_engine);
            }
            else
            {
                preferred_engine = aqd_route_cost_threshold(planned_stmt, 
                                                           aqd_query_router.config.cost_threshold);
                strcpy(decision.reason, "LightGBM fallback to cost threshold");
            }
            break;
            
        case AQD_ROUTE_GNN:
            if (features && aqd_query_router.gnn_model)
            {
                preferred_engine = aqd_route_gnn(planned_stmt, features);
                strcpy(decision.reason, "Graph Neural Network prediction");
                decision.confidence = aqd_compute_routing_confidence(features, preferred_engine);
            }
            else
            {
                preferred_engine = aqd_route_cost_threshold(planned_stmt, 
                                                           aqd_query_router.config.cost_threshold);
                strcpy(decision.reason, "GNN fallback to cost threshold");
            }
            break;
            
        default:
            preferred_engine = AQD_ENGINE_POSTGRES;
            strcpy(decision.reason, "Unknown method, defaulting to PostgreSQL");
            break;
    }
    
    /* Apply Thompson sampling if enabled */
    if (aqd_query_router.config.enable_feedback)
    {
        AQDExecutionEngine sampled_engine = aqd_thompson_sample(&aqd_query_router.bandit);
        if (sampled_engine != preferred_engine)
        {
            preferred_engine = sampled_engine;
            strcat(decision.reason, " + Thompson sampling");
        }
    }
    
    /* Apply resource regulation if enabled */
    if (aqd_query_router.config.enable_resource_regulation)
    {
        aqd_update_resource_usage(&aqd_query_router.regulator);
        AQDExecutionEngine regulated_engine = aqd_apply_resource_regulation(preferred_engine,
                                                                            &aqd_query_router.regulator);
        if (regulated_engine != preferred_engine)
        {
            preferred_engine = regulated_engine;
            strcat(decision.reason, " + resource regulation");
        }
    }
    
    /* Set final decision */
    decision.engine = preferred_engine;
    
    /* Predict execution times if we have features */
    if (features)
    {
        decision.predicted_pg_time = aqd_predict_execution_time(features, AQD_ENGINE_POSTGRES);
        decision.predicted_duck_time = aqd_predict_execution_time(features, AQD_ENGINE_DUCKDB);
    }
    
    /* Calculate decision latency */
    end_time = GetCurrentTimestamp();
    decision.decision_latency_us = (double)(end_time - start_time);
    
    /* Update statistics */
    aqd_query_router.total_queries++;
    aqd_query_router.total_routing_time_us += decision.decision_latency_us;
    
    if (decision.engine == AQD_ENGINE_POSTGRES)
        aqd_query_router.pg_routed_queries++;
    else
        aqd_query_router.duck_routed_queries++;
    
    /* Log decision */
    aqd_log_routing_decision(&decision);
    
    return decision;
}

/* Default routing (pg_duckdb heuristic) */
AQDExecutionEngine
aqd_route_default(const char *query_text, PlannedStmt *planned_stmt)
{
    /* Implement simple heuristics similar to pg_duckdb */
    
    /* DML operations go to PostgreSQL */
    if (planned_stmt->commandType != CMD_SELECT)
        return AQD_ENGINE_POSTGRES;
    
    /* Simple selects with small result sets go to PostgreSQL */
    if (planned_stmt->planTree->plan_rows < 1000)
        return AQD_ENGINE_POSTGRES;
    
    /* Queries with aggregation go to DuckDB */
    if (aqd_count_aggregates(planned_stmt->planTree) > 0)
        return AQD_ENGINE_DUCKDB;
    
    /* Queries with many joins go to DuckDB */
    if (aqd_count_joins(planned_stmt->planTree) > 2)
        return AQD_ENGINE_DUCKDB;
    
    /* High cost queries go to DuckDB */
    if (planned_stmt->planTree->total_cost > 10000.0)
        return AQD_ENGINE_DUCKDB;
    
    /* Default to PostgreSQL */
    return AQD_ENGINE_POSTGRES;
}

/* Cost threshold routing */
AQDExecutionEngine
aqd_route_cost_threshold(PlannedStmt *planned_stmt, double threshold)
{
    if (planned_stmt->planTree->total_cost > threshold)
        return AQD_ENGINE_DUCKDB;
    else
        return AQD_ENGINE_POSTGRES;
}

/* LightGBM-based routing */
AQDExecutionEngine
aqd_route_lightgbm(AQDQueryFeatures *features)
{
    /* Use LightGBM predictor if loaded; otherwise fallback decided by caller */
    if (!features || features->num_features <= 0 || aqd_query_router.lightgbm_model == NULL)
        return AQD_ENGINE_POSTGRES;

    const LightGBMPredictor *predictor = (const LightGBMPredictor *)aqd_query_router.lightgbm_model;
    if (!lightgbm_is_loaded(predictor))
        return AQD_ENGINE_POSTGRES;

    /* Build feature vector from AQD features (truncate to model max) */
    double vec[LIGHTGBM_MAX_FEATURES];
    int n = features->num_features;
    if (n > LIGHTGBM_MAX_FEATURES)
        n = LIGHTGBM_MAX_FEATURES;
    for (int i = 0; i < n; i++)
        vec[i] = features->features[i].is_valid ? features->features[i].value : 0.0;
    for (int i = n; i < LIGHTGBM_MAX_FEATURES; i++)
        vec[i] = 0.0;

    double y = lightgbm_predict(predictor, vec);
    /* Classification threshold: > 0 => DuckDB, else PostgreSQL.
       If regression (log time gap), route to DuckDB if predicted duck faster (negative gap). */
    if (y > 0.0)
        return AQD_ENGINE_DUCKDB;
    else
        return AQD_ENGINE_POSTGRES;
}

/* GNN-based routing */
AQDExecutionEngine
aqd_route_gnn(PlannedStmt *planned_stmt, AQDQueryFeatures *features)
{
    /* This would require actual GNN implementation
     * For now, use plan structure complexity as a heuristic */
    
    if (!planned_stmt || !features)
        return AQD_ENGINE_POSTGRES;
    
    int complexity = features->complexity_score;
    
    /* Use complexity score as a simple decision criterion */
    if (complexity > 20)
        return AQD_ENGINE_DUCKDB;
    else
        return AQD_ENGINE_POSTGRES;
}

/* Initialize Thompson sampling bandit */
void
aqd_init_thompson_bandit(AQDThompsonBandit *bandit)
{
    if (!bandit)
        return;
        
    memset(bandit, 0, sizeof(AQDThompsonBandit));
    
    /* Initialize Beta distribution parameters */
    bandit->pg_alpha = 1.0;
    bandit->pg_beta = 1.0;
    bandit->duck_alpha = 1.0;
    bandit->duck_beta = 1.0;
    
    /* Set exploration parameters */
    bandit->exploration_factor = 0.1;
    bandit->decay_factor = 0.95;
    
    elog(DEBUG1, "AQD: Thompson sampling bandit initialized");
}

/* Sample from Thompson bandit */
AQDExecutionEngine
aqd_thompson_sample(AQDThompsonBandit *bandit)
{
    if (!bandit)
        return AQD_ENGINE_POSTGRES;
    
    /* Sample from Beta distributions */
    double pg_sample = /* beta_sample(bandit->pg_alpha, bandit->pg_beta) */ 0.5; /* Simplified */
    double duck_sample = /* beta_sample(bandit->duck_alpha, bandit->duck_beta) */ 0.5; /* Simplified */
    
    if (pg_sample > duck_sample)
        return AQD_ENGINE_POSTGRES;
    else
        return AQD_ENGINE_DUCKDB;
}

/* Update Thompson bandit with feedback */
void
aqd_update_thompson_bandit(AQDThompsonBandit *bandit, 
                          AQDExecutionEngine engine,
                          double performance)
{
    if (!bandit)
        return;
    
    /* Update performance history */
    if (bandit->history_count < 100)
        bandit->history_count++;
    
    int idx = bandit->history_index % 100;
    
    if (engine == AQD_ENGINE_POSTGRES)
    {
        bandit->pg_recent_performance[idx] = performance;
        
        /* Update Beta parameters based on performance */
        if (performance < 1000.0) /* Good performance */
            bandit->pg_alpha += 1.0;
        else /* Poor performance */
            bandit->pg_beta += 1.0;
    }
    else
    {
        bandit->duck_recent_performance[idx] = performance;
        
        /* Update Beta parameters based on performance */
        if (performance < 1000.0) /* Good performance */
            bandit->duck_alpha += 1.0;
        else /* Poor performance */
            bandit->duck_beta += 1.0;
    }
    
    bandit->history_index = (bandit->history_index + 1) % 100;
    
    /* Apply decay to historical parameters */
    bandit->pg_alpha *= bandit->decay_factor;
    bandit->pg_beta *= bandit->decay_factor;
    bandit->duck_alpha *= bandit->decay_factor;
    bandit->duck_beta *= bandit->decay_factor;
    
    /* Ensure minimum values */
    if (bandit->pg_alpha < 0.1) bandit->pg_alpha = 0.1;
    if (bandit->pg_beta < 0.1) bandit->pg_beta = 0.1;
    if (bandit->duck_alpha < 0.1) bandit->duck_alpha = 0.1;
    if (bandit->duck_beta < 0.1) bandit->duck_beta = 0.1;
}

/* Initialize resource regulator */
void
aqd_init_resource_regulator(AQDResourceRegulator *regulator)
{
    if (!regulator)
        return;
        
    memset(regulator, 0, sizeof(AQDResourceRegulator));
    
    /* Initialize identity covariance matrix */
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            regulator->resource_cov[i][j] = (i == j) ? 1.0 : 0.0;
            regulator->resource_inv_cov[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    regulator->load_balance_threshold = 0.7;
    
    elog(DEBUG1, "AQD: Resource regulator initialized");
}

/* Update resource usage */
void
aqd_update_resource_usage(AQDResourceRegulator *regulator)
{
    if (!regulator)
        return;
    
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        /* Update CPU usage (simplified estimation) */
        double cpu_time = (double)usage.ru_utime.tv_sec + 
                         (double)usage.ru_utime.tv_usec / 1000000.0;
        regulator->pg_cpu_usage = cpu_time * 0.01; /* Simplified */
        
        /* Update memory usage */
        regulator->pg_memory_usage = (double)usage.ru_maxrss / 1024.0; /* KB to MB */
    }
    
    /* Simplified DuckDB resource tracking */
    regulator->duck_cpu_usage = regulator->pg_cpu_usage * 0.8;
    regulator->duck_memory_usage = regulator->pg_memory_usage * 0.6;
}

/* Compute Mahalanobis distance */
double
aqd_compute_mahalanobis_distance(AQDResourceRegulator *regulator,
                                double resource_usage[4])
{
    if (!regulator || !resource_usage)
        return 0.0;
    
    /* Simplified Mahalanobis distance calculation */
    double diff[4];
    for (int i = 0; i < 4; i++)
        diff[i] = resource_usage[i] - regulator->resource_mean[i];
    
    double distance = 0.0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            distance += diff[i] * regulator->resource_inv_cov[i][j] * diff[j];
        }
    }
    
    return sqrt(distance);
}

/* Check if load balancing is needed */
bool
aqd_should_balance_load(AQDResourceRegulator *regulator)
{
    if (!regulator)
        return false;
    
    /* Check CPU usage imbalance */
    double cpu_imbalance = fabs(regulator->pg_cpu_usage - regulator->duck_cpu_usage);
    if (cpu_imbalance > regulator->load_balance_threshold)
        return true;
    
    /* Check memory usage imbalance */
    double mem_imbalance = fabs(regulator->pg_memory_usage - regulator->duck_memory_usage);
    if (mem_imbalance > regulator->load_balance_threshold * 100.0) /* MB */
        return true;
    
    /* Check query load imbalance */
    int total_queries = regulator->pg_active_queries + regulator->duck_active_queries;
    if (total_queries > 0)
    {
        double query_imbalance = fabs((double)regulator->pg_active_queries - 
                                     (double)regulator->duck_active_queries) / total_queries;
        if (query_imbalance > regulator->load_balance_threshold)
            return true;
    }
    
    return false;
}

/* Apply resource regulation */
AQDExecutionEngine
aqd_apply_resource_regulation(AQDExecutionEngine preferred_engine,
                             AQDResourceRegulator *regulator)
{
    if (!regulator || !aqd_should_balance_load(regulator))
        return preferred_engine;
    
    /* Choose engine with lower resource utilization */
    if (regulator->pg_cpu_usage + regulator->pg_memory_usage / 100.0 <
        regulator->duck_cpu_usage + regulator->duck_memory_usage / 100.0)
        return AQD_ENGINE_POSTGRES;
    else
        return AQD_ENGINE_DUCKDB;
}

/* Record execution feedback */
void
aqd_record_execution_feedback(AQDRoutingDecision *decision,
                             double actual_time_ms,
                             AQDExecutionEngine actual_engine)
{
    if (!decision || !aqd_query_router.initialized)
        return;
    
    /* Update Thompson sampling bandit if enabled */
    if (aqd_query_router.config.enable_feedback)
    {
        aqd_update_thompson_bandit(&aqd_query_router.bandit, 
                                  actual_engine, actual_time_ms);
    }
    
    /* Update performance statistics */
    if (actual_engine == AQD_ENGINE_POSTGRES)
    {
        aqd_query_router.avg_pg_time_ms = 
            (aqd_query_router.avg_pg_time_ms * (aqd_query_router.pg_routed_queries - 1) + actual_time_ms) /
            aqd_query_router.pg_routed_queries;
    }
    else
    {
        aqd_query_router.avg_duck_time_ms = 
            (aqd_query_router.avg_duck_time_ms * (aqd_query_router.duck_routed_queries - 1) + actual_time_ms) /
            aqd_query_router.duck_routed_queries;
    }
    
    /* Update routing accuracy */
    bool correct_prediction = (decision->engine == actual_engine);
    if (decision->predicted_pg_time > 0 && decision->predicted_duck_time > 0)
    {
        bool pg_would_be_faster = (decision->predicted_pg_time < decision->predicted_duck_time);
        bool pg_was_actually_used = (actual_engine == AQD_ENGINE_POSTGRES);
        correct_prediction = (pg_would_be_faster == pg_was_actually_used);
    }
    
    aqd_query_router.routing_accuracy = 
        (aqd_query_router.routing_accuracy * (aqd_query_router.total_queries - 1) +
         (correct_prediction ? 1.0 : 0.0)) / aqd_query_router.total_queries;
}

/* Utility functions */

double
aqd_predict_execution_time(AQDQueryFeatures *features, AQDExecutionEngine engine)
{
    if (!features)
        return 1000.0; /* Default estimate */
    
    /* Simplified prediction based on plan cost and complexity */
    double base_time = 100.0;
    
    if (features->planned_stmt)
    {
        base_time = features->planned_stmt->planTree->total_cost / 100.0;
    }
    
    /* Adjust based on engine characteristics */
    if (engine == AQD_ENGINE_DUCKDB)
    {
        /* DuckDB is typically faster for analytical queries */
        if (features->is_olap)
            base_time *= 0.6;
        else
            base_time *= 1.2; /* Slower for OLTP */
    }
    
    return base_time;
}

double
aqd_compute_routing_confidence(AQDQueryFeatures *features,
                              AQDExecutionEngine selected_engine)
{
    if (!features)
        return 0.5;
    
    /* Simple confidence based on feature completeness and complexity */
    double feature_completeness = (double)features->num_features / AQD_MAX_FEATURES;
    double complexity_factor = 1.0 - (features->complexity_score / 100.0);
    
    return (feature_completeness + complexity_factor) / 2.0;
}

void
aqd_log_routing_decision(const AQDRoutingDecision *decision)
{
    if (!decision)
        return;
    
    elog(DEBUG2, "AQD: Routed to %s (confidence: %.2f, latency: %.2fμs) - %s",
         decision->engine == AQD_ENGINE_POSTGRES ? "PostgreSQL" : "DuckDB",
         decision->confidence,
         decision->decision_latency_us,
         decision->reason);
}

/* Model loading (placeholder implementations) */

bool
aqd_load_lightgbm_model(const char *model_path)
{
    if (!model_path || strlen(model_path) == 0)
        return false;

    /* Free existing */
    if (aqd_query_router.lightgbm_model)
    {
        lightgbm_free_predictor((LightGBMPredictor *)aqd_query_router.lightgbm_model);
        aqd_query_router.lightgbm_model = NULL;
    }

    LightGBMPredictor *pred = lightgbm_create_predictor();
    if (!pred)
        return false;
    if (!lightgbm_load_model(pred, model_path))
    {
        lightgbm_free_predictor(pred);
        elog(WARNING, "AQD: Failed to load LightGBM model from %s", model_path);
        return false;
    }
    aqd_query_router.lightgbm_model = (void *)pred;
    elog(LOG, "AQD: LightGBM model loaded from %s", model_path);
    return true;
}

bool
aqd_load_gnn_model(const char *model_path)
{
    if (!model_path)
        return false;
    
    /* TODO: Implement actual GNN model loading */
    elog(LOG, "AQD: GNN model loading from %s (placeholder)", model_path);
    
    return true;
}

void
aqd_unload_models(void)
{
    if (aqd_query_router.lightgbm_model)
    {
        lightgbm_free_predictor((LightGBMPredictor *)aqd_query_router.lightgbm_model);
        aqd_query_router.lightgbm_model = NULL;
    }
    aqd_query_router.gnn_model = NULL;
}

/* Statistics functions */

void
aqd_print_router_stats(void)
{
    if (!aqd_query_router.initialized)
        return;
    
    elog(LOG, "AQD Router Statistics:");
    elog(LOG, "  Total queries: %d", aqd_query_router.total_queries);
    elog(LOG, "  PostgreSQL: %d (%.1f%%)", 
         aqd_query_router.pg_routed_queries,
         (double)aqd_query_router.pg_routed_queries / aqd_query_router.total_queries * 100.0);
    elog(LOG, "  DuckDB: %d (%.1f%%)", 
         aqd_query_router.duck_routed_queries,
         (double)aqd_query_router.duck_routed_queries / aqd_query_router.total_queries * 100.0);
    elog(LOG, "  Average routing time: %.2fμs", 
         aqd_query_router.total_routing_time_us / aqd_query_router.total_queries);
    elog(LOG, "  Routing accuracy: %.2f%%", aqd_query_router.routing_accuracy * 100.0);
    elog(LOG, "  Avg PostgreSQL time: %.2fms", aqd_query_router.avg_pg_time_ms);
    elog(LOG, "  Avg DuckDB time: %.2fms", aqd_query_router.avg_duck_time_ms);
}

/* GUC variable definitions */

void
aqd_define_routing_guc_variables(void)
{
    DefineCustomIntVariable("aqd.routing_method",
                           "AQD routing method (0=default, 1=cost, 2=lightgbm, 3=gnn)",
                           "Selects the query routing algorithm to use",
                           &aqd_routing_method,
                           AQD_ROUTE_DEFAULT,
                           AQD_ROUTE_DEFAULT, AQD_ROUTE_GNN,
                           PGC_SUSET,
                           0,
                           NULL, NULL, NULL);
                           
    DefineCustomRealVariable("aqd.cost_threshold",
                            "Cost threshold for routing to DuckDB",
                            "Queries with cost above this threshold are routed to DuckDB",
                            &aqd_cost_threshold,
                            10000.0,
                            0.0, 1000000.0,
                            PGC_SUSET,
                            0,
                            NULL, NULL, NULL);
                            
    DefineCustomStringVariable("aqd.lightgbm_model_path",
                              "Path to LightGBM model file",
                              "File path for the trained LightGBM routing model",
                              &aqd_lightgbm_model_path,
                              "",
                              PGC_SUSET,
                              0,
                              NULL, NULL, NULL);
                              
    DefineCustomStringVariable("aqd.gnn_model_path",
                              "Path to GNN model file",
                              "File path for the trained GNN routing model",
                              &aqd_gnn_model_path,
                              "",
                              PGC_SUSET,
                              0,
                              NULL, NULL, NULL);
                              
    DefineCustomBoolVariable("aqd.enable_thompson_sampling",
                            "Enable Thompson sampling bandit learning",
                            "Use online reinforcement learning for routing decisions",
                            &aqd_enable_thompson_sampling,
                            false,
                            PGC_SUSET,
                            0,
                            NULL, NULL, NULL);
                            
    DefineCustomBoolVariable("aqd.enable_resource_regulation",
                            "Enable Mahalanobis resource regulation",
                            "Use resource usage for load balancing decisions",
                            &aqd_enable_resource_regulation,
                            false,
                            PGC_SUSET,
                            0,
                            NULL, NULL, NULL);
                            
    DefineCustomRealVariable("aqd.resource_balance_factor",
                            "Resource balance factor for load regulation",
                            "Threshold for triggering resource-based load balancing",
                            &aqd_resource_balance_factor,
                            0.5,
                            0.0, 1.0,
                            PGC_SUSET,
                            0,
                            NULL, NULL, NULL);
}

/* Configuration functions */

void
aqd_set_routing_method(AQDRoutingMethod method)
{
    aqd_routing_method = (int)method;
    if (aqd_query_router.initialized)
    {
        aqd_query_router.config.method = method;
        elog(LOG, "AQD: Routing method changed to %d", method);
    }
}

AQDRoutingMethod
aqd_get_routing_method(void)
{
    return (AQDRoutingMethod)aqd_routing_method;
}
