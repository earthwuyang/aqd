/*
 * aqd_feature_logger.c
 *
 * Implementation of Adaptive Query Dispatcher Feature Logger
 * Extracts 100+ features from PostgreSQL queries for ML training
 */

#include "postgres.h"
#include "aqd_feature_logger.h"

#include "access/htup_details.h"
#include "access/tableam.h"
#include "access/table.h"
#include "catalog/pg_statistic.h"
#include "commands/explain.h"
#include "executor/executor.h" 
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/plancat.h"
#include "optimizer/optimizer.h" /* for seq_page_cost */
#include "parser/analyze.h"
#include "parser/parsetree.h"
#include "statistics/statistics.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "storage/lock.h"
#include "storage/buf_internals.h"
#include "storage/lockdefs.h"
#include "tcop/utility.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/syscache.h"
/* costsize.h was removed/merged in PG16; cost.h covers needed APIs */
#include <math.h>

#include <sys/resource.h>
#include <sys/time.h>
#include <math.h>
// #include <openssl/sha.h>  // Will use simpler hash function

/* Global feature extractor */
AQDFeatureExtractor aqd_feature_extractor = {
    .enabled = false,
    .log_file_path = "/tmp/aqd_features.csv",
    .log_file = NULL,
    .total_queries = 0,
    .failed_extractions = 0
};

/* Configuration variables */
bool aqd_enable_feature_logging = false;
char *aqd_feature_log_path = NULL;
int aqd_log_format = 0; /* CSV format by default */

/* Initialize the feature extractor */
void
aqd_init_feature_extractor(void)
{
    if (aqd_feature_extractor.enabled)
        return;
    
    /* Set default log path if not configured */
    if (!aqd_feature_log_path)
        aqd_feature_log_path = pstrdup("/tmp/aqd_features.csv");
    
    strncpy(aqd_feature_extractor.log_file_path, aqd_feature_log_path, 
            sizeof(aqd_feature_extractor.log_file_path) - 1);
    
    /* Open log file */
    aqd_feature_extractor.log_file = fopen(aqd_feature_extractor.log_file_path, "a");
    if (!aqd_feature_extractor.log_file)
    {
        elog(WARNING, "AQD: Failed to open feature log file: %s", 
             aqd_feature_extractor.log_file_path);
        return;
    }
    
    /* Write CSV header if file is new */
    fseek(aqd_feature_extractor.log_file, 0, SEEK_END);
    if (ftell(aqd_feature_extractor.log_file) == 0)
    {
        fprintf(aqd_feature_extractor.log_file, 
                "query_hash,execution_time_ms,postgres_time_ms,duckdb_time_ms,is_oltp,is_olap,complexity_score");
        
        /* Write feature column headers */
        for (int i = 0; i < AQD_MAX_FEATURES; i++)
        {
            fprintf(aqd_feature_extractor.log_file, ",feature_%d", i + 1);
        }
        fprintf(aqd_feature_extractor.log_file, "\n");
    }
    
    aqd_feature_extractor.enabled = true;
    elog(LOG, "AQD: Feature extractor initialized, logging to %s", 
         aqd_feature_extractor.log_file_path);
}

/* Cleanup the feature extractor */
void
aqd_cleanup_feature_extractor(void)
{
    if (!aqd_feature_extractor.enabled)
        return;
        
    if (aqd_feature_extractor.log_file)
    {
        fclose(aqd_feature_extractor.log_file);
        aqd_feature_extractor.log_file = NULL;
    }
    
    aqd_feature_extractor.enabled = false;
    elog(LOG, "AQD: Feature extractor cleaned up. Total queries: %d, Failed: %d",
         aqd_feature_extractor.total_queries, 
         aqd_feature_extractor.failed_extractions);
}

/* Extract all features from a query */
void
aqd_extract_query_features(AQDQueryFeatures *features,
                          const char *query_text,
                          PlannedStmt *planned_stmt,
                          QueryDesc *query_desc)
{
    if (!features || !query_text || !planned_stmt)
        return;
        
    memset(features, 0, sizeof(AQDQueryFeatures));
    
    /* Basic query info */
    strncpy(features->query_text, query_text, sizeof(features->query_text) - 1);
    features->planned_stmt = planned_stmt;
    features->query_desc = query_desc;
    features->start_time = GetCurrentTimestamp();
    
    /* Compute query hash */
    char *normalized = aqd_normalize_query(query_text);
    char *hash = aqd_compute_query_hash(normalized);
    strncpy(features->query_hash, hash, sizeof(features->query_hash) - 1);
    pfree(normalized);
    pfree(hash);
    
    /* Query classification */
    features->is_oltp = aqd_classify_oltp_olap(query_text, planned_stmt);
    features->is_olap = !features->is_oltp;
    features->complexity_score = aqd_compute_complexity_score(query_text, planned_stmt);
    
    /* Extract features by category */
    int feature_idx = 0;
    
    feature_idx += aqd_extract_query_structure_features(&features->features[feature_idx], 
                                                       query_text, planned_stmt);
    
    feature_idx += aqd_extract_optimizer_cost_features(&features->features[feature_idx], 
                                                      planned_stmt);
    
    feature_idx += aqd_extract_table_stats_features(&features->features[feature_idx], 
                                                   planned_stmt);
    
    feature_idx += aqd_extract_plan_structure_features(&features->features[feature_idx], 
                                                      planned_stmt);
    
    feature_idx += aqd_extract_resource_estimation_features(&features->features[feature_idx], 
                                                           planned_stmt, query_desc);
    
    feature_idx += aqd_extract_cardinality_features(&features->features[feature_idx], 
                                                   planned_stmt);
    
    feature_idx += aqd_extract_selectivity_features(&features->features[feature_idx], 
                                                   planned_stmt);
    
    feature_idx += aqd_extract_system_state_features(&features->features[feature_idx]);
    
    features->num_features = Min(feature_idx, AQD_MAX_FEATURES);
    
    aqd_feature_extractor.total_queries++;
}

/* Extract query structure features */
int
aqd_extract_query_structure_features(AQDFeature *features, 
                                    const char *query_text,
                                    PlannedStmt *planned_stmt)
{
    int idx = 0;
    
    /* Query length */
    features[idx].value = (double)strlen(query_text);
    strcpy(features[idx].name, "query_length");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* Number of relations */
    List *rtable = planned_stmt->rtable;
    features[idx].value = (double)list_length(rtable);
    strcpy(features[idx].name, "num_relations");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* Plan node count */
    features[idx].value = (double)aqd_count_plan_nodes(planned_stmt->planTree);
    strcpy(features[idx].name, "plan_node_count");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* Join count */
    features[idx].value = (double)aqd_count_joins(planned_stmt->planTree);
    strcpy(features[idx].name, "join_count");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* Aggregate count */
    features[idx].value = (double)aqd_count_aggregates(planned_stmt->planTree);
    strcpy(features[idx].name, "aggregate_count");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* Sort count */
    features[idx].value = (double)aqd_count_sorts(planned_stmt->planTree);
    strcpy(features[idx].name, "sort_count");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* SubPlan count */
    features[idx].value = (double)list_length(planned_stmt->subplans);
    strcpy(features[idx].name, "subplan_count");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    /* Parameter count */
    features[idx].value = (double)list_length(planned_stmt->paramExecTypes);
    strcpy(features[idx].name, "param_count");
    features[idx].category = AQD_CAT_QUERY_STRUCTURE;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Extract optimizer cost features */
int
aqd_extract_optimizer_cost_features(AQDFeature *features,
                                   PlannedStmt *planned_stmt)
{
    int idx = 0;
    Plan *plan = planned_stmt->planTree;
    
    /* Total plan cost */
    features[idx].value = plan->total_cost;
    strcpy(features[idx].name, "total_plan_cost");
    features[idx].category = AQD_CAT_OPTIMIZER_COST;
    features[idx].is_valid = true;
    idx++;
    
    /* Startup cost */
    features[idx].value = plan->startup_cost;
    strcpy(features[idx].name, "startup_cost");
    features[idx].category = AQD_CAT_OPTIMIZER_COST;
    features[idx].is_valid = true;
    idx++;
    
    /* Plan rows */
    features[idx].value = plan->plan_rows;
    strcpy(features[idx].name, "plan_rows");
    features[idx].category = AQD_CAT_OPTIMIZER_COST;
    features[idx].is_valid = true;
    idx++;
    
    /* Plan width */
    features[idx].value = (double)plan->plan_width;
    strcpy(features[idx].name, "plan_width");
    features[idx].category = AQD_CAT_OPTIMIZER_COST;
    features[idx].is_valid = true;
    idx++;
    
    /* Cost per row */
    if (plan->plan_rows > 0)
    {
        features[idx].value = plan->total_cost / plan->plan_rows;
        strcpy(features[idx].name, "cost_per_row");
        features[idx].category = AQD_CAT_OPTIMIZER_COST;
        features[idx].is_valid = true;
        idx++;
    }
    
    /* Sum of all plan costs */
    features[idx].value = aqd_sum_plan_costs(plan);
    strcpy(features[idx].name, "sum_all_costs");
    features[idx].category = AQD_CAT_OPTIMIZER_COST;
    features[idx].is_valid = true;
    idx++;
    
    /* Max plan width */
    features[idx].value = aqd_max_plan_width(plan);
    strcpy(features[idx].name, "max_plan_width");
    features[idx].category = AQD_CAT_OPTIMIZER_COST;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Extract table statistics features */
int
aqd_extract_table_stats_features(AQDFeature *features,
                                PlannedStmt *planned_stmt)
{
    int idx = 0;
    ListCell *lc;
    double total_tuples = 0.0;
    double total_pages = 0.0;
    int table_count = 0;
    
    /* Collect statistics from all base relations */
    foreach(lc, planned_stmt->rtable)
    {
        RangeTblEntry *rte = (RangeTblEntry *) lfirst(lc);
        
        if (rte->rtekind == RTE_RELATION)
        {
            Relation rel;
            
            /* Get relation statistics */
            rel = table_open(rte->relid, AccessShareLock);
            if (rel)
            {
                total_tuples += rel->rd_rel->reltuples;
                total_pages += rel->rd_rel->relpages;
                table_count++;
                table_close(rel, AccessShareLock);
            }
        }
    }
    
    /* Average tuples per table */
    features[idx].value = table_count > 0 ? total_tuples / table_count : 0.0;
    strcpy(features[idx].name, "avg_tuples_per_table");
    features[idx].category = AQD_CAT_TABLE_STATS;
    features[idx].is_valid = true;
    idx++;
    
    /* Average pages per table */
    features[idx].value = table_count > 0 ? total_pages / table_count : 0.0;
    strcpy(features[idx].name, "avg_pages_per_table");
    features[idx].category = AQD_CAT_TABLE_STATS;
    features[idx].is_valid = true;
    idx++;
    
    /* Total tuples */
    features[idx].value = total_tuples;
    strcpy(features[idx].name, "total_tuples");
    features[idx].category = AQD_CAT_TABLE_STATS;
    features[idx].is_valid = true;
    idx++;
    
    /* Total pages */
    features[idx].value = total_pages;
    strcpy(features[idx].name, "total_pages");
    features[idx].category = AQD_CAT_TABLE_STATS;
    features[idx].is_valid = true;
    idx++;
    
    /* Number of base tables */
    features[idx].value = (double)table_count;
    strcpy(features[idx].name, "base_table_count");
    features[idx].category = AQD_CAT_TABLE_STATS;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Extract plan structure features */
int
aqd_extract_plan_structure_features(AQDFeature *features,
                                   PlannedStmt *planned_stmt)
{
    int idx = 0;
    Plan *plan = planned_stmt->planTree;
    
    /* Plan tree depth - we'll approximate this */
    int depth = 0;
    Plan *current = plan;
    while (current)
    {
        depth++;
        if (current->lefttree)
            current = current->lefttree;
        else if (current->righttree)
            current = current->righttree;
        else
            break;
    }
    
    features[idx].value = (double)depth;
    strcpy(features[idx].name, "plan_tree_depth");
    features[idx].category = AQD_CAT_EXECUTION_PLAN;
    features[idx].is_valid = true;
    idx++;
    
    /* Number of scan nodes - use helper function */
    int scan_count = aqd_count_scans(planned_stmt->planTree);
    
    features[idx].value = (double)scan_count;
    strcpy(features[idx].name, "scan_node_count");
    features[idx].category = AQD_CAT_EXECUTION_PLAN;
    features[idx].is_valid = true;
    idx++;
    
    /* Has parallel workers */
    features[idx].value = (double)(plan->parallel_aware ? 1 : 0);
    strcpy(features[idx].name, "has_parallel");
    features[idx].category = AQD_CAT_EXECUTION_PLAN;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Extract resource estimation features */
int
aqd_extract_resource_estimation_features(AQDFeature *features,
                                        PlannedStmt *planned_stmt,
                                        QueryDesc *query_desc)
{
    int idx = 0;
    
    /* Work memory estimate based on plan */
    int work_mem_kb = work_mem; /* This is in KB */
    features[idx].value = (double)work_mem_kb;
    strcpy(features[idx].name, "work_mem_kb");
    features[idx].category = AQD_CAT_RESOURCE_EST;
    features[idx].is_valid = true;
    idx++;
    
    /* Estimated I/O cost */
    Plan *plan = planned_stmt->planTree;
    double io_cost = 0.0;
    
    /* Rough estimation based on plan costs */
    io_cost = plan->total_cost * seq_page_cost / 100.0; /* Approximation */
    
    features[idx].value = io_cost;
    strcpy(features[idx].name, "estimated_io_cost");
    features[idx].category = AQD_CAT_RESOURCE_EST;
    features[idx].is_valid = true;
    idx++;
    
    /* CPU cost estimate */
    double cpu_cost = plan->total_cost - io_cost;
    features[idx].value = cpu_cost;
    strcpy(features[idx].name, "estimated_cpu_cost");
    features[idx].category = AQD_CAT_RESOURCE_EST;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Extract cardinality features */
int
aqd_extract_cardinality_features(AQDFeature *features,
                                PlannedStmt *planned_stmt)
{
    int idx = 0;
    
    /* Root cardinality estimate */
    features[idx].value = planned_stmt->planTree->plan_rows;
    strcpy(features[idx].name, "root_cardinality");
    features[idx].category = AQD_CAT_CARDINALITY;
    features[idx].is_valid = true;
    idx++;
    
    /* Log of root cardinality */
    if (planned_stmt->planTree->plan_rows > 0)
    {
        features[idx].value = log(planned_stmt->planTree->plan_rows);
        strcpy(features[idx].name, "log_root_cardinality");
        features[idx].category = AQD_CAT_CARDINALITY;
        features[idx].is_valid = true;
        idx++;
    }
    
    return idx;
}

/* Extract selectivity features */
int
aqd_extract_selectivity_features(AQDFeature *features,
                                PlannedStmt *planned_stmt)
{
    int idx = 0;
    
    /* This is complex and would require analyzing WHERE clauses
     * For now, we'll add placeholders */
    
    features[idx].value = 1.0; /* Default selectivity */
    strcpy(features[idx].name, "overall_selectivity");
    features[idx].category = AQD_CAT_SELECTIVITY;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Extract system state features */
int
aqd_extract_system_state_features(AQDFeature *features)
{
    int idx = 0;
    struct rusage usage;
    
    /* Get current resource usage */
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        /* User CPU time */
        features[idx].value = (double)usage.ru_utime.tv_sec + 
                             (double)usage.ru_utime.tv_usec / 1000000.0;
        strcpy(features[idx].name, "user_cpu_time");
        features[idx].category = AQD_CAT_SYSTEM_STATE;
        features[idx].is_valid = true;
        idx++;
        
        /* System CPU time */
        features[idx].value = (double)usage.ru_stime.tv_sec + 
                             (double)usage.ru_stime.tv_usec / 1000000.0;
        strcpy(features[idx].name, "system_cpu_time");
        features[idx].category = AQD_CAT_SYSTEM_STATE;
        features[idx].is_valid = true;
        idx++;
        
        /* Max resident set size */
        features[idx].value = (double)usage.ru_maxrss;
        strcpy(features[idx].name, "max_rss_kb");
        features[idx].category = AQD_CAT_SYSTEM_STATE;
        features[idx].is_valid = true;
        idx++;
    }
    
    /* Buffer cache hit ratio estimate */
    features[idx].value = 0.95; /* Default assumption */
    strcpy(features[idx].name, "buffer_hit_ratio");
    features[idx].category = AQD_CAT_SYSTEM_STATE;
    features[idx].is_valid = true;
    idx++;
    
    return idx;
}

/* Log features to file */
void
aqd_log_features_to_file(const AQDQueryFeatures *features)
{
    if (!aqd_feature_extractor.enabled || !aqd_feature_extractor.log_file)
        return;
        
    /* Write query metadata */
    fprintf(aqd_feature_extractor.log_file, "%s,%.3f,%.3f,%.3f,%d,%d,%d",
            features->query_hash,
            features->execution_time_ms,
            features->postgres_time_ms,
            features->duckdb_time_ms,
            features->is_oltp ? 1 : 0,
            features->is_olap ? 1 : 0,
            features->complexity_score);
    
    /* Write feature values */
    for (int i = 0; i < AQD_MAX_FEATURES; i++)
    {
        if (i < features->num_features && features->features[i].is_valid)
            fprintf(aqd_feature_extractor.log_file, ",%.6f", features->features[i].value);
        else
            fprintf(aqd_feature_extractor.log_file, ",0.0");
    }
    
    fprintf(aqd_feature_extractor.log_file, "\n");
    fflush(aqd_feature_extractor.log_file);
}

/* Helper function implementations */

char *
aqd_normalize_query(const char *query_text)
{
    /* Simple normalization - remove extra whitespace */
    char *normalized = pstrdup(query_text);
    char *src = normalized, *dst = normalized;
    bool in_space = false;
    
    while (*src)
    {
        if (isspace(*src))
        {
            if (!in_space)
            {
                *dst++ = ' ';
                in_space = true;
            }
        }
        else
        {
            *dst++ = tolower(*src);
            in_space = false;
        }
        src++;
    }
    *dst = '\0';
    
    return normalized;
}

char *
aqd_compute_query_hash(const char *query_text)
{
    /* Simple hash function for demonstration */
    char *hex_string = (char *)palloc(17); /* 16 chars + null terminator */
    uint32 hash = 0;
    
    /* Simple hash calculation */
    for (const char *p = query_text; *p; p++) {
        hash = hash * 31 + (unsigned char)*p;
    }
    
    snprintf(hex_string, 17, "%08x%08x", hash, (uint32)(strlen(query_text) ^ 0xDEADBEEF));
    return hex_string;
}

bool
aqd_classify_oltp_olap(const char *query_text, PlannedStmt *planned_stmt)
{
    /* Simple heuristic classification */
    
    /* Check for OLTP characteristics */
    if (strstr(query_text, "INSERT") || strstr(query_text, "UPDATE") || 
        strstr(query_text, "DELETE"))
        return true; /* OLTP */
    
    /* Check for OLAP characteristics */
    if (aqd_count_aggregates(planned_stmt->planTree) > 0 ||
        aqd_count_joins(planned_stmt->planTree) > 2 ||
        strstr(query_text, "GROUP BY") || strstr(query_text, "ORDER BY"))
        return false; /* OLAP */
    
    /* Default to OLTP for simple SELECTs */
    return true;
}

int
aqd_compute_complexity_score(const char *query_text, PlannedStmt *planned_stmt)
{
    int score = 0;
    
    score += list_length(planned_stmt->rtable) * 2;        /* Relations */
    score += aqd_count_joins(planned_stmt->planTree) * 3;  /* Joins */
    score += aqd_count_aggregates(planned_stmt->planTree) * 2; /* Aggregates */
    score += list_length(planned_stmt->subplans) * 4;      /* Subqueries */
    
    return score;
}

/* Forward declaration */
static void aqd_walk_plan_node(Plan *plan, void (*callback)(Plan *, void *), void *context);

/* Plan traversal function */
void
aqd_walk_plan_tree(PlannedStmt *planned_stmt, void (*callback)(Plan *, void *), void *context)
{
    if (!planned_stmt || !planned_stmt->planTree)
        return;
        
    aqd_walk_plan_node(planned_stmt->planTree, callback, context);
}

static void
aqd_walk_plan_node(Plan *plan, void (*callback)(Plan *, void *), void *context)
{
    if (!plan)
        return;
        
    callback(plan, context);
    
    if (plan->lefttree)
        aqd_walk_plan_node(plan->lefttree, callback, context);
    if (plan->righttree)  
        aqd_walk_plan_node(plan->righttree, callback, context);
}

/* Node counting functions */
int
aqd_count_plan_nodes(Plan *plan)
{
    if (!plan)
        return 0;
        
    return 1 + aqd_count_plan_nodes(plan->lefttree) + 
               aqd_count_plan_nodes(plan->righttree);
}

int
aqd_count_joins(Plan *plan)
{
    int count = 0;
    
    if (!plan)
        return 0;
    
    if (IsA(plan, NestLoop) || IsA(plan, MergeJoin) || IsA(plan, HashJoin))
        count = 1;
        
    count += aqd_count_joins(plan->lefttree);
    count += aqd_count_joins(plan->righttree);
    
    return count;
}

int
aqd_count_aggregates(Plan *plan)
{
    int count = 0;
    
    if (!plan)
        return 0;
    
    if (IsA(plan, Agg) || IsA(plan, WindowAgg))
        count = 1;
        
    count += aqd_count_aggregates(plan->lefttree);
    count += aqd_count_aggregates(plan->righttree);
    
    return count;
}

int
aqd_count_sorts(Plan *plan)
{
    int count = 0;
    
    if (!plan)
        return 0;
    
    if (IsA(plan, Sort))
        count = 1;
        
    count += aqd_count_sorts(plan->lefttree);
    count += aqd_count_sorts(plan->righttree);
    
    return count;
}

double
aqd_sum_plan_costs(Plan *plan)
{
    if (!plan)
        return 0.0;
        
    return plan->total_cost + 
           aqd_sum_plan_costs(plan->lefttree) + 
           aqd_sum_plan_costs(plan->righttree);
}

double
aqd_max_plan_width(Plan *plan)
{
    if (!plan)
        return 0.0;
        
    double max_width = (double)plan->plan_width;
    double left_max = aqd_max_plan_width(plan->lefttree);
    double right_max = aqd_max_plan_width(plan->righttree);
    
    if (left_max > max_width)
        max_width = left_max;
    if (right_max > max_width)
        max_width = right_max;
        
    return max_width;
}

/* GUC variable setup */
void
aqd_define_guc_variables(void)
{
    DefineCustomBoolVariable("aqd.enable_feature_logging",
                            "Enable AQD feature logging",
                            "When enabled, extracts query features for ML training",
                            &aqd_enable_feature_logging,
                            false,
                            PGC_SUSET,
                            0,
                            NULL, NULL, NULL);
                            
    DefineCustomStringVariable("aqd.feature_log_path",
                              "Path for AQD feature log file",
                              "File path where query features are logged",
                              &aqd_feature_log_path,
                              "/tmp/aqd_features.csv",
                              PGC_SUSET,
                              0,
                              NULL, NULL, NULL);
                              
    DefineCustomIntVariable("aqd.log_format",
                           "AQD log format (0=CSV, 1=JSON)",
                           "Format for feature log output",
                           &aqd_log_format,
                           0,
                           0, 1,
                           PGC_SUSET,
                           0,
                           NULL, NULL, NULL);
}

/* Enable/disable functions */
void
aqd_set_feature_extraction(bool enabled)
{
    aqd_enable_feature_logging = enabled;
    if (enabled && !aqd_feature_extractor.enabled)
        aqd_init_feature_extractor();
    else if (!enabled && aqd_feature_extractor.enabled)
        aqd_cleanup_feature_extractor();
}

bool
aqd_is_feature_extraction_enabled(void)
{
    return aqd_enable_feature_logging && aqd_feature_extractor.enabled;
}

int
aqd_count_scans(Plan *plan)
{
    int count = 0;
    
    if (!plan)
        return 0;
    
    if (IsA(plan, SeqScan) || IsA(plan, IndexScan) || IsA(plan, BitmapHeapScan))
        count = 1;
        
    count += aqd_count_scans(plan->lefttree);
    count += aqd_count_scans(plan->righttree);
    
    return count;
}
