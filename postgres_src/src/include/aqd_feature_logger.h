/*
 * aqd_feature_logger.h
 *
 * Adaptive Query Dispatcher (AQD) Feature Logger
 * Logs 100+ query execution features for ML model training
 * 
 * Based on AQD paper: https://github.com/thuml/AQD
 */

#ifndef AQD_FEATURE_LOGGER_H
#define AQD_FEATURE_LOGGER_H

#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/execnodes.h"
#include "executor/instrument.h"
#include "commands/explain.h"
#include "utils/timestamp.h"
#include "tcop/tcopprot.h"

/* Maximum number of features we extract */
#define AQD_MAX_FEATURES 150

/* Feature categories */
typedef enum AQDFeatureCategory
{
    AQD_CAT_QUERY_STRUCTURE = 0,    /* Query complexity metrics */
    AQD_CAT_OPTIMIZER_COST = 1,     /* Cost estimation features */
    AQD_CAT_TABLE_STATS = 2,        /* Table statistics */
    AQD_CAT_EXECUTION_PLAN = 3,     /* Plan structure features */
    AQD_CAT_RESOURCE_EST = 4,       /* Resource estimation */
    AQD_CAT_CARDINALITY = 5,        /* Cardinality estimates */
    AQD_CAT_SELECTIVITY = 6,        /* Selectivity estimates */
    AQD_CAT_SYSTEM_STATE = 7        /* System resource state */
} AQDFeatureCategory;

/* Feature structure */
typedef struct AQDFeature
{
    char name[64];              /* Feature name */
    double value;               /* Feature value */
    AQDFeatureCategory category; /* Feature category */
    bool is_valid;              /* Whether feature is valid */
} AQDFeature;

/* Query feature collection */
typedef struct AQDQueryFeatures
{
    char query_hash[65];        /* SHA256 hash of normalized query */
    char query_text[8192];      /* Original query text */
    TimestampTz start_time;     /* Query start time */
    TimestampTz end_time;       /* Query end time */
    double execution_time_ms;   /* Execution time in milliseconds */
    double postgres_time_ms;    /* PostgreSQL execution time */
    double duckdb_time_ms;      /* DuckDB execution time (if available) */
    int num_features;           /* Number of features extracted */
    AQDFeature features[AQD_MAX_FEATURES]; /* Feature array */
    
    /* Plan information */
    PlannedStmt *planned_stmt;  /* Planned statement */
    QueryDesc *query_desc;      /* Query descriptor */
    
    /* Execution statistics */
    BufferUsage buffer_usage;   /* Buffer usage stats */
    WalUsage wal_usage;         /* WAL usage stats */
    
    /* Query classification */
    bool is_oltp;               /* True if OLTP query */
    bool is_olap;               /* True if OLAP query */
    int complexity_score;       /* Query complexity score */
} AQDQueryFeatures;

/* Global feature extractor state */
typedef struct AQDFeatureExtractor
{
    bool enabled;               /* Whether feature extraction is enabled */
    char log_file_path[1024];   /* Path to feature log file */
    FILE *log_file;             /* Log file handle */
    char plan_log_file_path[1024]; /* Path to JSON plan log file (JSONL) */
    FILE *plan_log_file;        /* Plan log file handle */
    int total_queries;          /* Total queries processed */
    int failed_extractions;     /* Failed feature extractions */
} AQDFeatureExtractor;

/* Global feature extractor instance */
extern AQDFeatureExtractor aqd_feature_extractor;

/* Function declarations */

/* Initialize the feature extractor */
extern void aqd_init_feature_extractor(void);

/* Cleanup the feature extractor */
extern void aqd_cleanup_feature_extractor(void);

/* Extract all features from a query */
extern void aqd_extract_query_features(AQDQueryFeatures *features,
                                      const char *query_text,
                                      PlannedStmt *planned_stmt,
                                      QueryDesc *query_desc);

/* Log features to file */
extern void aqd_log_features_to_file(const AQDQueryFeatures *features);

/* Enable/disable feature extraction */
extern void aqd_set_feature_extraction(bool enabled);

/* Get current feature extraction status */
extern bool aqd_is_feature_extraction_enabled(void);

/* Feature extraction functions by category */
extern int aqd_extract_query_structure_features(AQDFeature *features, 
                                               const char *query_text,
                                               PlannedStmt *planned_stmt);

extern int aqd_extract_optimizer_cost_features(AQDFeature *features,
                                              PlannedStmt *planned_stmt);

extern int aqd_extract_table_stats_features(AQDFeature *features,
                                           PlannedStmt *planned_stmt);

extern int aqd_extract_plan_structure_features(AQDFeature *features,
                                              PlannedStmt *planned_stmt);

extern int aqd_extract_resource_estimation_features(AQDFeature *features,
                                                   PlannedStmt *planned_stmt,
                                                   QueryDesc *query_desc);

extern int aqd_extract_cardinality_features(AQDFeature *features,
                                           PlannedStmt *planned_stmt);

extern int aqd_extract_selectivity_features(AQDFeature *features,
                                           PlannedStmt *planned_stmt);

extern int aqd_extract_system_state_features(AQDFeature *features);

/* Helper functions */
extern char *aqd_normalize_query(const char *query_text);
extern char *aqd_compute_query_hash(const char *query_text);
extern bool aqd_classify_oltp_olap(const char *query_text, PlannedStmt *planned_stmt);
extern int aqd_compute_complexity_score(const char *query_text, PlannedStmt *planned_stmt);
extern void aqd_walk_plan_tree(PlannedStmt *planned_stmt, void (*callback)(Plan *, void *), void *context);

/* Plan node visitor functions */
extern int aqd_count_plan_nodes(Plan *plan);
extern int aqd_count_joins(Plan *plan);
extern int aqd_count_aggregates(Plan *plan);
extern int aqd_count_sorts(Plan *plan);
extern int aqd_count_scans(Plan *plan);
extern double aqd_sum_plan_costs(Plan *plan);
extern double aqd_max_plan_width(Plan *plan);

/* Configuration variables */
extern bool aqd_enable_feature_logging;
extern char *aqd_feature_log_path;
extern int aqd_log_format; /* 0=CSV, 1=JSON */
extern char *aqd_plan_log_path;  /* Path to JSONL plans */

/* GUC setup function */
extern void aqd_define_guc_variables(void);

/* Plan JSON logging */
extern void aqd_log_plan_json(QueryDesc *query_desc, const AQDQueryFeatures *features);

#endif /* AQD_FEATURE_LOGGER_H */
