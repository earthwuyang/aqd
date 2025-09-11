/*
 * plan_logger.c
 *
 * PostgreSQL extension for logging optimizer plans in JSON format
 * Controlled by GUC variable: plan_logger.enable
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/executor.h"
#include "commands/explain.h"
#include "utils/guc.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "tcop/tcopprot.h"
#include "miscadmin.h"
#include "catalog/pg_database.h"
#include "access/htup_details.h"
#include "utils/syscache.h"
#include <unistd.h>

PG_MODULE_MAGIC;

/* GUC variables */
static bool plan_logger_enabled = false;
static char *plan_logger_path = NULL;
static char *plan_logger_format = NULL;

/* File handle for plan logging */
static FILE *plan_log_file = NULL;

/* Previous executor hooks */
static ExecutorStart_hook_type prev_ExecutorStart = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd = NULL;

/* Function prototypes */
void _PG_init(void);
void _PG_fini(void);
static void plan_logger_ExecutorStart(QueryDesc *queryDesc, int eflags);
static void plan_logger_ExecutorEnd(QueryDesc *queryDesc);
static void log_query_plan(QueryDesc *queryDesc);
static void open_log_file(void);
static void close_log_file(void);

/*
 * Module initialization
 */
void
_PG_init(void)
{
    /* Define GUC variables */
    DefineCustomBoolVariable("plan_logger.enable",
                            "Enable query plan logging",
                            "When enabled, logs query plans in JSON format",
                            &plan_logger_enabled,
                            false,
                            PGC_SUSET,
                            0,
                            NULL, NULL, NULL);

    DefineCustomStringVariable("plan_logger.path",
                              "Path for plan log file",
                              "File path where query plans are logged",
                              &plan_logger_path,
                              "/tmp/postgres_plans.jsonl",
                              PGC_SUSET,
                              0,
                              NULL, NULL, NULL);

    DefineCustomStringVariable("plan_logger.format",
                              "Output format for plans",
                              "Format for plan output (json or jsonl)",
                              &plan_logger_format,
                              "jsonl",
                              PGC_SUSET,
                              0,
                              NULL, NULL, NULL);

    /* Install executor hooks */
    prev_ExecutorStart = ExecutorStart_hook;
    ExecutorStart_hook = plan_logger_ExecutorStart;
    
    prev_ExecutorEnd = ExecutorEnd_hook;
    ExecutorEnd_hook = plan_logger_ExecutorEnd;

    /* Open log file if enabled */
    if (plan_logger_enabled)
        open_log_file();
}

/*
 * Module cleanup
 */
void
_PG_fini(void)
{
    /* Restore previous hooks */
    ExecutorStart_hook = prev_ExecutorStart;
    ExecutorEnd_hook = prev_ExecutorEnd;
    
    /* Close log file */
    close_log_file();
}

/*
 * ExecutorStart hook - log plan before execution
 */
static void
plan_logger_ExecutorStart(QueryDesc *queryDesc, int eflags)
{
    /* Call previous hook if exists */
    if (prev_ExecutorStart)
        prev_ExecutorStart(queryDesc, eflags);
    else
        standard_ExecutorStart(queryDesc, eflags);

    /* Log plan if enabled and not EXPLAIN ONLY */
    if (plan_logger_enabled && !(eflags & EXEC_FLAG_EXPLAIN_ONLY))
    {
        /* Open log file if not already open */
        if (!plan_log_file)
            open_log_file();
            
        if (plan_log_file)
            log_query_plan(queryDesc);
    }
}

/*
 * ExecutorEnd hook - cleanup
 */
static void
plan_logger_ExecutorEnd(QueryDesc *queryDesc)
{
    /* Call previous hook if exists */
    if (prev_ExecutorEnd)
        prev_ExecutorEnd(queryDesc);
    else
        standard_ExecutorEnd(queryDesc);
}

/*
 * Log query plan in JSON format
 */
static void
log_query_plan(QueryDesc *queryDesc)
{
    ExplainState *es;
    StringInfoData output;
    TimestampTz now;
    const char *timestamp_str;
    HeapTuple tup;
    char *dbname = NULL;
    
    if (!plan_log_file || !queryDesc)
        return;

    /* Create EXPLAIN state for JSON output */
    es = NewExplainState();
    
    es->format = EXPLAIN_FORMAT_JSON;
    es->verbose = false;
    es->costs = true;
    es->buffers = false;
    es->timing = false;
    es->summary = false;
    es->analyze = false;

    /* Generate JSON plan */
    ExplainBeginOutput(es);
    
    /* Add query text */
    if (queryDesc->sourceText)
        ExplainPropertyText("Query Text", queryDesc->sourceText, es);
    
    /* Add the plan */
    ExplainPrintPlan(es, queryDesc);
    
    ExplainEndOutput(es);

    /* Write to log file as JSONL (one line per plan) */
    if (es->str && es->str->data)
    {
        /* Create metadata wrapper */
        initStringInfo(&output);
        
        /* Get current timestamp */
        now = GetCurrentTimestamp();
        timestamp_str = timestamptz_to_str(now);
        
        /* Get database name */
        tup = SearchSysCache1(DATABASEOID, ObjectIdGetDatum(MyDatabaseId));
        if (HeapTupleIsValid(tup))
        {
            dbname = NameStr(((Form_pg_database) GETSTRUCT(tup))->datname);
            ReleaseSysCache(tup);
        }
        
        /* Build JSONL entry with metadata */
        appendStringInfo(&output, 
                        "{\"timestamp\":\"%s\",\"pid\":%d,\"database\":\"%s\",\"plan\":%s}",
                        timestamp_str,
                        MyProcPid,
                        dbname ? dbname : "unknown",
                        es->str->data);
        
        /* Write to file (one line for JSONL format) */
        fprintf(plan_log_file, "%s\n", output.data);
        fflush(plan_log_file);
        
        pfree(output.data);
    }
    
    /* Cleanup */
    pfree(es->str->data);
    pfree(es);
}

/*
 * Open log file
 */
static void
open_log_file(void)
{
    if (plan_log_file)
        return;
        
    if (!plan_logger_path)
        plan_logger_path = "/tmp/postgres_plans.jsonl";
    
    plan_log_file = fopen(plan_logger_path, "a");
    
    if (!plan_log_file)
        ereport(WARNING,
                (errcode_for_file_access(),
                 errmsg("could not open plan log file \"%s\": %m",
                        plan_logger_path)));
}

/*
 * Close log file
 */
static void
close_log_file(void)
{
    if (plan_log_file)
    {
        fclose(plan_log_file);
        plan_log_file = NULL;
    }
}