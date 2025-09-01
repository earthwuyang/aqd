def combine_traces(runs):
    start_plan = runs[0]
    for p in runs[1:]: # wuy: because runs[i] are in the same database, so their database_stats (including column_stats and table_stats) and run_kwargs are the same, so only keep the first one as below lines do
        start_plan.query_list += p.query_list
        start_plan.total_time_secs += p.total_time_secs

    return start_plan
