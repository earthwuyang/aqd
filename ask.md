# Request for GPT-5-Pro: Enriching Pre-Optimization Features for Query Routing

## Context
We perform pre-planner routing between PostgreSQL (row-store) and DuckDB (columnar) using a LightGBM model. The extractor now emits 85 numeric/boolean features directly from the raw `Query` tree plus lightweight catalog lookups, so we avoid running both planners. Models trained on this feature set still show weak recall/precision, suggesting additional information could help.

**Available data at extraction time**
- Parsed `Query` tree (no `PlannerInfo` or executor statistics).
- System catalogs (`pg_class`, `pg_stats`, `pg_index`, `pg_constraint`, etc.).
- DuckDB foreign table metadata (options list, parquet hints).
- We can afford a few catalog lookups per table per query, but not full sampling.

## Current Feature Set (85)
Grouped roughly by theme in the exact order exposed to LightGBM:

1. **Structural / workload shape**: `num_tables`, `num_joins`, `query_depth`, `complexity_score`, `command_type` and clause booleans (`has_*`).
2. **Join mix**: counts for each join type (`join_type_inner`, `left`, `right`, `full`, `cross`).
3. **Predicate categories**: equality/range/like/in/exists flags plus parameter & CTE stats (`has_parameters`, `num_cte`, `max_subquery_depth`, `has_recursive_cte`, `has_lateral_join`).
4. **Legacy selectivity heuristics**: `selectivity_high/medium/low`, `cardinality_large/medium`, `index_usage_likely`, `partition_pruning_likely`, `parallel_safe`, `has_volatile_funcs`, `cost_estimate_high`.
5. **Projection metrics**: `total_projected_bytes`, `avg_projected_row_fraction`, `max_projected_row_fraction`, counts of projected columns by type, `output_row_width`, `limit_value`, `has_order_by_limit`.
6. **Scan & volume estimates**: `avg_scan_fraction`, `max_scan_fraction`, `total_rowstore_bytes_est`, `total_columnar_bytes_est`.
7. **Index leverage**: `has_covering_index`, `covering_index_score`, `order_by_index_match`, `topk_indexed`, `topk_log_limit`.
8. **Column correlation signals**: `predicate_correlation_max`, `predicate_correlation_avg`.
9. **Grouping hardness**: `group_ndv_est`, `groups_per_input_row`.
10. **Join semantics**: `fk_to_pk_joins`, `many_to_many_joins`, `star_schema_score`.
11. **Text predicate coverage**: `text_predicate_indexable`, `text_predicate_nonindexable`.
12. **DuckDB hints**: `duckdb_table_count`, `duckdb_parquet_table_count`, `duckdb_pushdown_score`.
13. **Function safety**: `volatile_function_count`, `parallel_unsafe_function_count`.
14. **Output volume**: `estimated_rows_output`, `estimated_result_bytes`.

## Ask
Despite this richer feature set, training/validation metrics remain sub-par (poor recall on DuckDB decisions, high false positives for Postgres). We want GPT-5-Pro to brainstorm additional cheap signals or transformations that could further differentiate row-store vs columnar routing *before planning*.

**Please provide:**
1. Additional feature ideas organised by theme (table stats, predicate analysis, projection characteristics, workload history, etc.).
2. Lightweight formulas using only pre-planner information (catalog stats, constant folding, metadata) – no reliance on `PlannerInfo` or executor stats.
3. Suggestions for interaction features / ratios (e.g., combining new bytes estimates with concurrency flags) that could help tree models.
4. Pointers to relevant prior art or heuristics we should study.

We especially care about signals that could improve recall on DuckDB-worthy analytical queries without misclassifying OLTP point lookups.
