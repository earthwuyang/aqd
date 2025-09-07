#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include <fstream>
#include <unordered_map>
#include <chrono>

namespace duckdb {

struct QueryFeatures {
    // Query identification
    string query_text;
    uint64_t query_hash;
    
    // Plan structure features
    idx_t num_tables = 0;
    idx_t num_joins = 0;
    idx_t num_filters = 0;
    idx_t num_projections = 0;
    idx_t num_aggregates = 0;
    idx_t num_sorts = 0;
    idx_t num_limits = 0;
    idx_t plan_depth = 0;
    idx_t total_operators = 0;
    
    // Operator type counts
    unordered_map<LogicalOperatorType, idx_t> operator_counts;
    
    // Cost estimates (to be filled by cost model)
    double estimated_cost = 0.0;
    idx_t estimated_cardinality = 0;
    
    // Join information
    vector<JoinType> join_types;
    idx_t max_join_depth = 0;
    
    // Scan information
    bool has_index_scan = false;
    bool has_seq_scan = false;
    
    // Aggregation information
    bool has_groupby = false;
    bool has_window = false;
    bool has_distinct = false;
    
    // Subquery information
    bool has_subquery = false;
    bool has_cte = false;
    
    // Convert to JSON string for logging
    string ToJSON() const;
};

class QueryFeatureLogger {
public:
    QueryFeatureLogger();
    ~QueryFeatureLogger();
    
    // Extract features from a logical plan
    QueryFeatures ExtractFeatures(const LogicalOperator &plan, const string &query_text);
    
    // Log features to file
    void LogFeatures(const QueryFeatures &features);
    
    // Enable/disable logging
    void SetEnabled(bool enabled) { logging_enabled = enabled; }
    bool IsEnabled() const { return logging_enabled; }
    
    // Set log file path
    void SetLogPath(const string &path);
    
private:
    bool logging_enabled = false;
    string log_path = "/data/duckdb_query_features.jsonl";
    unique_ptr<std::ofstream> log_file;
    
    // Helper functions for feature extraction
    void ExtractOperatorFeatures(const LogicalOperator &op, QueryFeatures &features, idx_t depth = 0);
    idx_t CountTables(const LogicalOperator &op);
    idx_t CountJoins(const LogicalOperator &op);
    idx_t CountFilters(const LogicalOperator &op);
    bool HasAggregation(const LogicalOperator &op);
    bool HasWindow(const LogicalOperator &op);
    bool HasSubquery(const LogicalOperator &op);
};

// Global instance for easy access
extern unique_ptr<QueryFeatureLogger> g_query_feature_logger;

} // namespace duckdb