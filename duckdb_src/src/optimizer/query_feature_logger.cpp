#include "duckdb/optimizer/query_feature_logger.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include <sstream>
#include <iomanip>

namespace duckdb {

// Global instance
unique_ptr<QueryFeatureLogger> g_query_feature_logger = make_uniq<QueryFeatureLogger>();

QueryFeatureLogger::QueryFeatureLogger() {
    // Check environment variable to enable logging
    const char* enable_logging = std::getenv("DUCKDB_ENABLE_QUERY_LOGGING");
    if (enable_logging && std::string(enable_logging) == "1") {
        SetEnabled(true);
        const char* custom_path = std::getenv("DUCKDB_QUERY_LOG_PATH");
        if (custom_path) {
            SetLogPath(custom_path);
        }
    }
}

QueryFeatureLogger::~QueryFeatureLogger() {
    if (log_file && log_file->is_open()) {
        log_file->close();
    }
}

void QueryFeatureLogger::SetLogPath(const string &path) {
    log_path = path;
    if (log_file && log_file->is_open()) {
        log_file->close();
    }
    log_file = make_uniq<std::ofstream>(log_path, std::ios::app);
}

QueryFeatures QueryFeatureLogger::ExtractFeatures(const LogicalOperator &plan, const string &query_text) {
    QueryFeatures features;
    features.query_text = query_text;
    features.query_hash = std::hash<string>{}(query_text);
    
    // Extract features from the plan tree
    ExtractOperatorFeatures(plan, features);
    
    return features;
}

void QueryFeatureLogger::ExtractOperatorFeatures(const LogicalOperator &op, QueryFeatures &features, idx_t depth) {
    // Update plan depth
    if (depth > features.plan_depth) {
        features.plan_depth = depth;
    }
    
    // Count total operators
    features.total_operators++;
    
    // Count operator types
    features.operator_counts[op.type]++;
    
    // Extract specific operator features
    switch (op.type) {
        case LogicalOperatorType::LOGICAL_GET:
            features.num_tables++;
            features.has_seq_scan = true;
            break;
            
        case LogicalOperatorType::LOGICAL_JOIN:
        case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
        case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
            features.num_joins++;
            auto &join = op.Cast<LogicalJoin>();
            features.join_types.push_back(join.join_type);
            if (depth > features.max_join_depth) {
                features.max_join_depth = depth;
            }
            break;
        }
        
        case LogicalOperatorType::LOGICAL_FILTER:
            features.num_filters++;
            break;
            
        case LogicalOperatorType::LOGICAL_PROJECTION:
            features.num_projections++;
            break;
            
        case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
            features.num_aggregates++;
            features.has_groupby = true;
            break;
            
        case LogicalOperatorType::LOGICAL_ORDER_BY:
            features.num_sorts++;
            break;
            
        case LogicalOperatorType::LOGICAL_LIMIT:
        case LogicalOperatorType::LOGICAL_TOP_N:
            features.num_limits++;
            break;
            
        case LogicalOperatorType::LOGICAL_WINDOW:
            features.has_window = true;
            break;
            
        case LogicalOperatorType::LOGICAL_DISTINCT:
            features.has_distinct = true;
            break;
            
        case LogicalOperatorType::LOGICAL_CTE_REF:
            features.has_cte = true;
            break;
            
        case LogicalOperatorType::LOGICAL_DELIM_JOIN:
        case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
            features.has_subquery = true;
            break;
            
        default:
            break;
    }
    
    // Get estimated cardinality if available
    if (op.estimated_cardinality > features.estimated_cardinality) {
        features.estimated_cardinality = op.estimated_cardinality;
    }
    
    // Recursively process children
    for (auto &child : op.children) {
        ExtractOperatorFeatures(*child, features, depth + 1);
    }
}

string QueryFeatures::ToJSON() const {
    std::stringstream ss;
    ss << "{";
    
    // Basic features
    ss << "\"query_hash\":" << query_hash << ",";
    ss << "\"num_tables\":" << num_tables << ",";
    ss << "\"num_joins\":" << num_joins << ",";
    ss << "\"num_filters\":" << num_filters << ",";
    ss << "\"num_projections\":" << num_projections << ",";
    ss << "\"num_aggregates\":" << num_aggregates << ",";
    ss << "\"num_sorts\":" << num_sorts << ",";
    ss << "\"num_limits\":" << num_limits << ",";
    ss << "\"plan_depth\":" << plan_depth << ",";
    ss << "\"total_operators\":" << total_operators << ",";
    
    // Boolean features
    ss << "\"has_index_scan\":" << (has_index_scan ? "true" : "false") << ",";
    ss << "\"has_seq_scan\":" << (has_seq_scan ? "true" : "false") << ",";
    ss << "\"has_groupby\":" << (has_groupby ? "true" : "false") << ",";
    ss << "\"has_window\":" << (has_window ? "true" : "false") << ",";
    ss << "\"has_distinct\":" << (has_distinct ? "true" : "false") << ",";
    ss << "\"has_subquery\":" << (has_subquery ? "true" : "false") << ",";
    ss << "\"has_cte\":" << (has_cte ? "true" : "false") << ",";
    
    // Join types
    ss << "\"join_types\":[";
    for (idx_t i = 0; i < join_types.size(); i++) {
        if (i > 0) ss << ",";
        ss << "\"" << JoinTypeToString(join_types[i]) << "\"";
    }
    ss << "],";
    
    // Operator counts
    ss << "\"operator_counts\":{";
    bool first = true;
    for (const auto &pair : operator_counts) {
        if (!first) ss << ",";
        ss << "\"" << LogicalOperatorToString(pair.first) << "\":" << pair.second;
        first = false;
    }
    ss << "},";
    
    // Cost estimates
    ss << "\"estimated_cost\":" << estimated_cost << ",";
    ss << "\"estimated_cardinality\":" << estimated_cardinality << ",";
    
    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    ss << "\"timestamp\":" << timestamp << ",";
    
    // Query text (escaped)
    ss << "\"query_text\":\"";
    for (char c : query_text) {
        if (c == '"') ss << "\\\"";
        else if (c == '\\') ss << "\\\\";
        else if (c == '\n') ss << "\\n";
        else if (c == '\r') ss << "\\r";
        else if (c == '\t') ss << "\\t";
        else ss << c;
    }
    ss << "\"";
    
    ss << "}";
    return ss.str();
}

void QueryFeatureLogger::LogFeatures(const QueryFeatures &features) {
    if (!logging_enabled) {
        return;
    }
    
    if (!log_file || !log_file->is_open()) {
        log_file = make_uniq<std::ofstream>(log_path, std::ios::app);
    }
    
    if (log_file && log_file->is_open()) {
        *log_file << features.ToJSON() << std::endl;
        log_file->flush();
    }
}

} // namespace duckdb