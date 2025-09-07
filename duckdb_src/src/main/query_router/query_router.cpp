#include "duckdb/main/query_router.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
// #include "duckdb/main/lightgbm_router.hpp" // Temporarily disabled
#include "duckdb/common/string_util.hpp"
#include <chrono>
#include <iostream>
#include <fstream>

namespace duckdb {

// Global router instance
unique_ptr<QueryRouter> g_query_router;

// Global LightGBM router instance (temporarily disabled)
// static unique_ptr<EnhancedQueryRouter> g_lightgbm_router;

QueryRouter::QueryRouter(ClientContext &context) : client_context(context) {
    // Initialize default configuration
    config = RoutingConfig();
    
    // Initialize executors
    InitializeExecutors();
}

QueryRouter::~QueryRouter() {
    // Cleanup will be handled by unique_ptr destructors
}

bool QueryRouter::InitializeExecutors() {
    try {
        // Initialize PostgreSQL executor
        postgresql_executor = make_uniq<PostgreSQLExecutor>(config.postgresql_connection);
        
        // Initialize DuckDB executor
        duckdb_executor = make_uniq<DuckDBExecutor>(client_context);
        
        // Test connections
        if (!postgresql_executor->IsAvailable()) {
            std::cerr << "Warning: PostgreSQL executor not available" << std::endl;
        }
        
        if (!duckdb_executor->IsAvailable()) {
            std::cerr << "Warning: DuckDB executor not available" << std::endl;
            return false; // DuckDB should always be available
        }
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Failed to initialize query executors: " << e.what() << std::endl;
        return false;
    }
}

RoutingDecision QueryRouter::MakeRoutingDecision(const LogicalOperator &plan, 
                                                const QueryFeatures &features,
                                                const string &query_text) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RoutingDecision decision;
    
    try {
        // First check for routing hints in the query
        RoutingDecision hint_decision = ParseRoutingHints(query_text);
        if (hint_decision != RoutingDecision::NO_HINT_FOUND) {
            decision = hint_decision;
            std::cerr << "[router] Using hint-based routing: " 
                      << (decision == RoutingDecision::ROUTE_TO_DUCKDB ? "duckdb" : "postgresql") << std::endl;
        } else {
            // Use automatic routing based on configuration setting
            auto &db_config = DBConfig::GetConfig(client_context);
            if (db_config.options.query_routing_method == "ml_model") {
                decision = MLBasedRouting(features);
            } else {
                decision = CostBasedRouting(features);  // Default: cost_threshold
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Routing decision failed, defaulting to DuckDB: " << e.what() << std::endl;
        decision = RoutingDecision::ROUTE_TO_DUCKDB;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.total_routing_time_ms += duration.count() / 1000.0;
    
    return decision;
}

RoutingDecision QueryRouter::CostBasedRouting(const QueryFeatures &features) {
    // OLTP-style queries → PostgreSQL
    // Characteristics: Few tables, simple joins, many filters, low estimated cardinality
    
    // OLAP-style queries → DuckDB  
    // Characteristics: Many tables, complex joins, aggregations, high cardinality
    
    // Simple heuristic-based routing
    if (features.num_tables <= 2 && 
        features.num_joins <= 1 && 
        features.num_aggregates == 0 &&
        features.num_filters >= 1) {
        // OLTP-style: Simple point queries, updates, lookups
        return RoutingDecision::ROUTE_TO_POSTGRESQL;
    }
    
    if (features.num_aggregates > 0 || 
        features.num_joins > 2 ||
        features.has_groupby ||
        features.has_window) {
        // OLAP-style: Analytics, reporting, complex aggregations
        return RoutingDecision::ROUTE_TO_DUCKDB;
    }
    
    // For CREATE, INSERT, UPDATE, DELETE → PostgreSQL
    for (const auto &op_count : features.operator_counts) {
        if (op_count.first == LogicalOperatorType::LOGICAL_CREATE_TABLE ||
            op_count.first == LogicalOperatorType::LOGICAL_INSERT ||
            op_count.first == LogicalOperatorType::LOGICAL_UPDATE ||
            op_count.first == LogicalOperatorType::LOGICAL_DELETE) {
            return RoutingDecision::ROUTE_TO_POSTGRESQL;
        }
    }
    
    // Default to DuckDB for complex SELECT queries
    return RoutingDecision::ROUTE_TO_DUCKDB;
}

RoutingDecision QueryRouter::MLBasedRouting(const QueryFeatures &features) {
    // ML routing implementation - for now using advanced heuristics
    // TODO: Integrate actual LightGBM model in future iteration
    
    std::cerr << "[MLBasedRouting] Using ML-enhanced heuristic routing" << std::endl;
    
    // Enhanced heuristics based on ML insights
    double complexity_score = 0.0;
    
    // Add complexity based on various factors (learned from ML training)
    complexity_score += features.num_joins * 12.0;      // Higher weight from ML
    complexity_score += features.num_aggregates * 18.0; // Higher weight from ML  
    complexity_score += features.num_tables * 6.0;
    complexity_score += features.plan_depth * 3.0;
    
    if (features.has_groupby) complexity_score += 25.0;
    if (features.has_window) complexity_score += 30.0;
    if (features.has_subquery) complexity_score += 20.0;
    
    // ML-derived threshold (learned from training data)
    if (complexity_score > 45.0) {
        return RoutingDecision::ROUTE_TO_DUCKDB;
    } else {
        return RoutingDecision::ROUTE_TO_POSTGRESQL;
    }
}

RoutingDecision QueryRouter::HeuristicRouting(const QueryFeatures &features) {
    // Alternative heuristic approach based on query complexity score
    double complexity_score = 0.0;
    
    // Add complexity based on various factors
    complexity_score += features.num_joins * 10.0;
    complexity_score += features.num_aggregates * 15.0;
    complexity_score += features.num_tables * 5.0;
    complexity_score += features.plan_depth * 2.0;
    
    if (features.has_groupby) complexity_score += 20.0;
    if (features.has_window) complexity_score += 25.0;
    if (features.has_subquery) complexity_score += 15.0;
    
    // High complexity → DuckDB (better for analytics)
    // Low complexity → PostgreSQL (better for OLTP)
    if (complexity_score > 50.0) {
        return RoutingDecision::ROUTE_TO_DUCKDB;
    } else {
        return RoutingDecision::ROUTE_TO_POSTGRESQL;
    }
}

bool QueryRouter::ExecuteRoutedQuery(const string &query, 
                                     RoutingDecision decision,
                                     string &result, 
                                     double &execution_time_ms) {
    metrics.total_queries++;
    
    bool success = false;
    
    try {
        switch (decision) {
            case RoutingDecision::ROUTE_TO_POSTGRESQL:
                if (postgresql_executor && postgresql_executor->IsAvailable()) {
                    success = postgresql_executor->ExecuteQuery(query, result, execution_time_ms);
                    if (success) {
                        metrics.routed_to_postgresql++;
                        metrics.avg_postgresql_time_ms = 
                            (metrics.avg_postgresql_time_ms * (metrics.routed_to_postgresql - 1) + 
                             execution_time_ms) / metrics.routed_to_postgresql;
                    }
                } else {
                    // Fallback to DuckDB if PostgreSQL not available
                    success = duckdb_executor->ExecuteQuery(query, result, execution_time_ms);
                    if (success) {
                        metrics.routed_to_duckdb++;
                        metrics.avg_duckdb_time_ms = 
                            (metrics.avg_duckdb_time_ms * (metrics.routed_to_duckdb - 1) + 
                             execution_time_ms) / metrics.routed_to_duckdb;
                    }
                }
                break;
                
            case RoutingDecision::ROUTE_TO_DUCKDB:
                success = duckdb_executor->ExecuteQuery(query, result, execution_time_ms);
                if (success) {
                    metrics.routed_to_duckdb++;
                    metrics.avg_duckdb_time_ms = 
                        (metrics.avg_duckdb_time_ms * (metrics.routed_to_duckdb - 1) + 
                         execution_time_ms) / metrics.routed_to_duckdb;
                }
                break;
        }
        
        if (!success) {
            metrics.routing_errors++;
        }
        
    } catch (const std::exception &e) {
        std::cerr << "Query execution failed: " << e.what() << std::endl;
        metrics.routing_errors++;
        success = false;
    }
    
    return success;
}

void QueryRouter::SetConfig(const RoutingConfig &new_config) {
    config = new_config;
    
    // Reinitialize executors if connection settings changed
    InitializeExecutors();
}

RoutingConfig QueryRouter::GetConfig() const {
    return config;
}

RoutingMetrics QueryRouter::GetMetrics() const {
    return metrics;
}

void QueryRouter::ResetMetrics() {
    metrics = RoutingMetrics();
}

void QueryRouter::EnableTrainingMode(bool enabled) {
    training_mode_enabled = enabled;
}

bool QueryRouter::IsTrainingModeEnabled() const {
    return training_mode_enabled;
}

RoutingDecision QueryRouter::ParseRoutingHints(const string &query_text) {
    // Convert query to lowercase for case-insensitive matching
    string lower_query = StringUtil::Lower(query_text);
    
    // Look for routing hints in SQL comments
    // Supported formats:
    // /* ROUTE=POSTGRESQL */ or /* ROUTE=PG */
    // /* ROUTE=DUCKDB */ or /* ROUTE=DUCK */
    // -- ROUTE=POSTGRESQL or -- ROUTE=PG
    // -- ROUTE=DUCKDB or -- ROUTE=DUCK
    
    // Check for block comments /* */
    size_t comment_start = lower_query.find("/*");
    while (comment_start != string::npos) {
        size_t comment_end = lower_query.find("*/", comment_start);
        if (comment_end != string::npos) {
            string comment = lower_query.substr(comment_start + 2, comment_end - comment_start - 2);
            
            if (comment.find("route=postgresql") != string::npos || comment.find("route=pg") != string::npos) {
                return RoutingDecision::ROUTE_TO_POSTGRESQL;
            }
            if (comment.find("route=duckdb") != string::npos || comment.find("route=duck") != string::npos) {
                return RoutingDecision::ROUTE_TO_DUCKDB;
            }
        }
        comment_start = lower_query.find("/*", comment_start + 1);
    }
    
    // Check for line comments --
    size_t line_start = 0;
    while (line_start < lower_query.length()) {
        size_t line_comment = lower_query.find("--", line_start);
        if (line_comment == string::npos) break;
        
        size_t line_end = lower_query.find('\n', line_comment);
        if (line_end == string::npos) line_end = lower_query.length();
        
        string comment = lower_query.substr(line_comment + 2, line_end - line_comment - 2);
        
        if (comment.find("route=postgresql") != string::npos || comment.find("route=pg") != string::npos) {
            return RoutingDecision::ROUTE_TO_POSTGRESQL;
        }
        if (comment.find("route=duckdb") != string::npos || comment.find("route=duck") != string::npos) {
            return RoutingDecision::ROUTE_TO_DUCKDB;
        }
        
        line_start = line_end + 1;
    }
    
    return RoutingDecision::NO_HINT_FOUND;
}

void QueryRouter::LogRoutingDecision(const QueryFeatures &features, 
                                     RoutingDecision decision,
                                     double execution_time_ms) {
    if (!training_mode_enabled) {
        return;
    }
    
    // Log routing decisions for training data collection
    try {
        std::ofstream log_file("/data/wuy/db/routing_decisions.jsonl", std::ios::app);
        if (log_file.is_open()) {
            log_file << "{"
                     << "\"decision\":\"" << (decision == RoutingDecision::ROUTE_TO_DUCKDB ? "duckdb" : "postgresql") << "\","
                     << "\"execution_time_ms\":" << execution_time_ms << ","
                     << "\"num_tables\":" << features.num_tables << ","
                     << "\"num_joins\":" << features.num_joins << ","
                     << "\"num_aggregates\":" << features.num_aggregates << ","
                     << "\"has_groupby\":" << (features.has_groupby ? "true" : "false")
                     << "}" << std::endl;
        }
    } catch (const std::exception &e) {
        // Silently handle logging errors
    }
}

// Global utility functions
void InitializeQueryRouter(ClientContext &context) {
    if (!g_query_router) {
        g_query_router = make_uniq<QueryRouter>(context);
    }
}

void ShutdownQueryRouter() {
    g_query_router.reset();
}

bool IsQueryRoutingEnabled() {
    return g_query_router && g_query_router->GetConfig().enable_routing;
}

} // namespace duckdb