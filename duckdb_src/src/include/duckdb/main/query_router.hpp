#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/logical_operator_type.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/optimizer/query_feature_logger.hpp"
#include "duckdb/main/client_context.hpp"
#include <memory>
#include <string>

namespace duckdb {

enum class RoutingDecision {
    ROUTE_TO_DUCKDB,
    ROUTE_TO_POSTGRESQL,
    NO_HINT_FOUND  // Used internally for hint parsing
};

struct RoutingConfig {
    bool enable_routing = true;
    bool use_ml_routing = false;
    double join_threshold = 1000.0;     // Cost threshold for joins
    double scan_threshold = 5000.0;     // Cost threshold for scans
    double aggregation_threshold = 2000.0;  // Cost threshold for aggregations
    string postgresql_connection = "host=localhost port=5433 dbname=postgres user=wuy";
    string ml_model_path = "";
    
    // Performance thresholds (milliseconds)
    double oltp_time_threshold = 100.0;    // Queries faster than this → PostgreSQL
    double olap_time_threshold = 1000.0;   // Queries slower than this → DuckDB
};

class DatabaseExecutor {
public:
    virtual ~DatabaseExecutor() = default;
    virtual bool ExecuteQuery(const string &query, string &result, double &execution_time_ms) = 0;
    virtual bool IsAvailable() = 0;
    virtual string GetEngineInfo() = 0;
};

class PostgreSQLExecutor : public DatabaseExecutor {
public:
    explicit PostgreSQLExecutor(const string &connection_string);
    ~PostgreSQLExecutor() override;
    
    bool ExecuteQuery(const string &query, string &result, double &execution_time_ms) override;
    bool IsAvailable() override;
    string GetEngineInfo() override;
    bool TestConnection();

private:
    string connection_string;
    void* conn; // PGconn* (forward declaration to avoid libpq dependency in header)
    bool ConnectToDatabase();
    void DisconnectFromDatabase();
};

class DuckDBExecutor : public DatabaseExecutor {
public:
    explicit DuckDBExecutor(ClientContext &context);
    ~DuckDBExecutor() override = default;
    
    bool ExecuteQuery(const string &query, string &result, double &execution_time_ms) override;
    bool IsAvailable() override;
    string GetEngineInfo() override;

private:
    ClientContext &client_context;
};

struct RoutingMetrics {
    uint64_t total_queries = 0;
    uint64_t routed_to_postgresql = 0;
    uint64_t routed_to_duckdb = 0;
    uint64_t routing_errors = 0;
    double total_routing_time_ms = 0.0;
    double avg_postgresql_time_ms = 0.0;
    double avg_duckdb_time_ms = 0.0;
};

class QueryRouter {
public:
    explicit QueryRouter(ClientContext &context);
    ~QueryRouter();
    
    // Main routing decision method
    RoutingDecision MakeRoutingDecision(const LogicalOperator &plan, 
                                       const QueryFeatures &features,
                                       const string &query_text);
    
    // Parse hints from query text to override routing decision
    RoutingDecision ParseRoutingHints(const string &query_text);
    
    // Execute query on chosen engine
    bool ExecuteRoutedQuery(const string &query, 
                           RoutingDecision decision,
                           string &result, 
                           double &execution_time_ms);
    
    // Configuration methods
    void SetConfig(const RoutingConfig &config);
    RoutingConfig GetConfig() const;
    
    // Metrics and monitoring
    RoutingMetrics GetMetrics() const;
    void ResetMetrics();
    
    // Training data collection
    void EnableTrainingMode(bool enabled);
    bool IsTrainingModeEnabled() const;
    
private:
    ClientContext &client_context;
    RoutingConfig config;
    RoutingMetrics metrics;
    bool training_mode_enabled = false;
    
    unique_ptr<PostgreSQLExecutor> postgresql_executor;
    unique_ptr<DuckDBExecutor> duckdb_executor;
    
    // Routing strategies
    RoutingDecision CostBasedRouting(const QueryFeatures &features);
    RoutingDecision MLBasedRouting(const QueryFeatures &features);
    RoutingDecision HeuristicRouting(const QueryFeatures &features);
    
    // Helper methods
    bool InitializeExecutors();
    void LogRoutingDecision(const QueryFeatures &features, 
                           RoutingDecision decision,
                           double execution_time_ms);
};

// Global router instance
extern unique_ptr<QueryRouter> g_query_router;

// Utility functions for integration
void InitializeQueryRouter(ClientContext &context);
void ShutdownQueryRouter();
bool IsQueryRoutingEnabled();

} // namespace duckdb