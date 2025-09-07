#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <fstream>

// Simplified standalone query router for testing
// This demonstrates the concept without deep DuckDB integration

struct QueryFeatures {
    std::string query_text;
    uint64_t query_hash = 0;
    int num_tables = 0;
    int num_joins = 0;
    int num_filters = 0;
    int num_aggregates = 0;
    bool has_groupby = false;
    bool has_window = false;
    bool has_distinct = false;
    bool has_subquery = false;
    
    // Simple query analysis
    void AnalyzeQuery(const std::string& query) {
        query_text = query;
        query_hash = std::hash<std::string>{}(query);
        
        // Convert to lowercase for analysis
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        // Count keywords (simple approach)
        if (lower_query.find("from") != std::string::npos) {
            // Count approximate number of tables (very basic)
            size_t from_pos = lower_query.find("from");
            std::string from_part = lower_query.substr(from_pos);
            num_tables = 1 + std::count(from_part.begin(), from_part.end(), ',');
        }
        
        // Count JOINs
        size_t pos = 0;
        while ((pos = lower_query.find("join", pos)) != std::string::npos) {
            num_joins++;
            pos += 4;
        }
        
        // Count WHERE conditions (approximate)
        if (lower_query.find("where") != std::string::npos) {
            std::string where_part = lower_query.substr(lower_query.find("where"));
            num_filters = 1 + std::count(where_part.begin(), where_part.end(), '=') + 
                         std::count(where_part.begin(), where_part.end(), '<') +
                         std::count(where_part.begin(), where_part.end(), '>');
        }
        
        // Count aggregates
        num_aggregates += std::count(lower_query.begin(), lower_query.end(), 'c'); // COUNT approximation
        if (lower_query.find("sum(") != std::string::npos) num_aggregates++;
        if (lower_query.find("avg(") != std::string::npos) num_aggregates++;
        if (lower_query.find("min(") != std::string::npos) num_aggregates++;
        if (lower_query.find("max(") != std::string::npos) num_aggregates++;
        
        // Check for other features
        has_groupby = lower_query.find("group by") != std::string::npos;
        has_window = lower_query.find("over(") != std::string::npos;
        has_distinct = lower_query.find("distinct") != std::string::npos;
        has_subquery = (std::count(lower_query.begin(), lower_query.end(), '(') > 
                       std::count(lower_query.begin(), lower_query.end(), ')') - 1);
    }
    
    std::string ToJSON() const {
        return "{\"query_hash\":" + std::to_string(query_hash) + 
               ",\"num_tables\":" + std::to_string(num_tables) +
               ",\"num_joins\":" + std::to_string(num_joins) +
               ",\"num_filters\":" + std::to_string(num_filters) +
               ",\"num_aggregates\":" + std::to_string(num_aggregates) +
               ",\"has_groupby\":" + (has_groupby ? "true" : "false") +
               ",\"has_window\":" + (has_window ? "true" : "false") +
               ",\"has_distinct\":" + (has_distinct ? "true" : "false") +
               ",\"has_subquery\":" + (has_subquery ? "true" : "false") + "}";
    }
};

enum class RoutingDecision {
    ROUTE_TO_DUCKDB,
    ROUTE_TO_POSTGRESQL
};

class SimpleQueryRouter {
public:
    RoutingDecision MakeRoutingDecision(const QueryFeatures& features) {
        // OLTP-style queries → PostgreSQL
        // Simple queries with few joins and filters, no aggregation
        if (features.num_tables <= 2 && 
            features.num_joins <= 1 && 
            features.num_aggregates == 0 &&
            !features.has_groupby &&
            !features.has_window) {
            return RoutingDecision::ROUTE_TO_POSTGRESQL;
        }
        
        // OLAP-style queries → DuckDB
        // Complex queries with aggregations, multiple joins, analytics
        if (features.num_aggregates > 0 || 
            features.num_joins > 2 ||
            features.has_groupby ||
            features.has_window) {
            return RoutingDecision::ROUTE_TO_DUCKDB;
        }
        
        // Check for DDL/DML statements → PostgreSQL
        std::string lower_query = features.query_text;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        if (lower_query.find("create") == 0 ||
            lower_query.find("insert") == 0 ||
            lower_query.find("update") == 0 ||
            lower_query.find("delete") == 0) {
            return RoutingDecision::ROUTE_TO_POSTGRESQL;
        }
        
        // Default to DuckDB for complex SELECT queries
        return RoutingDecision::ROUTE_TO_DUCKDB;
    }
    
    std::string GetDecisionString(RoutingDecision decision) {
        return (decision == RoutingDecision::ROUTE_TO_DUCKDB) ? "DuckDB" : "PostgreSQL";
    }
    
    void LogDecision(const QueryFeatures& features, RoutingDecision decision) {
        std::ofstream log_file("/data/wuy/db/routing_decisions.jsonl", std::ios::app);
        if (log_file.is_open()) {
            log_file << "{\"decision\":\"" << GetDecisionString(decision) << "\","
                     << "\"features\":" << features.ToJSON() << "}"
                     << std::endl;
        }
    }
};

// Test queries representing different workload types
std::vector<std::string> test_queries = {
    // OLTP-style queries (should route to PostgreSQL)
    "SELECT * FROM users WHERE id = 1",
    "SELECT name, email FROM users WHERE created_at > '2024-01-01' AND status = 'active'",
    "INSERT INTO orders (user_id, product_id, quantity) VALUES (1, 2, 3)",
    "UPDATE users SET last_login = NOW() WHERE id = 123",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    
    // OLAP-style queries (should route to DuckDB)
    "SELECT u.name, COUNT(o.id) as order_count FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name ORDER BY order_count DESC",
    "SELECT product_id, SUM(quantity * price) as revenue FROM orders GROUP BY product_id HAVING revenue > 1000",
    "SELECT name, SUM(price) OVER (PARTITION BY user_id ORDER BY order_date) as running_total FROM orders JOIN users ON orders.user_id = users.id",
    "SELECT category, AVG(price), COUNT(*) FROM products GROUP BY category",
    "WITH monthly_sales AS (SELECT DATE_TRUNC('month', order_date) as month, SUM(price) as total FROM orders GROUP BY month) SELECT * FROM monthly_sales WHERE total > 10000",
    
    // Complex analytics queries (should route to DuckDB)
    "SELECT u.name, COUNT(DISTINCT o.product_id) as unique_products, AVG(o.price) as avg_order_value FROM users u JOIN orders o ON u.id = o.user_id WHERE o.order_date >= '2024-01-01' GROUP BY u.name HAVING COUNT(o.id) > 5",
    "SELECT p.category, COUNT(*) as product_count, MIN(p.price) as min_price, MAX(p.price) as max_price, AVG(p.price) as avg_price FROM products p GROUP BY p.category ORDER BY product_count DESC"
};

int main() {
    std::cout << "🚀 Phase 2: Intelligent Query Routing System Demo" << std::endl;
    std::cout << "=================================================" << std::endl << std::endl;
    
    SimpleQueryRouter router;
    int postgresql_count = 0;
    int duckdb_count = 0;
    
    std::cout << "Analyzing " << test_queries.size() << " test queries..." << std::endl << std::endl;
    
    for (size_t i = 0; i < test_queries.size(); i++) {
        QueryFeatures features;
        features.AnalyzeQuery(test_queries[i]);
        
        RoutingDecision decision = router.MakeRoutingDecision(features);
        router.LogDecision(features, decision);
        
        std::string engine = router.GetDecisionString(decision);
        
        std::cout << "Query " << (i + 1) << ": " << engine << std::endl;
        std::cout << "SQL: " << test_queries[i] << std::endl;
        std::cout << "Features: tables=" << features.num_tables 
                  << ", joins=" << features.num_joins
                  << ", filters=" << features.num_filters
                  << ", aggregates=" << features.num_aggregates;
        if (features.has_groupby) std::cout << ", GROUP BY";
        if (features.has_window) std::cout << ", WINDOW";
        std::cout << std::endl;
        std::cout << "Reasoning: ";
        
        if (decision == RoutingDecision::ROUTE_TO_POSTGRESQL) {
            postgresql_count++;
            std::cout << "OLTP-style (simple lookup/modification)";
        } else {
            duckdb_count++;
            std::cout << "OLAP-style (analytics/aggregation)";
        }
        
        std::cout << std::endl << std::endl;
    }
    
    std::cout << "📊 Routing Summary:" << std::endl;
    std::cout << "PostgreSQL (OLTP): " << postgresql_count << " queries" << std::endl;
    std::cout << "DuckDB (OLAP): " << duckdb_count << " queries" << std::endl;
    
    double postgresql_percentage = (postgresql_count * 100.0) / test_queries.size();
    double duckdb_percentage = (duckdb_count * 100.0) / test_queries.size();
    
    std::cout << std::endl << "📈 Distribution:" << std::endl;
    std::cout << "PostgreSQL: " << postgresql_percentage << "%" << std::endl;
    std::cout << "DuckDB: " << duckdb_percentage << "%" << std::endl;
    
    std::cout << std::endl << "✅ Query routing decisions logged to: /data/wuy/db/routing_decisions.jsonl" << std::endl;
    std::cout << "✅ Ready for training data collection and ML model training!" << std::endl;
    
    return 0;
}