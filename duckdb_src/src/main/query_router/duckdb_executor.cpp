#include "duckdb/main/query_router.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/common/string_util.hpp"
#include <chrono>
#include <sstream>

namespace duckdb {

DuckDBExecutor::DuckDBExecutor(ClientContext &context) : client_context(context) {
    // DuckDB executor uses the existing client context
}

bool DuckDBExecutor::ExecuteQuery(const string &query, string &result, double &execution_time_ms) {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute query through the client context
        auto query_result = client_context.Query(query, false);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        execution_time_ms = duration.count() / 1000.0;
        
        if (query_result->HasError()) {
            result = "DuckDB Error: " + query_result->GetError();
            return false;
        }
        
        // Format result as string
        std::stringstream ss;
        
        if (query_result->type == QueryResultType::STREAM_RESULT) {
            auto &stream_result = query_result->Cast<StreamQueryResult>();
            
            // Get column names
            auto &names = stream_result.names;
            for (size_t i = 0; i < names.size(); i++) {
                if (i > 0) ss << "\t";
                ss << names[i];
            }
            if (!names.empty()) ss << "\n";
            
            // Get data chunks
            unique_ptr<DataChunk> chunk;
            while ((chunk = stream_result.Fetch())) {
                if (chunk->size() == 0) {
                    break;
                }
                
                for (idx_t row = 0; row < chunk->size(); row++) {
                    for (idx_t col = 0; col < chunk->ColumnCount(); col++) {
                        if (col > 0) ss << "\t";
                        ss << chunk->GetValue(col, row).ToString();
                    }
                    ss << "\n";
                }
            }
        } else {
            // Handle materialized result
            auto &materialized_result = query_result->Cast<MaterializedQueryResult>();
            
            // Get column names
            auto &names = materialized_result.names;
            for (size_t i = 0; i < names.size(); i++) {
                if (i > 0) ss << "\t";
                ss << names[i];
            }
            if (!names.empty()) ss << "\n";
            
            // Get data rows
            for (idx_t row = 0; row < materialized_result.RowCount(); row++) {
                for (idx_t col = 0; col < materialized_result.ColumnCount(); col++) {
                    if (col > 0) ss << "\t";
                    ss << materialized_result.GetValue(col, row).ToString();
                }
                ss << "\n";
            }
        }
        
        result = ss.str();
        return true;
        
    } catch (const std::exception &e) {
        result = "DuckDB Exception: ";
        result += e.what();
        execution_time_ms = 0.0;
        return false;
    }
}

bool DuckDBExecutor::IsAvailable() {
    // DuckDB executor is always available if we have a valid client context
    return true;
}

string DuckDBExecutor::GetEngineInfo() {
    try {
        auto version_result = client_context.Query("SELECT version()", false);
        if (version_result->HasError()) {
            return "DuckDB (version unknown)";
        }
        
        if (version_result->type == QueryResultType::STREAM_RESULT) {
            auto &stream_result = version_result->Cast<StreamQueryResult>();
            auto chunk = stream_result.Fetch();
            if (chunk && chunk->size() > 0) {
                return chunk->GetValue(0, 0).ToString();
            }
        }
        
        return "DuckDB (custom build with query routing)";
        
    } catch (const std::exception &e) {
        return "DuckDB (version query failed)";
    }
}

} // namespace duckdb