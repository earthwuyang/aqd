#include "duckdb/main/query_router.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

// Include libpq for PostgreSQL connectivity
#ifndef DISABLE_POSTGRESQL_EXECUTOR
#ifdef __cplusplus
extern "C" {
#endif
#include <libpq-fe.h>
#ifdef __cplusplus
}
#endif
#endif

namespace duckdb {

PostgreSQLExecutor::PostgreSQLExecutor(const string &connection_string) 
    : connection_string(connection_string), conn(nullptr) {
#ifndef DISABLE_POSTGRESQL_EXECUTOR
    // Initial connection test
    ConnectToDatabase();
#endif
}

PostgreSQLExecutor::~PostgreSQLExecutor() {
#ifndef DISABLE_POSTGRESQL_EXECUTOR
    DisconnectFromDatabase();
#endif
}

bool PostgreSQLExecutor::ConnectToDatabase() {
#ifdef DISABLE_POSTGRESQL_EXECUTOR
    return false;
#else
    if (conn) {
        DisconnectFromDatabase();
    }
    
    conn = PQconnectdb(connection_string.c_str());
    
    if (PQstatus(static_cast<PGconn*>(conn)) != CONNECTION_OK) {
        std::cerr << "Connection to PostgreSQL failed: " 
                  << PQerrorMessage(static_cast<PGconn*>(conn)) << std::endl;
        PQfinish(static_cast<PGconn*>(conn));
        conn = nullptr;
        return false;
    }
    
    return true;
#endif
}

void PostgreSQLExecutor::DisconnectFromDatabase() {
#ifndef DISABLE_POSTGRESQL_EXECUTOR
    if (conn) {
        PQfinish(static_cast<PGconn*>(conn));
        conn = nullptr;
    }
#endif
}

bool PostgreSQLExecutor::TestConnection() {
#ifdef DISABLE_POSTGRESQL_EXECUTOR
    return false;
#else
    if (!conn) {
        return ConnectToDatabase();
    }
    
    // Test connection with a simple query
    PGresult *res = PQexec(static_cast<PGconn*>(conn), "SELECT 1");
    bool success = (PQresultStatus(res) == PGRES_TUPLES_OK);
    PQclear(res);
    
    if (!success) {
        // Try to reconnect
        return ConnectToDatabase();
    }
    
    return true;
#endif
}

bool PostgreSQLExecutor::ExecuteQuery(const string &query, string &result, double &execution_time_ms) {
#ifdef DISABLE_POSTGRESQL_EXECUTOR
    result = "PostgreSQL executor disabled at compile time";
    execution_time_ms = 0.0;
    return false;
#else
    if (!conn || !TestConnection()) {
        result = "PostgreSQL connection not available";
        execution_time_ms = 0.0;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PGresult *res = PQexec(static_cast<PGconn*>(conn), query.c_str());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    execution_time_ms = duration.count() / 1000.0;
    
    ExecStatusType status = PQresultStatus(res);
    bool success = false;
    
    switch (status) {
        case PGRES_COMMAND_OK:
            // For INSERT, UPDATE, DELETE, CREATE, etc.
            result = "Command completed successfully";
            success = true;
            break;
            
        case PGRES_TUPLES_OK: {
            // For SELECT queries
            std::stringstream ss;
            int rows = PQntuples(res);
            int cols = PQnfields(res);
            
            // Add column headers
            for (int col = 0; col < cols; col++) {
                if (col > 0) ss << "\t";
                ss << PQfname(res, col);
            }
            if (cols > 0) ss << "\n";
            
            // Add data rows
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    if (col > 0) ss << "\t";
                    const char *value = PQgetvalue(res, row, col);
                    ss << (PQgetisnull(res, row, col) ? "NULL" : value);
                }
                ss << "\n";
            }
            
            result = ss.str();
            success = true;
            break;
        }
        
        default:
            result = "PostgreSQL Error: ";
            result += PQerrorMessage(static_cast<PGconn*>(conn));
            success = false;
            break;
    }
    
    PQclear(res);
    return success;
#endif
}

bool PostgreSQLExecutor::IsAvailable() {
#ifdef DISABLE_POSTGRESQL_EXECUTOR
    return false;
#else
    return conn && (PQstatus(static_cast<PGconn*>(conn)) == CONNECTION_OK);
#endif
}

string PostgreSQLExecutor::GetEngineInfo() {
#ifdef DISABLE_POSTGRESQL_EXECUTOR
    return "PostgreSQL (disabled)";
#else
    if (!IsAvailable()) {
        return "PostgreSQL (disconnected)";
    }
    
    int version = PQserverVersion(static_cast<PGconn*>(conn));
    int major = version / 10000;
    int minor = (version / 100) % 100;
    int patch = version % 100;
    
    std::stringstream ss;
    ss << "PostgreSQL " << major << "." << minor;
    if (patch > 0) {
        ss << "." << patch;
    }
    
    return ss.str();
#endif
}

} // namespace duckdb