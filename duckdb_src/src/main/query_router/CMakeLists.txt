add_library(duckdb_query_router OBJECT
    query_router.cpp
    postgresql_executor.cpp
    duckdb_executor.cpp
)

# Use custom PostgreSQL installation
set(PostgreSQL_ROOT "${CMAKE_SOURCE_DIR}/../postgresql")
find_path(PostgreSQL_INCLUDE_DIR 
    NAMES libpq-fe.h 
    PATHS ${PostgreSQL_ROOT}/include
    NO_DEFAULT_PATH
)

find_library(PostgreSQL_LIBRARY 
    NAMES pq 
    PATHS ${PostgreSQL_ROOT}/lib
    NO_DEFAULT_PATH
)

# Disable PostgreSQL executor completely for initial testing
target_compile_definitions(duckdb_query_router PRIVATE DISABLE_POSTGRESQL_EXECUTOR)
message(STATUS "PostgreSQL executor disabled - focusing on core routing functionality")

# if(PostgreSQL_INCLUDE_DIR AND PostgreSQL_LIBRARY)
#     target_include_directories(duckdb_query_router PRIVATE ${PostgreSQL_INCLUDE_DIR})
#     target_link_libraries(duckdb_query_router PRIVATE ${PostgreSQL_LIBRARY})
#     message(STATUS "Found custom PostgreSQL: ${PostgreSQL_LIBRARY}")
# else()
#     message(WARNING "PostgreSQL not found, query router will be disabled")
#     target_compile_definitions(duckdb_query_router PRIVATE DISABLE_POSTGRESQL_EXECUTOR)
# endif()

set(ALL_OBJECT_FILES ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_query_router> PARENT_SCOPE)