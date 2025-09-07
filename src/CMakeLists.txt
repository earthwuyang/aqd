# Query Router Extension for DuckDB
# CMake configuration for LightGBM-based query routing

cmake_minimum_required(VERSION 3.12)

# Add query router sources
set(QUERY_ROUTER_SOURCES
    main/query_router/lightgbm_router.cpp
    main/query_router/query_router.cpp
    main/query_router/postgresql_executor.cpp
    optimizer/query_feature_logger.cpp
)

# Add query router headers
set(QUERY_ROUTER_HEADERS
    include/duckdb/main/lightgbm_router.hpp
    include/duckdb/main/query_router.hpp
)

# Optional LightGBM integration
option(ENABLE_LIGHTGBM "Enable LightGBM ML-based query routing" OFF)

if(ENABLE_LIGHTGBM)
    # Find LightGBM
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIGHTGBM REQUIRED lightgbm)
    
    if(LIGHTGBM_FOUND)
        message(STATUS "LightGBM found: ${LIGHTGBM_VERSION}")
        
        # Add compile definitions
        add_compile_definitions(ENABLE_LIGHTGBM)
        
        # Add LightGBM include directories
        include_directories(${LIGHTGBM_INCLUDE_DIRS})
        
        # Link LightGBM library
        link_directories(${LIGHTGBM_LIBRARY_DIRS})
        target_link_libraries(duckdb_static ${LIGHTGBM_LIBRARIES})
        
        message(STATUS "LightGBM query routing enabled")
    else()
        message(WARNING "LightGBM requested but not found. Disabling ML routing.")
        set(ENABLE_LIGHTGBM OFF)
    endif()
else()
    message(STATUS "LightGBM query routing disabled. Use -DENABLE_LIGHTGBM=ON to enable.")
endif()

# Add query router sources to DuckDB
if(TARGET duckdb_static)
    target_sources(duckdb_static PRIVATE ${QUERY_ROUTER_SOURCES})
    target_include_directories(duckdb_static PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# Installation rules for headers
install(FILES ${QUERY_ROUTER_HEADERS} 
        DESTINATION include/duckdb/main)

# Configuration summary
message(STATUS "Query Router Configuration:")
message(STATUS "  LightGBM support: ${ENABLE_LIGHTGBM}")
if(ENABLE_LIGHTGBM)
    message(STATUS "  LightGBM version: ${LIGHTGBM_VERSION}")
    message(STATUS "  LightGBM libraries: ${LIGHTGBM_LIBRARIES}")
endif()