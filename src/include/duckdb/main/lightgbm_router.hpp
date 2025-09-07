#pragma once

/*
 * LightGBM-based Query Router for DuckDB Kernel Integration
 * 
 * This module provides ML-based query routing using an embedded LightGBM model
 * trained on 200K+ benchmark queries with dual-engine execution data.
 */

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/query_router.hpp"

#ifdef ENABLE_LIGHTGBM
#include <LightGBM/c_api.h>
#endif

namespace duckdb {

// Forward declarations
struct QueryFeatures;
class LogicalOperator;

//! LightGBM-based query routing decision
class LightGBMRouter {
public:
    LightGBMRouter();
    ~LightGBMRouter();
    
    //! Initialize the LightGBM model from file
    bool Initialize(const string &model_path);
    
    //! Make routing decision using ML model
    RoutingDecision PredictRouting(const QueryFeatures &features);
    
    //! Check if model is loaded and ready
    bool IsInitialized() const { return model_loaded; }
    
    //! Get model information
    string GetModelInfo() const;
    
    //! Extract features suitable for LightGBM prediction
    vector<double> ExtractMLFeatures(const QueryFeatures &features);

private:
    //! Scale features using pre-trained scaler parameters
    vector<double> ScaleFeatures(const vector<double> &raw_features);
    
    //! Load feature scaling parameters
    bool LoadScalerParameters(const string &scaler_path);

private:
#ifdef ENABLE_LIGHTGBM
    BoosterHandle model_handle;
#else
    void *model_handle;  // Placeholder when LightGBM not available
#endif
    
    bool model_loaded;
    string model_path;
    
    // Feature scaling parameters (loaded from training)
    vector<double> feature_means;
    vector<double> feature_scales; 
    vector<string> feature_names;
    
    // Model configuration
    int num_features;
    double prediction_threshold;
    int num_trees;
    
    // Performance tracking
    mutable idx_t total_predictions;
    mutable idx_t postgresql_predictions;
    mutable idx_t duckdb_predictions;
};

//! Factory function to create and initialize LightGBM router
unique_ptr<LightGBMRouter> CreateLightGBMRouter(const string &model_dir);

//! Feature extraction specifically for LightGBM model
struct MLQueryFeatures {
    // Basic query structure
    int num_tables;
    int num_joins;
    int num_filters;
    int num_aggregates;
    
    // Query characteristics
    int has_groupby;
    int has_orderby; 
    int has_limit;
    int has_distinct;
    int has_subquery;
    int has_window;
    
    // Derived features (computed during extraction)
    double query_complexity;
    int is_analytical;
    int is_transactional;
    int has_complex_operations;
    double estimated_time_ratio;  // Heuristic estimate
    
    //! Convert to vector format for ML prediction
    vector<double> ToVector() const;
    
    //! Extract from DuckDB QueryFeatures
    static MLQueryFeatures FromQueryFeatures(const QueryFeatures &features);
};

//! Integration with existing query router
class EnhancedQueryRouter : public QueryRouter {
public:
    EnhancedQueryRouter();
    virtual ~EnhancedQueryRouter() override;
    
    //! Initialize with LightGBM model
    bool InitializeLightGBM(const string &model_dir);
    
    //! Override routing decision to use ML when available
    RoutingDecision MakeRoutingDecision(const LogicalOperator &plan, 
                                       const QueryFeatures &features, 
                                       const string &query_text) override;
    
    //! Get routing statistics
    string GetRoutingStats() const override;

private:
    unique_ptr<LightGBMRouter> ml_router;
    bool use_ml_routing;
    
    // Fallback to heuristic routing when ML fails
    RoutingDecision FallbackHeuristicRouting(const QueryFeatures &features);
};

//! Global functions for model management
bool LoadQueryRoutingModel(const string &model_dir);
void UnloadQueryRoutingModel();
bool IsMLRoutingAvailable();

} // namespace duckdb