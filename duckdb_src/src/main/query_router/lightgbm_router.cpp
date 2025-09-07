/*
 * LightGBM-based Query Router Implementation
 * 
 * This implementation provides ML-based query routing using LightGBM
 * integrated directly into the DuckDB kernel for production deployment.
 */

#include "duckdb/main/lightgbm_router.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/serializer/buffered_file_reader.hpp"
#include "duckdb/optimizer/query_feature_logger.hpp"
#include "duckdb/common/string_util.hpp"
#include <fstream>
#include <iostream>

namespace duckdb {

// Model constants (will be updated by training pipeline)
constexpr int ROUTING_MODEL_NUM_FEATURES = 15;
constexpr double ROUTING_MODEL_THRESHOLD = 0.5;
constexpr int ROUTING_MODEL_NUM_TREES = 100;

// Default feature scaling parameters (updated during model loading)
static vector<double> DEFAULT_FEATURE_MEANS = {
    2.5, 1.2, 2.8, 0.8, 0.3, 0.4, 0.6, 0.2, 0.1, 0.05,  // Basic features
    4.5, 0.35, 0.25, 0.15, 1.0                             // Derived features
};

static vector<double> DEFAULT_FEATURE_SCALES = {
    2.1, 1.8, 2.4, 1.3, 0.46, 0.49, 0.49, 0.4, 0.3, 0.22, // Basic features  
    3.2, 0.48, 0.43, 0.36, 0.8                              // Derived features
};

static vector<string> ROUTING_FEATURE_NAMES = {
    "num_tables", "num_joins", "num_filters", "num_aggregates",
    "has_groupby", "has_orderby", "has_limit", "has_distinct", 
    "has_subquery", "has_window", "query_complexity", 
    "is_analytical", "is_transactional", "has_complex_operations", 
    "estimated_time_ratio"
};

//===--------------------------------------------------------------------===//
// MLQueryFeatures Implementation
//===--------------------------------------------------------------------===//

vector<double> MLQueryFeatures::ToVector() const {
    return {
        static_cast<double>(num_tables),
        static_cast<double>(num_joins),
        static_cast<double>(num_filters), 
        static_cast<double>(num_aggregates),
        static_cast<double>(has_groupby),
        static_cast<double>(has_orderby),
        static_cast<double>(has_limit),
        static_cast<double>(has_distinct),
        static_cast<double>(has_subquery),
        static_cast<double>(has_window),
        query_complexity,
        static_cast<double>(is_analytical),
        static_cast<double>(is_transactional),
        static_cast<double>(has_complex_operations),
        estimated_time_ratio
    };
}

MLQueryFeatures MLQueryFeatures::FromQueryFeatures(const QueryFeatures &features) {
    MLQueryFeatures ml_features;
    
    // Basic features
    ml_features.num_tables = features.num_tables;
    ml_features.num_joins = features.num_joins;  
    ml_features.num_filters = features.num_filters;
    ml_features.num_aggregates = features.num_aggregates;
    
    // Boolean features
    ml_features.has_groupby = features.has_groupby ? 1 : 0;
    ml_features.has_orderby = features.num_orderby > 0 ? 1 : 0;
    ml_features.has_limit = features.num_limits > 0 ? 1 : 0;
    ml_features.has_distinct = features.has_distinct ? 1 : 0;
    ml_features.has_subquery = features.has_subquery ? 1 : 0;
    ml_features.has_window = features.has_window ? 1 : 0;
    
    // Derived features
    ml_features.query_complexity = static_cast<double>(
        features.num_tables + features.num_joins + 
        features.num_filters + features.num_aggregates
    );
    
    // Analytical query heuristic
    ml_features.is_analytical = (features.num_aggregates > 0 || 
                                features.has_groupby || 
                                features.num_joins > 2) ? 1 : 0;
    
    // Transactional query heuristic  
    ml_features.is_transactional = (features.num_aggregates == 0 &&
                                   features.num_limits > 0 &&
                                   features.num_joins <= 1) ? 1 : 0;
    
    // Complex operations
    ml_features.has_complex_operations = (features.has_subquery ||
                                        features.has_window) ? 1 : 0;
    
    // Estimated time ratio (heuristic based on query complexity)
    double complexity_score = ml_features.query_complexity;
    if (ml_features.is_analytical) {
        complexity_score += 2.0;  // Analytical queries favor DuckDB
    }
    if (ml_features.is_transactional) {
        complexity_score -= 1.0;  // Transactional queries favor PostgreSQL
    }
    
    ml_features.estimated_time_ratio = std::max(0.1, std::min(10.0, complexity_score / 5.0));
    
    return ml_features;
}

//===--------------------------------------------------------------------===//
// LightGBMRouter Implementation  
//===--------------------------------------------------------------------===//

LightGBMRouter::LightGBMRouter() 
    : model_handle(nullptr), model_loaded(false), num_features(ROUTING_MODEL_NUM_FEATURES),
      prediction_threshold(ROUTING_MODEL_THRESHOLD), num_trees(ROUTING_MODEL_NUM_TREES),
      total_predictions(0), postgresql_predictions(0), duckdb_predictions(0) {
    
    // Initialize with default scaling parameters
    feature_means = DEFAULT_FEATURE_MEANS;
    feature_scales = DEFAULT_FEATURE_SCALES;
    feature_names = ROUTING_FEATURE_NAMES;
}

LightGBMRouter::~LightGBMRouter() {
#ifdef ENABLE_LIGHTGBM
    if (model_loaded && model_handle) {
        LGBM_BoosterFree(model_handle);
        model_handle = nullptr;
    }
#endif
    model_loaded = false;
}

bool LightGBMRouter::Initialize(const string &model_path) {
    this->model_path = model_path;
    
#ifdef ENABLE_LIGHTGBM
    try {
        // Load LightGBM model from text file
        string model_file = model_path + "/query_routing_model.txt";
        
        int result = LGBM_BoosterCreateFromModelfile(model_file.c_str(), &num_trees, &model_handle);
        if (result != 0) {
            std::cerr << "[LightGBMRouter] Failed to load model from " << model_file << std::endl;
            return false;
        }
        
        // Load scaler parameters
        if (!LoadScalerParameters(model_path + "/feature_scaler.json")) {
            std::cerr << "[LightGBMRouter] Failed to load scaler parameters" << std::endl;
            LGBM_BoosterFree(model_handle);
            model_handle = nullptr;
            return false;
        }
        
        model_loaded = true;
        std::cerr << "[LightGBMRouter] Model loaded successfully: " << num_trees << " trees" << std::endl;
        return true;
        
    } catch (const std::exception &e) {
        std::cerr << "[LightGBMRouter] Exception during model loading: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "[LightGBMRouter] LightGBM support not compiled. Using heuristic routing." << std::endl;
    return false;
#endif
}

bool LightGBMRouter::LoadScalerParameters(const string &scaler_path) {
    try {
        std::ifstream file(scaler_path);
        if (!file.is_open()) {
            return false;
        }
        
        // Simple JSON parsing for scaler parameters
        // In production, would use a proper JSON library
        string line;
        bool in_means = false, in_scales = false;
        
        feature_means.clear();
        feature_scales.clear();
        
        while (std::getline(file, line)) {
            line = StringUtil::Trim(line);
            
            if (line.find("\"mean\":") != string::npos) {
                in_means = true;
                continue;
            }
            if (line.find("\"scale\":") != string::npos) {
                in_scales = true;
                in_means = false;
                continue;
            }
            
            if (in_means && line.find("]") != string::npos) {
                in_means = false;
            }
            if (in_scales && line.find("]") != string::npos) {
                in_scales = false;
            }
            
            // Parse numeric values
            if ((in_means || in_scales) && line.find_first_of("0123456789.-") != string::npos) {
                // Extract numeric value
                size_t start = line.find_first_of("0123456789.-");
                size_t end = line.find_first_of(",]", start);
                if (end == string::npos) end = line.length();
                
                string num_str = line.substr(start, end - start);
                try {
                    double value = std::stod(num_str);
                    if (in_means) {
                        feature_means.push_back(value);
                    } else {
                        feature_scales.push_back(value);
                    }
                } catch (...) {
                    // Skip invalid numbers
                }
            }
        }
        
        // Validate loaded parameters
        if (feature_means.size() != ROUTING_MODEL_NUM_FEATURES || 
            feature_scales.size() != ROUTING_MODEL_NUM_FEATURES) {
            std::cerr << "[LightGBMRouter] Invalid scaler parameters: expected " 
                     << ROUTING_MODEL_NUM_FEATURES << " features, got means=" 
                     << feature_means.size() << ", scales=" << feature_scales.size() << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception &e) {
        std::cerr << "[LightGBMRouter] Error loading scaler: " << e.what() << std::endl;
        return false;
    }
}

RoutingDecision LightGBMRouter::PredictRouting(const QueryFeatures &features) {
    if (!model_loaded) {
        // Fallback to simple heuristic
        if (features.num_aggregates > 0 || features.has_groupby || features.num_joins > 2) {
            return RoutingDecision::ROUTE_TO_DUCKDB;
        }
        return RoutingDecision::ROUTE_TO_POSTGRESQL;
    }
    
#ifdef ENABLE_LIGHTGBM
    try {
        // Convert to ML features
        MLQueryFeatures ml_features = MLQueryFeatures::FromQueryFeatures(features);
        vector<double> raw_features = ml_features.ToVector();
        
        // Scale features
        vector<double> scaled_features = ScaleFeatures(raw_features);
        
        // Make prediction
        double prediction_result;
        const double* feature_data = scaled_features.data();
        
        int result = LGBM_BoosterPredictForMat(
            model_handle,
            feature_data,
            C_API_DTYPE_FLOAT64,
            1,  // nrow
            static_cast<int>(scaled_features.size()),  // ncol
            C_API_PREDICT_NORMAL,
            0,  // start_iteration
            -1, // num_iteration (use best)
            "", // parameter
            &prediction_result
        );
        
        if (result != 0) {
            std::cerr << "[LightGBMRouter] Prediction failed, using fallback" << std::endl;
            goto fallback;
        }
        
        // Convert probability to routing decision
        RoutingDecision decision = (prediction_result > prediction_threshold) ?
                                  RoutingDecision::ROUTE_TO_DUCKDB : 
                                  RoutingDecision::ROUTE_TO_POSTGRESQL;
        
        // Update statistics
        total_predictions++;
        if (decision == RoutingDecision::ROUTE_TO_POSTGRESQL) {
            postgresql_predictions++;
        } else {
            duckdb_predictions++;
        }
        
        return decision;
        
    } catch (const std::exception &e) {
        std::cerr << "[LightGBMRouter] Exception during prediction: " << e.what() << std::endl;
        goto fallback;
    }
    
fallback:
#endif
    // Fallback heuristic routing
    if (features.num_aggregates > 0 || features.has_groupby || features.num_joins > 2) {
        return RoutingDecision::ROUTE_TO_DUCKDB;
    }
    return RoutingDecision::ROUTE_TO_POSTGRESQL;
}

vector<double> LightGBMRouter::ScaleFeatures(const vector<double> &raw_features) {
    vector<double> scaled_features(raw_features.size());
    
    for (size_t i = 0; i < raw_features.size() && i < feature_means.size(); i++) {
        if (feature_scales[i] != 0.0) {
            scaled_features[i] = (raw_features[i] - feature_means[i]) / feature_scales[i];
        } else {
            scaled_features[i] = raw_features[i] - feature_means[i];
        }
    }
    
    return scaled_features;
}

string LightGBMRouter::GetModelInfo() const {
    string info = "LightGBM Query Router\n";
    info += "Model loaded: " + string(model_loaded ? "yes" : "no") + "\n";
    if (model_loaded) {
        info += "Model path: " + model_path + "\n";
        info += "Number of trees: " + std::to_string(num_trees) + "\n";
        info += "Number of features: " + std::to_string(num_features) + "\n";
        info += "Prediction threshold: " + std::to_string(prediction_threshold) + "\n";
        info += "Total predictions: " + std::to_string(total_predictions) + "\n";
        info += "PostgreSQL predictions: " + std::to_string(postgresql_predictions) + "\n";
        info += "DuckDB predictions: " + std::to_string(duckdb_predictions) + "\n";
    }
    return info;
}

//===--------------------------------------------------------------------===//
// EnhancedQueryRouter Implementation
//===--------------------------------------------------------------------===//

EnhancedQueryRouter::EnhancedQueryRouter() : use_ml_routing(false) {
}

EnhancedQueryRouter::~EnhancedQueryRouter() {
}

bool EnhancedQueryRouter::InitializeLightGBM(const string &model_dir) {
    ml_router = make_uniq<LightGBMRouter>();
    use_ml_routing = ml_router->Initialize(model_dir);
    
    if (use_ml_routing) {
        std::cerr << "[EnhancedQueryRouter] LightGBM routing enabled" << std::endl;
    } else {
        std::cerr << "[EnhancedQueryRouter] Using heuristic routing (LightGBM unavailable)" << std::endl;
    }
    
    return use_ml_routing;
}


RoutingDecision EnhancedQueryRouter::FallbackHeuristicRouting(const QueryFeatures &features) {
    // OLTP-style queries → PostgreSQL
    if (features.num_tables <= 2 && 
        features.num_joins <= 1 && 
        features.num_aggregates == 0 &&
        features.num_filters >= 1) {
        return RoutingDecision::ROUTE_TO_POSTGRESQL;
    }
    
    // OLAP-style queries → DuckDB
    if (features.num_aggregates > 0 || 
        features.num_joins > 2 ||
        features.has_groupby ||
        features.has_window) {
        return RoutingDecision::ROUTE_TO_DUCKDB;
    }
    
    // Default to DuckDB for complex queries
    return RoutingDecision::ROUTE_TO_DUCKDB;
}

string EnhancedQueryRouter::GetRoutingStats() const {
    string stats = "Enhanced Query Router Stats:\n";
    
    if (use_ml_routing && ml_router) {
        stats += "ML Routing: Enabled\n";
        stats += ml_router->GetModelInfo();
    } else {
        stats += "ML Routing: Disabled";
    }
    
    return stats;
}

bool EnhancedQueryRouter::IsMLRoutingEnabled() const {
    return use_ml_routing;
}

//===--------------------------------------------------------------------===//
// Factory Functions
//===--------------------------------------------------------------------===//

unique_ptr<LightGBMRouter> CreateLightGBMRouter(const string &model_dir) {
    auto router = make_uniq<LightGBMRouter>();
    if (router->Initialize(model_dir)) {
        return router;
    }
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Global Model Management
//===--------------------------------------------------------------------===//

static unique_ptr<EnhancedQueryRouter> g_enhanced_router = nullptr;

bool LoadQueryRoutingModel(const string &model_dir) {
    g_enhanced_router = make_uniq<EnhancedQueryRouter>();
    return g_enhanced_router->InitializeLightGBM(model_dir);
}

void UnloadQueryRoutingModel() {
    g_enhanced_router.reset();
}

bool IsMLRoutingAvailable() {
    return g_enhanced_router && g_enhanced_router->IsMLRoutingEnabled();
}

RoutingDecision EnhancedQueryRouter::PredictRoutingFromFeatures(const QueryFeatures &features) {
    // Use ML routing if available
    if (use_ml_routing && ml_router) {
        try {
            return ml_router->PredictRouting(features);
        } catch (const std::exception &e) {
            std::cerr << "[EnhancedQueryRouter] ML prediction failed: " << e.what() << std::endl;
            // Fall through to heuristic routing
        }
    }
    
    // Fallback to heuristic routing
    return FallbackHeuristicRouting(features);
}

} // namespace duckdb