#include <iostream>
#include <map>
#include "lightgbm_inference.h"

int main() {
    std::cout << "Testing LightGBM model loading and inference..." << std::endl;
    
    lightgbm_inference::LightGBMPredictor predictor;
    
    // Try to load the trained model
    bool loaded = predictor.load_model("/home/wuy/DB/pg_duckdb_postgres/models/lightgbm_model.txt");
    if (loaded) {
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "Model is ready: " << (predictor.is_loaded() ? "Yes" : "No") << std::endl;
        
        // Test prediction with some sample features
        std::map<std::string, double> features;
        features["time_ratio"] = 1.5;
        features["log_time_ratio"] = 0.41;
        features["absolute_time_diff"] = 0.1;
        features["log_postgres_time"] = 2.0;
        features["log_duckdb_time"] = 1.8;
        features["query_length"] = 100.0;
        features["has_group_by"] = 0.0;
        features["is_postgres_faster"] = 0.0;
        features["is_ap_query"] = 1.0;
        features["has_aggregation"] = 1.0;
        features["has_order_by"] = 0.0;
        features["is_tp_query"] = 0.0;
        features["has_where"] = 1.0;
        features["has_join"] = 0.0;
        
        double prediction = predictor.predict(features);
        std::cout << "Sample prediction: " << prediction << std::endl;
        
        // Get feature names
        auto feature_names = predictor.get_feature_names();
        std::cout << "Model expects " << feature_names.size() << " features" << std::endl;
        
    } else {
        std::cout << "❌ Failed to load model" << std::endl;
        return 1;
    }
    
    return 0;
}