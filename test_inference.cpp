#include "lightgbm_inference.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    lightgbm_inference::LightGBMPredictor predictor;
    
    // Load the trained model
    std::string model_path = "/home/wuy/DB/pg_duckdb_postgres/models/lightgbm_model.txt";
    
    std::cout << "Loading LightGBM model from: " << model_path << std::endl;
    
    if (!predictor.load_model(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Feature names:" << std::endl;
    
    const auto& feature_names = predictor.get_feature_names();
    for (size_t i = 0; i < feature_names.size(); ++i) {
        std::cout << "  " << i << ": " << feature_names[i] << std::endl;
    }
    
    // Test prediction with sample features
    std::map<std::string, double> sample_features = {
        {"query_length", 100.0},
        {"has_join", 1.0},
        {"has_group_by", 0.0},
        {"has_order_by", 1.0},
        {"has_where", 1.0},
        {"has_aggregation", 0.0},
        {"is_ap_query", 0.0},
        {"is_tp_query", 1.0},
        {"time_ratio", 1.2},
        {"log_time_ratio", 0.18},
        {"is_postgres_faster", 0.0},
        {"absolute_time_diff", 150.0},
        {"log_postgres_time", 2.5},
        {"log_duckdb_time", 2.3}
    };
    
    std::cout << "\nMaking prediction with sample features:" << std::endl;
    for (const auto& pair : sample_features) {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }
    
    double prediction = predictor.predict(sample_features);
    std::cout << "\nPrediction (log time difference): " << std::fixed << std::setprecision(6) << prediction << std::endl;
    
    // Convert back to time ratio
    double time_ratio_prediction = std::exp(prediction);
    std::cout << "Predicted time ratio (postgres/duckdb): " << std::fixed << std::setprecision(3) << time_ratio_prediction << std::endl;
    
    if (time_ratio_prediction > 1.0) {
        std::cout << "Recommendation: Use DuckDB (PostgreSQL is " << std::fixed << std::setprecision(1) 
                  << time_ratio_prediction << "x slower)" << std::endl;
    } else {
        std::cout << "Recommendation: Use PostgreSQL (DuckDB is " << std::fixed << std::setprecision(1) 
                  << (1.0 / time_ratio_prediction) << "x slower)" << std::endl;
    }
    
    return 0;
}