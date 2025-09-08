#include <iostream>
#include "lightgbm_inference.h"

int main() {
    std::cout << "Testing LightGBM inference engine..." << std::endl;
    
    lightgbm_inference::LightGBMPredictor predictor;
    std::cout << "Predictor created successfully" << std::endl;
    
    // Test with dummy features
    std::map<std::string, double> features;
    features["feature1"] = 1.0;
    features["feature2"] = 2.0;
    
    double prediction = predictor.predict(features);
    std::cout << "Prediction (no model loaded): " << prediction << std::endl;
    
    return 0;
}