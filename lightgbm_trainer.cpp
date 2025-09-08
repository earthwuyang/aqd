/*
 * lightgbm_trainer.cpp
 *
 * C++ LightGBM Training Framework for AQD
 * Trains models to predict log-transformed execution time gap between PostgreSQL and DuckDB
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

// JSON parsing
#include <nlohmann/json.hpp>

// LightGBM
#include <LightGBM/c_api.h>

// SHAP analysis
#include <shap/shap.hpp>

using json = nlohmann::json;

class AQDLightGBMTrainer {
private:
    struct TrainingData {
        std::vector<std::vector<double>> features;
        std::vector<double> targets;
        std::vector<std::string> feature_names;
        std::vector<std::string> query_ids;
        
        size_t num_samples() const { return features.size(); }
        size_t num_features() const { return features.empty() ? 0 : features[0].size(); }
    };
    
    struct ModelParams {
        std::string objective = "regression";
        std::string metric = "rmse";
        int num_leaves = 31;
        double learning_rate = 0.05;
        int feature_fraction_seed = 2;
        int bagging_seed = 3;
        int data_random_seed = 1;
        double feature_fraction = 0.9;
        double bagging_fraction = 0.8;
        int bagging_freq = 5;
        int min_data_in_leaf = 20;
        double min_sum_hessian_in_leaf = 1e-3;
        int num_boost_round = 1000;
        int early_stopping_rounds = 50;
        bool verbose = true;
        
        // Taylor-weighted boosting parameters (AQD specific)
        double taylor_weight_alpha = 0.5;
        bool use_taylor_weighting = true;
    };
    
    TrainingData training_data;
    TrainingData validation_data;
    ModelParams params;
    BoosterHandle booster = nullptr;
    
    // Feature importance and SHAP analysis
    std::vector<double> feature_importance;
    std::vector<std::string> selected_features;
    
    // Training statistics
    struct TrainingStats {
        double train_rmse = 0.0;
        double val_rmse = 0.0;
        double train_mae = 0.0;
        double val_mae = 0.0;
        double training_time_seconds = 0.0;
        int best_iteration = 0;
        size_t num_selected_features = 0;
    } stats;

public:
    AQDLightGBMTrainer() = default;
    
    ~AQDLightGBMTrainer() {
        if (booster != nullptr) {
            LGBM_BoosterFree(booster);
        }
    }
    
    bool load_training_data(const std::string& json_file) {
        std::cout << "Loading training data from: " << json_file << std::endl;
        
        std::ifstream file(json_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << json_file << std::endl;
            return false;
        }
        
        json data;
        try {
            file >> data;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            return false;
        }
        
        // Parse the training data
        return parse_json_data(data, training_data);
    }
    
    bool parse_json_data(const json& data, TrainingData& td) {
        if (!data.is_array()) {
            std::cerr << "Error: JSON data should be an array of records" << std::endl;
            return false;
        }
        
        std::cout << "Parsing " << data.size() << " records..." << std::endl;
        
        // Collect feature names from first record
        std::set<std::string> all_feature_names;
        for (const auto& record : data) {
            if (record.contains("features") && record["features"].is_object()) {
                for (auto& [key, value] : record["features"].items()) {
                    if (key.starts_with("feature_") && value.is_number()) {
                        all_feature_names.insert(key);
                    }
                }
            }
        }
        
        td.feature_names = std::vector<std::string>(all_feature_names.begin(), all_feature_names.end());
        std::sort(td.feature_names.begin(), td.feature_names.end());
        
        std::cout << "Found " << td.feature_names.size() << " features" << std::endl;
        
        // Parse each record
        for (const auto& record : data) {
            // Skip if missing required fields
            if (!record.contains("log_time_gap") || 
                !record.contains("pg_success") || 
                !record.contains("duck_success")) {
                continue;
            }
            
            // Skip if either execution failed
            if (!record["pg_success"].get<bool>() || !record["duck_success"].get<bool>()) {
                continue;
            }
            
            // Extract target (log time gap)
            double target = record["log_time_gap"].get<double>();
            
            // Skip if target is invalid
            if (std::isnan(target) || std::isinf(target)) {
                continue;
            }
            
            // Extract features
            std::vector<double> feature_vector(td.feature_names.size(), 0.0);
            
            if (record.contains("features") && record["features"].is_object()) {
                for (size_t i = 0; i < td.feature_names.size(); ++i) {
                    const std::string& fname = td.feature_names[i];
                    if (record["features"].contains(fname) && record["features"][fname].is_number()) {
                        feature_vector[i] = record["features"][fname].get<double>();
                    }
                }
            }
            
            // Add to training data
            td.features.push_back(feature_vector);
            td.targets.push_back(target);
            
            if (record.contains("query_id")) {
                td.query_ids.push_back(record["query_id"].get<std::string>());
            } else {
                td.query_ids.push_back("query_" + std::to_string(td.targets.size()));
            }
        }
        
        std::cout << "Loaded " << td.num_samples() << " valid samples with " 
                  << td.num_features() << " features each" << std::endl;
        
        return td.num_samples() > 0;
    }
    
    void split_train_validation(double val_ratio = 0.2) {
        std::cout << "Splitting data: " << (1.0 - val_ratio) * 100 << "% train, " 
                  << val_ratio * 100 << "% validation" << std::endl;
        
        size_t total_samples = training_data.num_samples();
        size_t val_samples = static_cast<size_t>(total_samples * val_ratio);
        
        // Create random indices
        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Split data
        validation_data.feature_names = training_data.feature_names;
        
        for (size_t i = 0; i < val_samples; ++i) {
            size_t idx = indices[i];
            validation_data.features.push_back(training_data.features[idx]);
            validation_data.targets.push_back(training_data.targets[idx]);
            validation_data.query_ids.push_back(training_data.query_ids[idx]);
        }
        
        // Keep remaining data for training
        TrainingData new_training_data;
        new_training_data.feature_names = training_data.feature_names;
        
        for (size_t i = val_samples; i < total_samples; ++i) {
            size_t idx = indices[i];
            new_training_data.features.push_back(training_data.features[idx]);
            new_training_data.targets.push_back(training_data.targets[idx]);
            new_training_data.query_ids.push_back(training_data.query_ids[idx]);
        }
        
        training_data = std::move(new_training_data);
        
        std::cout << "Training samples: " << training_data.num_samples() << std::endl;
        std::cout << "Validation samples: " << validation_data.num_samples() << std::endl;
    }
    
    std::vector<double> compute_taylor_weights(const std::vector<double>& targets) {
        std::vector<double> weights(targets.size());
        
        if (!params.use_taylor_weighting) {
            std::fill(weights.begin(), weights.end(), 1.0);
            return weights;
        }
        
        // Compute weights based on prediction cost
        // Higher weights for costly mispredictions (as in AQD paper)
        for (size_t i = 0; i < targets.size(); ++i) {
            double abs_gap = std::abs(targets[i]);
            // Exponential weighting for larger gaps
            weights[i] = 1.0 + params.taylor_weight_alpha * std::exp(abs_gap / 5.0);
        }
        
        return weights;
    }
    
    bool train_model() {
        if (training_data.num_samples() == 0) {
            std::cerr << "Error: No training data loaded" << std::endl;
            return false;
        }
        
        std::cout << "\n=== Training LightGBM Model ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Prepare data for LightGBM
        DatasetHandle train_dataset, val_dataset;
        
        // Flatten training features
        std::vector<double> train_features_flat;
        for (const auto& sample : training_data.features) {
            train_features_flat.insert(train_features_flat.end(), sample.begin(), sample.end());
        }
        
        // Compute Taylor weights
        std::vector<double> weights = compute_taylor_weights(training_data.targets);
        
        // Create training dataset
        int ret = LGBM_DatasetCreateFromMat(
            train_features_flat.data(),
            C_API_DTYPE_FLOAT64,
            static_cast<int>(training_data.num_samples()),
            static_cast<int>(training_data.num_features()),
            1, // is_row_major
            "", // parameters
            nullptr, // reference
            &train_dataset
        );
        
        if (ret != 0) {
            std::cerr << "Error creating training dataset: " << LGBM_GetLastError() << std::endl;
            return false;
        }
        
        // Set labels and weights
        ret = LGBM_DatasetSetField(train_dataset, "label", 
                                  training_data.targets.data(), 
                                  static_cast<int>(training_data.targets.size()),
                                  C_API_DTYPE_FLOAT64);
        if (ret != 0) {
            std::cerr << "Error setting labels: " << LGBM_GetLastError() << std::endl;
            return false;
        }
        
        ret = LGBM_DatasetSetField(train_dataset, "weight", 
                                  weights.data(), 
                                  static_cast<int>(weights.size()),
                                  C_API_DTYPE_FLOAT64);
        if (ret != 0) {
            std::cerr << "Error setting weights: " << LGBM_GetLastError() << std::endl;
            return false;
        }
        
        // Create validation dataset if available
        if (validation_data.num_samples() > 0) {
            std::vector<double> val_features_flat;
            for (const auto& sample : validation_data.features) {
                val_features_flat.insert(val_features_flat.end(), sample.begin(), sample.end());
            }
            
            ret = LGBM_DatasetCreateFromMat(
                val_features_flat.data(),
                C_API_DTYPE_FLOAT64,
                static_cast<int>(validation_data.num_samples()),
                static_cast<int>(validation_data.num_features()),
                1, // is_row_major
                "", // parameters
                train_dataset, // reference
                &val_dataset
            );
            
            if (ret != 0) {
                std::cerr << "Error creating validation dataset: " << LGBM_GetLastError() << std::endl;
                return false;
            }
            
            ret = LGBM_DatasetSetField(val_dataset, "label", 
                                      validation_data.targets.data(), 
                                      static_cast<int>(validation_data.targets.size()),
                                      C_API_DTYPE_FLOAT64);
        }
        
        // Create parameter string
        std::string param_str = build_param_string();
        std::cout << "Parameters: " << param_str << std::endl;
        
        // Create booster
        ret = LGBM_BoosterCreate(train_dataset, param_str.c_str(), &booster);
        if (ret != 0) {
            std::cerr << "Error creating booster: " << LGBM_GetLastError() << std::endl;
            return false;
        }
        
        // Add validation data if available
        if (validation_data.num_samples() > 0) {
            ret = LGBM_BoosterAddValidData(booster, val_dataset);
            if (ret != 0) {
                std::cerr << "Error adding validation data: " << LGBM_GetLastError() << std::endl;
                return false;
            }
        }
        
        // Train model
        std::cout << "Training for " << params.num_boost_round << " rounds..." << std::endl;
        
        int best_iter = 0;
        double best_score = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < params.num_boost_round; ++iter) {
            int is_finished;
            ret = LGBM_BoosterUpdateOneIter(booster, &is_finished);
            if (ret != 0) {
                std::cerr << "Error in training iteration " << iter << ": " << LGBM_GetLastError() << std::endl;
                break;
            }
            
            if (is_finished) {
                std::cout << "Training finished early at iteration " << iter << std::endl;
                break;
            }
            
            // Evaluate on validation set
            if (validation_data.num_samples() > 0 && iter % 10 == 0) {
                double val_score;
                ret = LGBM_BoosterGetEval(booster, 1, &val_score);
                if (ret == 0) {
                    if (val_score < best_score) {
                        best_score = val_score;
                        best_iter = iter;
                    }
                    
                    if (params.verbose && iter % 50 == 0) {
                        std::cout << "Iteration " << iter << ", validation RMSE: " 
                                  << val_score << std::endl;
                    }
                    
                    // Early stopping
                    if (iter - best_iter > params.early_stopping_rounds) {
                        std::cout << "Early stopping at iteration " << iter 
                                  << " (best: " << best_iter << ")" << std::endl;
                        break;
                    }
                }
            }
        }
        
        // Record training time
        auto end_time = std::chrono::high_resolution_clock::now();
        stats.training_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        stats.best_iteration = best_iter;
        
        // Evaluate final model
        evaluate_model();
        
        // Extract feature importance
        extract_feature_importance();
        
        // Cleanup datasets
        LGBM_DatasetFree(train_dataset);
        if (validation_data.num_samples() > 0) {
            LGBM_DatasetFree(val_dataset);
        }
        
        std::cout << "Training completed in " << stats.training_time_seconds << " seconds" << std::endl;
        return true;
    }
    
    std::string build_param_string() {
        std::ostringstream oss;
        oss << "objective=" << params.objective
            << " metric=" << params.metric
            << " num_leaves=" << params.num_leaves
            << " learning_rate=" << params.learning_rate
            << " feature_fraction=" << params.feature_fraction
            << " bagging_fraction=" << params.bagging_fraction
            << " bagging_freq=" << params.bagging_freq
            << " min_data_in_leaf=" << params.min_data_in_leaf
            << " min_sum_hessian_in_leaf=" << params.min_sum_hessian_in_leaf
            << " feature_fraction_seed=" << params.feature_fraction_seed
            << " bagging_seed=" << params.bagging_seed
            << " data_random_seed=" << params.data_random_seed
            << " verbosity=" << (params.verbose ? 1 : -1);
        return oss.str();
    }
    
    void evaluate_model() {
        if (!booster) return;
        
        // Evaluate on training set
        std::vector<double> train_pred = predict(training_data);
        stats.train_rmse = compute_rmse(training_data.targets, train_pred);
        stats.train_mae = compute_mae(training_data.targets, train_pred);
        
        // Evaluate on validation set
        if (validation_data.num_samples() > 0) {
            std::vector<double> val_pred = predict(validation_data);
            stats.val_rmse = compute_rmse(validation_data.targets, val_pred);
            stats.val_mae = compute_mae(validation_data.targets, val_pred);
        }
        
        std::cout << "\n=== Model Evaluation ===" << std::endl;
        std::cout << "Training RMSE: " << stats.train_rmse << std::endl;
        std::cout << "Training MAE: " << stats.train_mae << std::endl;
        if (validation_data.num_samples() > 0) {
            std::cout << "Validation RMSE: " << stats.val_rmse << std::endl;
            std::cout << "Validation MAE: " << stats.val_mae << std::endl;
        }
    }
    
    std::vector<double> predict(const TrainingData& data) {
        if (!booster || data.num_samples() == 0) {
            return {};
        }
        
        std::vector<double> predictions(data.num_samples());
        
        for (size_t i = 0; i < data.num_samples(); ++i) {
            int64_t out_len;
            double pred;
            
            int ret = LGBM_BoosterPredictForMat(
                booster,
                data.features[i].data(),
                C_API_DTYPE_FLOAT64,
                1, // nrow
                static_cast<int>(data.num_features()), // ncol
                1, // is_row_major
                C_API_PREDICT_NORMAL, // predict_type
                0, // start_iteration
                -1, // num_iteration
                "", // parameter
                &out_len,
                &pred
            );
            
            if (ret == 0 && out_len > 0) {
                predictions[i] = pred;
            } else {
                predictions[i] = 0.0;
            }
        }
        
        return predictions;
    }
    
    double compute_rmse(const std::vector<double>& actual, const std::vector<double>& predicted) {
        double sum_sq_error = 0.0;
        for (size_t i = 0; i < actual.size() && i < predicted.size(); ++i) {
            double error = actual[i] - predicted[i];
            sum_sq_error += error * error;
        }
        return std::sqrt(sum_sq_error / actual.size());
    }
    
    double compute_mae(const std::vector<double>& actual, const std::vector<double>& predicted) {
        double sum_abs_error = 0.0;
        for (size_t i = 0; i < actual.size() && i < predicted.size(); ++i) {
            sum_abs_error += std::abs(actual[i] - predicted[i]);
        }
        return sum_abs_error / actual.size();
    }
    
    void extract_feature_importance() {
        if (!booster) return;
        
        int num_features = static_cast<int>(training_data.num_features());
        feature_importance.resize(num_features);
        
        int ret = LGBM_BoosterFeatureImportance(
            booster,
            0, // num_iteration (0 means use best)
            0, // importance_type (0 = split, 1 = gain)
            feature_importance.data()
        );
        
        if (ret != 0) {
            std::cerr << "Error getting feature importance: " << LGBM_GetLastError() << std::endl;
            return;
        }
        
        std::cout << "\n=== Top 20 Most Important Features ===" << std::endl;
        
        // Create pairs of (importance, feature_name) for sorting
        std::vector<std::pair<double, std::string>> importance_pairs;
        for (size_t i = 0; i < feature_importance.size(); ++i) {
            importance_pairs.emplace_back(feature_importance[i], training_data.feature_names[i]);
        }
        
        // Sort by importance (descending)
        std::sort(importance_pairs.begin(), importance_pairs.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Print top features
        for (size_t i = 0; i < std::min(size_t(20), importance_pairs.size()); ++i) {
            std::cout << i + 1 << ". " << importance_pairs[i].second 
                      << ": " << importance_pairs[i].first << std::endl;
        }
    }
    
    void perform_shap_analysis(size_t max_samples = 1000) {
        std::cout << "\n=== SHAP Analysis for Feature Selection ===" << std::endl;
        
        // This would require actual SHAP library integration
        // For now, we use feature importance as a proxy
        
        // Select top features based on importance
        std::vector<std::pair<double, size_t>> importance_with_index;
        for (size_t i = 0; i < feature_importance.size(); ++i) {
            importance_with_index.emplace_back(feature_importance[i], i);
        }
        
        std::sort(importance_with_index.begin(), importance_with_index.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Select top 50 features (or fewer if we have fewer features)
        size_t num_selected = std::min(size_t(50), importance_with_index.size());
        selected_features.clear();
        
        for (size_t i = 0; i < num_selected; ++i) {
            size_t feature_idx = importance_with_index[i].second;
            selected_features.push_back(training_data.feature_names[feature_idx]);
        }
        
        stats.num_selected_features = selected_features.size();
        
        std::cout << "Selected " << selected_features.size() << " features based on importance" << std::endl;
        
        // Export selected features
        export_selected_features();
    }
    
    void export_selected_features() {
        std::ofstream file("aqd_selected_features.txt");
        for (const auto& feature : selected_features) {
            file << feature << std::endl;
        }
        std::cout << "Selected features exported to: aqd_selected_features.txt" << std::endl;
    }
    
    bool save_model(const std::string& filename) {
        if (!booster) {
            std::cerr << "Error: No trained model to save" << std::endl;
            return false;
        }
        
        int ret = LGBM_BoosterSaveModel(booster, 0, -1, 0, filename.c_str());
        if (ret != 0) {
            std::cerr << "Error saving model: " << LGBM_GetLastError() << std::endl;
            return false;
        }
        
        std::cout << "Model saved to: " << filename << std::endl;
        
        // Also export training statistics
        std::string stats_file = filename + "_stats.json";
        export_training_stats(stats_file);
        
        return true;
    }
    
    void export_training_stats(const std::string& filename) {
        json stats_json;
        stats_json["train_rmse"] = stats.train_rmse;
        stats_json["val_rmse"] = stats.val_rmse;
        stats_json["train_mae"] = stats.train_mae;
        stats_json["val_mae"] = stats.val_mae;
        stats_json["training_time_seconds"] = stats.training_time_seconds;
        stats_json["best_iteration"] = stats.best_iteration;
        stats_json["num_selected_features"] = stats.num_selected_features;
        stats_json["model_params"] = {
            {"objective", params.objective},
            {"num_leaves", params.num_leaves},
            {"learning_rate", params.learning_rate},
            {"use_taylor_weighting", params.use_taylor_weighting}
        };
        
        std::ofstream file(filename);
        file << stats_json.dump(2) << std::endl;
        
        std::cout << "Training statistics exported to: " << filename << std::endl;
    }
    
    void print_summary() {
        std::cout << "\n=== AQD LightGBM Training Summary ===" << std::endl;
        std::cout << "Training samples: " << training_data.num_samples() << std::endl;
        std::cout << "Validation samples: " << validation_data.num_samples() << std::endl;
        std::cout << "Features: " << training_data.num_features() << std::endl;
        std::cout << "Selected features: " << stats.num_selected_features << std::endl;
        std::cout << "Training time: " << stats.training_time_seconds << " seconds" << std::endl;
        std::cout << "Best iteration: " << stats.best_iteration << std::endl;
        std::cout << "Final validation RMSE: " << stats.val_rmse << std::endl;
        std::cout << "Final validation MAE: " << stats.val_mae << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <training_data.json> [output_model.txt]" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_model = argc > 2 ? argv[2] : "aqd_lightgbm_model.txt";
    
    std::cout << "AQD LightGBM Trainer" << std::endl;
    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Output: " << output_model << std::endl;
    
    try {
        AQDLightGBMTrainer trainer;
        
        // Load training data
        if (!trainer.load_training_data(input_file)) {
            return 1;
        }
        
        // Split train/validation
        trainer.split_train_validation(0.2);
        
        // Train model
        if (!trainer.train_model()) {
            return 1;
        }
        
        // Perform SHAP analysis
        trainer.perform_shap_analysis();
        
        // Save model
        if (!trainer.save_model(output_model)) {
            return 1;
        }
        
        // Print summary
        trainer.print_summary();
        
        std::cout << "\nTraining completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}