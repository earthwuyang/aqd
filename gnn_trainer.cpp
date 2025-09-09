#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <glob.h>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <cmath>

/*
 GNN trainer for unified JSON format:
 - Reads unified JSON file where each record contains:
   - query_text: SQL query
   - postgres_time / duckdb_time: execution times
   - log_time_difference: target (log ratio)
   - features: extracted AQD features
   - postgres_plan_json: query plan tree
 - Extracts graph features from plan tree
 - Trains a GNN model with mean aggregation
 - Outputs model compatible with kernel gnn_inference.c
*/

using json = nlohmann::json;

static int op_type_id(const std::string& node_type) {
    if (node_type == "Seq Scan") return 1;
    if (node_type == "Index Scan") return 2;
    if (node_type == "Bitmap Heap Scan") return 3;
    if (node_type == "Nested Loop") return 4;
    if (node_type == "Merge Join") return 5;
    if (node_type == "Hash Join") return 6;
    if (node_type == "Sort") return 7;
    if (node_type == "Aggregate") return 8;
    if (node_type == "Group") return 9;
    if (node_type == "Hash") return 10;
    if (node_type == "Materialize") return 11;
    if (node_type == "Merge Append") return 12;
    if (node_type == "Result") return 13;
    if (node_type == "Append") return 14;
    if (node_type == "Limit") return 15;
    return 0; // Unknown
}

struct NodeFeat {
    std::vector<double> h; // hidden vector
};

static void relu(std::vector<double>& v) {
    for (auto& x : v) if (x < 0.0) x = 0.0;
}

static void add_inplace(std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

static void div_inplace(std::vector<double>& a, double d) {
    if (d == 0.0) return;
    for (auto& x : a) x /= d;
}

// Build fixed hidden from plan tree: h = ReLU(x + mean(child_h)) with x as one-hot and numeric fields
static std::vector<double> aggregate_plan(const json& plan, int in_features) {
    const int k = 16; // number of one-hot operation types
    std::vector<double> x(in_features, 0.0);
    
    // Extract node type and create one-hot encoding
    std::string node_type = plan.value("Node Type", "");
    int id = op_type_id(node_type);
    if (id >= 0 && id < k && id < in_features) x[id] = 1.0;
    
    // Add numeric features
    int idx = k;
    double plan_rows = plan.value("Plan Rows", 0.0);
    double plan_width = plan.value("Plan Width", 0.0);
    double total_cost = plan.value("Total Cost", 0.0);
    double startup_cost = plan.value("Startup Cost", 0.0);
    
    // Normalize features
    double cpr = (plan_rows > 0.0) ? (total_cost / plan_rows) : total_cost;
    double log_rows = (plan_rows > 0.0) ? std::log(plan_rows + 1.0) : 0.0;
    
    if (idx < in_features) x[idx++] = log_rows;
    if (idx < in_features) x[idx++] = plan_width / 100.0;  // normalize width
    if (idx < in_features) x[idx++] = std::log(total_cost + 1.0);
    if (idx < in_features) x[idx++] = std::log(startup_cost + 1.0);
    if (idx < in_features) x[idx++] = cpr / 1000.0;  // normalize cost per row

    // Aggregate children
    std::vector<double> agg(in_features, 0.0);
    int child_count = 0;
    if (plan.contains("Plans") && plan["Plans"].is_array()) {
        for (const auto& ch : plan["Plans"]) {
            auto hc = aggregate_plan(ch, in_features);
            add_inplace(agg, hc);
            child_count++;
        }
    }
    if (child_count > 0) div_inplace(agg, (double)child_count);
    
    // Combine node features with aggregated children
    add_inplace(agg, x);
    relu(agg);
    return agg;
}

static void solve_ridge(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double lambda,
                        std::vector<double>& w, double& b) {
    // Solve (Xb^T Xb + lambda I) w = Xb^T y, where Xb = [X | 1]
    const size_t n = X.size();
    if (n == 0) { w.clear(); b = 0.0; return; }
    const size_t d = X[0].size();
    const size_t D = d + 1; // bias
    std::vector<double> A(D*D, 0.0);
    std::vector<double> rhs(D, 0.0);
    
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> xb(D, 1.0);
        for (size_t j = 0; j < d; ++j) xb[j] = X[i][j];
        // accumulate
        for (size_t r = 0; r < D; ++r) {
            rhs[r] += xb[r] * y[i];
            for (size_t c = 0; c < D; ++c) A[r*D + c] += xb[r] * xb[c];
        }
    }
    
    // ridge regularization
    for (size_t j = 0; j < d; ++j) A[j*D + j] += lambda;
    
    // Solve by Gaussian elimination
    for (size_t r = 0; r < D; ++r) {
        // pivot
        size_t piv = r;
        for (size_t rr = r+1; rr < D; ++rr) {
            if (std::fabs(A[rr*D + r]) > std::fabs(A[piv*D + r])) piv = rr;
        }
        if (piv != r) {
            for (size_t c = 0; c < D; ++c) std::swap(A[r*D + c], A[piv*D + c]);
            std::swap(rhs[r], rhs[piv]);
        }
        double diag = A[r*D + r];
        if (std::fabs(diag) < 1e-12) continue;
        for (size_t c = r; c < D; ++c) A[r*D + c] /= diag;
        rhs[r] /= diag;
        for (size_t rr = 0; rr < D; ++rr) if (rr != r) {
            double f = A[rr*D + r];
            for (size_t c = r; c < D; ++c) A[rr*D + c] -= f * A[r*D + c];
            rhs[rr] -= f * rhs[r];
        }
    }
    
    // solution in rhs
    w.assign(d, 0.0);
    for (size_t j = 0; j < d; ++j) w[j] = rhs[j];
    b = rhs[d];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output_model.txt> <data_file1.json> [data_file2.json ...]\n";
        std::cerr << "   or: " << argv[0] << " <output_model.txt> --dir <data_directory>\n";
        std::cerr << "  output_model.txt: Output GNN model file\n";
        std::cerr << "  data_file.json: Unified training data JSON files\n";
        std::cerr << "  --dir: Load all *_unified_training_data.json files from directory\n";
        return 1;
    }
    
    std::string out_path = argv[1];
    std::vector<std::string> data_files;
    
    // Parse command line arguments
    if (argc >= 3 && std::string(argv[2]) == "--dir") {
        if (argc != 4) {
            std::cerr << "Error: --dir requires a directory path\n";
            return 1;
        }
        // Load all unified files from directory
        std::string dir_path = argv[3];
        std::string pattern = dir_path + "/*_unified_training_data.json";
        
        // Use directory listing (platform specific, simplified for Linux)
        glob_t glob_result;
        glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            data_files.push_back(glob_result.gl_pathv[i]);
        }
        globfree(&glob_result);
        
        if (data_files.empty()) {
            std::cerr << "No unified training data files found in " << dir_path << "\n";
            return 1;
        }
    } else {
        // Load specific files from command line
        for (int i = 2; i < argc; ++i) {
            data_files.push_back(argv[i]);
        }
    }
    
    std::cerr << "Loading data from " << data_files.size() << " file(s)...\n";
    
    // Collect all data from multiple files
    json all_data = json::array();
    
    for (const auto& data_path : data_files) {
        std::cerr << "  Loading: " << data_path << "\n";
        std::ifstream fin(data_path);
        if (!fin) {
            std::cerr << "    Warning: Failed to open " << data_path << ", skipping\n";
            continue;
        }
        
        json data;
        try {
            fin >> data;
        } catch (const std::exception& e) {
            std::cerr << "    Warning: Failed to parse JSON from " << data_path << ": " << e.what() << "\n";
            fin.close();
            continue;
        }
        fin.close();
        
        if (data.is_array()) {
            for (const auto& item : data) {
                all_data.push_back(item);
            }
            std::cerr << "    Added " << data.size() << " records\n";
        } else {
            std::cerr << "    Warning: JSON is not an array, skipping\n";
        }
    }
    
    if (all_data.empty()) {
        std::cerr << "No data loaded from any file\n";
        return 1;
    }
    
    std::cerr << "Total records loaded: " << all_data.size() << "\n";
    
    // Now process all_data as before
    json& data = all_data;
    
    const int in_features = 32; // Increased for more features
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    // Store original data for analysis
    std::vector<double> postgres_times;
    std::vector<double> duckdb_times;
    std::vector<std::string> datasets;
    std::vector<std::string> query_types;
    
    size_t parsed = 0, matched = 0, skipped = 0;
    
    // Process each record
    for (const auto& record : data) {
        parsed++;
        
        // Check for required fields
        if (!record.contains("log_time_difference") || record["log_time_difference"].is_null()) {
            skipped++;
            continue;
        }
        
        if (!record.contains("postgres_plan_json") || record["postgres_plan_json"].is_null()) {
            skipped++;
            continue;
        }
        
        double target = record["log_time_difference"];
        
        try {
            const json& plan = record["postgres_plan_json"];
            
            // Handle different plan formats
            const json* plan_root = nullptr;
            if (plan.is_array() && !plan.empty()) {
                // Array format - take first element
                if (plan[0].contains("Plan")) {
                    plan_root = &plan[0]["Plan"];
                } else {
                    plan_root = &plan[0];
                }
            } else if (plan.is_object()) {
                // Object format
                if (plan.contains("Plan")) {
                    plan_root = &plan["Plan"];
                } else {
                    plan_root = &plan;
                }
            }
            
            if (!plan_root) {
                skipped++;
                continue;
            }
            
            // Extract features from plan tree
            auto h = aggregate_plan(*plan_root, in_features);
            
            // Optional: Add AQD features if available
            if (record.contains("features") && record["features"].is_object()) {
                const auto& features = record["features"];
                int feat_idx = in_features - 10; // Reserve last 10 slots for AQD features
                
                // Add some key AQD features if they exist
                if (features.contains("aqd_feature_query_length") && feat_idx < in_features) {
                    h[feat_idx++] = std::log(features["aqd_feature_query_length"].get<double>() + 1.0);
                }
                if (features.contains("aqd_complexity_score") && feat_idx < in_features) {
                    h[feat_idx++] = features["aqd_complexity_score"].get<double>() / 100.0;
                }
            }
            
            X.push_back(std::move(h));
            y.push_back(target);
            
            // Store original times for analysis
            if (record.contains("postgres_time")) {
                postgres_times.push_back(record["postgres_time"].get<double>());
            } else {
                postgres_times.push_back(0.0);
            }
            
            if (record.contains("duckdb_time")) {
                duckdb_times.push_back(record["duckdb_time"].get<double>());
            } else {
                duckdb_times.push_back(0.0);
            }
            
            if (record.contains("dataset")) {
                datasets.push_back(record["dataset"].get<std::string>());
            } else {
                datasets.push_back("unknown");
            }
            
            if (record.contains("query_type")) {
                query_types.push_back(record["query_type"].get<std::string>());
            } else {
                query_types.push_back("unknown");
            }
            
            matched++;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing record " << parsed << ": " << e.what() << "\n";
            skipped++;
            continue;
        }
    }
    
    std::cerr << "Loaded " << matched << " training samples from " << parsed << " records (skipped " << skipped << ")\n";
    
    if (X.empty()) {
        std::cerr << "No valid training data found\n";
        return 1;
    }
    
    // Train linear readout with ridge regression
    std::vector<double> W2;
    double b2 = 0.0;
    solve_ridge(X, y, 1e-3, W2, b2);
    
    // Compute training MSE for reporting
    double mse = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred = b2;
        for (size_t j = 0; j < W2.size(); ++j) {
            pred += W2[j] * X[i][j];
        }
        double err = pred - y[i];
        mse += err * err;
    }
    mse /= X.size();
    std::cerr << "Training MSE: " << mse << "\n";
    std::cerr << "Training RMSE: " << std::sqrt(mse) << "\n";
    
    // Calculate routing accuracy and confusion matrix
    int true_positive = 0;  // Correctly predicted DuckDB is faster
    int true_negative = 0;  // Correctly predicted Postgres is faster
    int false_positive = 0; // Incorrectly predicted DuckDB is faster
    int false_negative = 0; // Incorrectly predicted Postgres is faster
    
    // Calculate execution times for different routing strategies
    double optimal_time = 0.0;
    double gnn_routing_time = 0.0;
    double always_postgres_time = 0.0;
    double always_duckdb_time = 0.0;
    double cost_threshold_100_time = 0.0;
    double cost_threshold_1000_time = 0.0;
    double cost_threshold_10000_time = 0.0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        // GNN prediction
        double pred = b2;
        for (size_t j = 0; j < W2.size(); ++j) {
            pred += W2[j] * X[i][j];
        }
        
        // Binary routing decision (positive means DuckDB is faster)
        bool actual_duckdb_faster = y[i] > 0;
        bool predicted_duckdb_faster = pred > 0;
        
        // Update confusion matrix
        if (actual_duckdb_faster && predicted_duckdb_faster) {
            true_positive++;
        } else if (!actual_duckdb_faster && !predicted_duckdb_faster) {
            true_negative++;
        } else if (!actual_duckdb_faster && predicted_duckdb_faster) {
            false_positive++;
        } else {
            false_negative++;
        }
        
        // Calculate execution times
        double pg_time = postgres_times[i];
        double duck_time = duckdb_times[i];
        
        // Optimal routing (always choose the faster one)
        optimal_time += std::min(pg_time, duck_time);
        
        // GNN-based routing
        gnn_routing_time += predicted_duckdb_faster ? duck_time : pg_time;
        
        // Always use one system
        always_postgres_time += pg_time;
        always_duckdb_time += duck_time;
        
        // Cost-threshold routing (using total_cost from features)
        // Assuming cost is stored in X[i][18] (after one-hot and basic features)
        double estimated_cost = (i < X.size() && X[i].size() > 18) ? std::exp(X[i][18]) : 1000.0;
        
        // Route based on cost thresholds
        cost_threshold_100_time += (estimated_cost < 100) ? duck_time : pg_time;
        cost_threshold_1000_time += (estimated_cost < 1000) ? duck_time : pg_time;
        cost_threshold_10000_time += (estimated_cost < 10000) ? duck_time : pg_time;
    }
    
    // Print confusion matrix
    std::cerr << "\n=== Confusion Matrix ===\n";
    std::cerr << "                 Predicted\n";
    std::cerr << "                 PG    DuckDB\n";
    std::cerr << "Actual PG      [" << std::setw(5) << true_negative << " " << std::setw(5) << false_positive << "]\n";
    std::cerr << "      DuckDB   [" << std::setw(5) << false_negative << " " << std::setw(5) << true_positive << "]\n";
    
    // Calculate metrics
    double accuracy = (double)(true_positive + true_negative) / X.size();
    double precision = (true_positive > 0) ? (double)true_positive / (true_positive + false_positive) : 0.0;
    double recall = (true_positive > 0) ? (double)true_positive / (true_positive + false_negative) : 0.0;
    double f1_score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
    
    std::cerr << "\n=== Classification Metrics ===\n";
    std::cerr << "Accuracy: " << std::fixed << std::setprecision(4) << accuracy << "\n";
    std::cerr << "Precision: " << precision << "\n";
    std::cerr << "Recall: " << recall << "\n";
    std::cerr << "F1 Score: " << f1_score << "\n";
    
    // Print routing strategy comparison
    std::cerr << "\n=== Routing Strategy Comparison ===\n";
    std::cerr << "Strategy               | Total Time (s) | Relative to Optimal\n";
    std::cerr << "-----------------------|----------------|-------------------\n";
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "Optimal                | " << std::setw(14) << optimal_time << " | 1.00x\n";
    std::cerr << "GNN Routing            | " << std::setw(14) << gnn_routing_time << " | " 
              << std::setprecision(2) << (gnn_routing_time / optimal_time) << "x\n";
    std::cerr << "Cost Threshold (100)   | " << std::setw(14) << cost_threshold_100_time << " | " 
              << (cost_threshold_100_time / optimal_time) << "x\n";
    std::cerr << "Cost Threshold (1000)  | " << std::setw(14) << cost_threshold_1000_time << " | " 
              << (cost_threshold_1000_time / optimal_time) << "x\n";
    std::cerr << "Cost Threshold (10000) | " << std::setw(14) << cost_threshold_10000_time << " | " 
              << (cost_threshold_10000_time / optimal_time) << "x\n";
    std::cerr << "Always PostgreSQL      | " << std::setw(14) << always_postgres_time << " | " 
              << (always_postgres_time / optimal_time) << "x\n";
    std::cerr << "Always DuckDB          | " << std::setw(14) << always_duckdb_time << " | " 
              << (always_duckdb_time / optimal_time) << "x\n";
    
    // Calculate and print routing overhead
    double gnn_overhead = gnn_routing_time - optimal_time;
    double overhead_percentage = (gnn_overhead / optimal_time) * 100;
    std::cerr << "\nGNN Routing Overhead: " << gnn_overhead << "s (" 
              << std::setprecision(1) << overhead_percentage << "%)\n";
    
    // Dataset-specific analysis
    std::unordered_map<std::string, int> dataset_correct;
    std::unordered_map<std::string, int> dataset_total;
    
    for (size_t i = 0; i < X.size(); ++i) {
        double pred = b2;
        for (size_t j = 0; j < W2.size(); ++j) {
            pred += W2[j] * X[i][j];
        }
        
        bool actual_duckdb_faster = y[i] > 0;
        bool predicted_duckdb_faster = pred > 0;
        bool correct = (actual_duckdb_faster == predicted_duckdb_faster);
        
        const std::string& dataset = datasets[i];
        dataset_total[dataset]++;
        if (correct) dataset_correct[dataset]++;
    }
    
    if (!dataset_total.empty()) {
        std::cerr << "\n=== Per-Dataset Accuracy ===\n";
        for (const auto& pair : dataset_total) {
            double dataset_accuracy = (double)dataset_correct[pair.first] / pair.second;
            std::cerr << pair.first << ": " << std::setprecision(4) << dataset_accuracy 
                      << " (" << dataset_correct[pair.first] << "/" << pair.second << ")\n";
        }
    }
    
    // Write model: W1=Identity, b1=0, W2 learned, b2 learned
    const int hidden = in_features;
    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "Failed to open output model file: " << out_path << "\n";
        return 1;
    }
    
    // Model format: in_features hidden_dim
    fout << in_features << " " << hidden << "\n";
    
    // W1 (identity matrix)
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < hidden; j++) {
            double val = (i == j) ? 1.0 : 0.0;
            fout << val << (j + 1 == hidden ? '\n' : ' ');
        }
    }
    
    // b1 (zeros)
    for (int j = 0; j < hidden; j++) {
        fout << 0.0 << (j + 1 == hidden ? '\n' : ' ');
    }
    
    // W2 (learned weights)
    for (int j = 0; j < hidden; j++) {
        fout << W2[j] << (j + 1 == hidden ? '\n' : ' ');
    }
    
    // b2 (learned bias)
    fout << b2 << "\n";
    fout.close();
    
    std::cerr << "GNN model saved to: " << out_path << "\n";
    std::cerr << "Model dimensions: input=" << in_features << ", hidden=" << hidden << "\n";
    
    return 0;
}