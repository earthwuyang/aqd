#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <limits>
#include <glob.h>
#include <nlohmann/json.hpp>
extern "C" {
#include "rginn.h"
}

using json = nlohmann::json;

static int op_type_id(const std::string& t){
    if (t=="Seq Scan") return 1; if (t=="Index Scan") return 2; if (t=="Bitmap Heap Scan") return 3;
    if (t=="Nested Loop") return 4; if (t=="Merge Join") return 5; if (t=="Hash Join") return 6;
    if (t=="Sort") return 7; if (t=="Aggregate") return 8; if (t=="Group") return 9; return 0;
}

struct GraphBuild {
    std::vector<double> X; // N * F
    std::vector<int> indptr; // R*(N+1)
    std::vector<int> indices; // concatenated for all relations in order r0..rR-1
    int N=0; int F=0; int R=1;
};

static void build_graph_from_plan(const json& plan, int in_dim, GraphBuild& gb){
    // BFS nodes
    std::vector<json> nodes;
    std::vector<std::vector<int>> children;
    std::vector<std::string> types;
    nodes.push_back(plan);
    children.emplace_back();
    types.push_back(plan.value("Node Type",""));
    for (size_t i=0; i<nodes.size(); ++i){
        const json& n = nodes[i];
        if (n.contains("Plans") && n["Plans"].is_array()){
            for (const auto& ch : n["Plans"]) {
                int idx = (int)nodes.size();
                json child = ch.contains("Plan")?ch["Plan"]:ch;
                nodes.push_back(child);
                children.emplace_back();
                types.push_back(child.value("Node Type",""));
                children[i].push_back(idx);
            }
        }
    }
    gb.N = (int)nodes.size(); gb.F = in_dim; gb.R = 1; // fallback to single relation for stability
    gb.X.assign((size_t)gb.N * gb.F, 0.0);
    gb.indptr.assign((size_t)gb.R * (gb.N + 1), 0);
    std::vector<int> neigh;
    // features
    const int k = 10; int idx;
    for (int i=0;i<gb.N;++i){
        const json& n = nodes[i];
        std::string t = n.value("Node Type", "");
        int id = op_type_id(t);
        if (id>=0 && id<k) gb.X[i*gb.F + id] = 1.0;
        idx = k;
        double rows = n.value("Plan Rows", 0.0);
        double width = n.value("Plan Width", 0.0);
        double tc = n.value("Total Cost", 0.0);
        double cpr = rows>0? tc/rows : tc;
        // Normalize features
        if (idx<gb.F) gb.X[i*gb.F + idx++] = std::log(rows + 1.0) / 10.0;
        if (idx<gb.F) gb.X[i*gb.F + idx++] = width / 100.0;
        if (idx<gb.F) gb.X[i*gb.F + idx++] = std::log(tc + 1.0) / 10.0;
        // neighbors single relation
        gb.indptr[i+1] = gb.indptr[i] + (int)children[i].size();
        for (int u : children[i]) neigh.push_back(u);
    }
    gb.indices = std::move(neigh);
}

// Structure to hold training examples
struct Example {
    json plan_json;
    double target;
    double postgres_time;
    double duckdb_time;
    std::string dataset;
    std::string query_type;
};

// Load examples from a single JSON file
std::vector<Example> load_examples_from_file(const std::string& filepath) {
    std::vector<Example> examples;
    
    std::ifstream fin(filepath);
    if (!fin) {
        std::cerr << "  Warning: Failed to open " << filepath << "\n";
        return examples;
    }
    
    json arr;
    try {
        fin >> arr;
    } catch(...) {
        std::cerr << "  Warning: Parse error in " << filepath << "\n";
        return examples;
    }
    
    if (!arr.is_array()) {
        std::cerr << "  Warning: " << filepath << " is not a JSON array\n";
        return examples;
    }
    
    for (const auto& j : arr) {
        double target = 0.0;
        bool has_target = false;
        double pg_time = 0.0, duck_time = 0.0;
        
        if (j.contains("log_time_difference") && !j["log_time_difference"].is_null()) {
            target = j.value("log_time_difference", 0.0);
            has_target = true;
        } else if (j.contains("postgres_time") && j.contains("duckdb_time")) {
            pg_time = j.value("postgres_time", 0.0);
            duck_time = j.value("duckdb_time", 0.0);
            if (pg_time > 0 && duck_time > 0) {
                target = std::log(pg_time/duck_time);
                has_target = true;
            }
        }
        
        if (!has_target) continue;
        if (!j.contains("postgres_plan_json")) continue;
        
        const json& arrp = j["postgres_plan_json"];
        if (!arrp.is_array() || arrp.empty()) continue;
        
        const json& root = arrp[0].contains("Plan") ? arrp[0]["Plan"] : arrp[0];
        
        // Store times if available
        if (j.contains("postgres_time")) pg_time = j.value("postgres_time", 0.0);
        if (j.contains("duckdb_time")) duck_time = j.value("duckdb_time", 0.0);
        
        Example ex;
        ex.plan_json = root;
        ex.target = target;
        ex.postgres_time = pg_time;
        ex.duckdb_time = duck_time;
        ex.dataset = j.value("dataset", "unknown");
        ex.query_type = j.value("query_type", "unknown");
        
        examples.push_back(ex);
    }
    
    return examples;
}

// Default model path for kernel integration
const char* DEFAULT_RGINN_MODEL_PATH = "models/rginn_routing_model.txt";

int main(int argc, char** argv){
    std::string out_path;
    std::vector<std::string> data_files;
    
    // Parse command line arguments
    if (argc == 1) {
        // No arguments - use default model path and auto-load data
        out_path = DEFAULT_RGINN_MODEL_PATH;
        std::cout << "Using default model path: " << out_path << "\n";
        
        // Auto-load training data
        std::string pattern = "data/execution_data/*_unified_training_data.json";
        glob_t glob_result;
        if (glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result) == 0) {
            for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
                data_files.push_back(glob_result.gl_pathv[i]);
            }
            globfree(&glob_result);
        }
    } else if (argc == 2) {
        // Default: load all unified training data files from data/execution_data
        out_path = argv[1];
        std::string pattern = "data/execution_data/*_unified_training_data.json";
        
        glob_t glob_result;
        if (glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result) == 0) {
            for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
                data_files.push_back(glob_result.gl_pathv[i]);
            }
            globfree(&glob_result);
        }
        
        if (data_files.empty()) {
            std::cerr << "No unified training data files found in data/execution_data/\n";
            std::cerr << "Usage: " << argv[0] << " [out_model.txt] [data_file1.json data_file2.json ...]\n";
            std::cerr << "   or: " << argv[0] << "  (uses default: " << DEFAULT_RGINN_MODEL_PATH << ")\n";
            return 1;
        }
        
        std::cerr << "Auto-detected " << data_files.size() << " training data files:\n";
        for (const auto& f : data_files) {
            std::cerr << "  - " << f << "\n";
        }
        std::cerr << "\n";
        
    } else if (argc >= 3) {
        // Specific files provided
        out_path = argv[2];
        data_files.push_back(argv[1]);
        
        // Check if this is the old format (single input file)
        if (argc == 3 && data_files[0].find(".json") != std::string::npos) {
            // Old format: input.json output.txt
            out_path = argv[2];
        } else {
            // New format: output.txt file1.json file2.json ...
            out_path = argv[1];
            data_files.clear();
            for (int i = 2; i < argc; ++i) {
                data_files.push_back(argv[i]);
            }
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " [out_model.txt] [data_file1.json data_file2.json ...]\n";
        std::cerr << "   or: " << argv[0] << " [out_model.txt]  (auto-loads all *_unified_training_data.json)\n";
        std::cerr << "   or: " << argv[0] << "  (uses default: " << DEFAULT_RGINN_MODEL_PATH << ")\n";
        return 1;
    }

    const int in_dim = 16; 
    const int hidden = 32;  // Increased hidden dimension
    const int num_rel = 1;
    const int max_epochs = 200;  // Maximum epochs with early stopping
    const double learning_rate = 1e-3;
    const int patience = 20;  // Early stopping patience
    
    RGINNModel model; 
    rginn_init(&model, in_dim, hidden, 1, num_rel, learning_rate);

    std::cerr << "\n=== R-GINN Training Configuration ===\n";
    std::cerr << "Input dimension: " << in_dim << "\n";
    std::cerr << "Hidden dimension: " << hidden << "\n";
    std::cerr << "Number of relations: " << num_rel << "\n";
    std::cerr << "Learning rate: " << learning_rate << "\n";
    std::cerr << "Max epochs: " << max_epochs << "\n";
    std::cerr << "Early stopping patience: " << patience << "\n";
    std::cerr << "Output model: " << out_path << "\n\n";

    // Load all examples from all files
    std::vector<Example> all_examples;
    std::unordered_map<std::string, int> dataset_counts;
    
    std::cerr << "Loading data from " << data_files.size() << " file(s)...\n";
    
    for (const auto& filepath : data_files) {
        std::cerr << "Loading: " << filepath << "\n";
        auto examples = load_examples_from_file(filepath);
        
        // Track dataset statistics
        for (const auto& ex : examples) {
            dataset_counts[ex.dataset]++;
        }
        
        std::cerr << "  Loaded " << examples.size() << " examples\n";
        all_examples.insert(all_examples.end(), examples.begin(), examples.end());
    }
    
    if (all_examples.empty()) {
        std::cerr << "\nNo valid training examples found!\n";
        return 1;
    }
    
    std::cerr << "\nTotal examples loaded: " << all_examples.size() << "\n";
    std::cerr << "Dataset distribution:\n";
    for (const auto& pair : dataset_counts) {
        std::cerr << "  " << pair.first << ": " << pair.second << " examples\n";
    }
    std::cerr << "\n";
    
    // Shuffle examples for better training
    std::random_shuffle(all_examples.begin(), all_examples.end());
    
    // Split into train/test (80/20)
    size_t train_size = all_examples.size() * 0.8;
    size_t test_size = all_examples.size() - train_size;
    
    std::cerr << "Training set: " << train_size << " examples\n";
    std::cerr << "Test set: " << test_size << " examples\n\n";
    
    // Training loop with early stopping
    std::vector<double> train_losses;
    std::vector<double> val_losses;
    double best_val_loss = std::numeric_limits<double>::max();
    int epochs_without_improvement = 0;
    int best_epoch = 0;
    
    // Save best model state
    RGINNModel best_model = model;
    
    for (int ep = 0; ep < max_epochs; ++ep) {
        double total_train_loss = 0.0;
        int train_batch_count = 0;
        
        // Train on training set
        for (size_t idx = 0; idx < train_size; ++idx) {
            const auto& ex = all_examples[idx];
            
            GraphBuild gb; 
            build_graph_from_plan(ex.plan_json, in_dim, gb);
            if (gb.N == 0) continue;
            
            RGGraph g; 
            g.N = gb.N; 
            g.in_dim = in_dim; 
            g.X = gb.X.data(); 
            g.num_rel = gb.R;
            
            std::vector<int> ind((size_t)g.num_rel * (g.N + 1));
            int base = 0;
            for (int r = 0; r < g.num_rel; ++r){
                int total_r = gb.indptr[(size_t)r * (g.N + 1) + g.N];
                for (int i = 0; i <= g.N; ++i) 
                    ind[(size_t)r * (g.N + 1) + i] = base + gb.indptr[(size_t)r * (g.N + 1) + i];
                base += total_r;
            }
            g.indptr = ind.data(); 
            g.indices = gb.indices.data();
            
            std::vector<double> h0((size_t)g.N * hidden), m1((size_t)g.N * hidden), 
                               h1((size_t)g.N * hidden), gr(hidden);
            
            double y = rginn_forward(&model, &g, h0.data(), m1.data(), h1.data(), gr.data());
            double loss = (y - ex.target) * (y - ex.target);
            total_train_loss += loss;
            
            rginn_backward_update(&model, &g, h0.data(), m1.data(), h1.data(), gr.data(), y, ex.target);
            train_batch_count++;
        }
        
        double avg_train_loss = total_train_loss / train_batch_count;
        train_losses.push_back(avg_train_loss);
        
        // Compute validation loss (on a subset for efficiency)
        double total_val_loss = 0.0;
        int val_batch_count = 0;
        size_t val_subset_size = std::min((size_t)2000, test_size);  // Use subset for validation
        
        for (size_t idx = train_size; idx < train_size + val_subset_size; ++idx) {
            const auto& ex = all_examples[idx];
            
            GraphBuild gb; 
            build_graph_from_plan(ex.plan_json, in_dim, gb);
            if (gb.N == 0) continue;
            
            RGGraph g; 
            g.N = gb.N; 
            g.in_dim = in_dim; 
            g.X = gb.X.data(); 
            g.num_rel = gb.R;
            
            std::vector<int> ind((size_t)g.num_rel * (g.N + 1));
            int base = 0;
            for (int r = 0; r < g.num_rel; ++r){
                int total_r = gb.indptr[(size_t)r * (g.N + 1) + g.N];
                for (int i = 0; i <= g.N; ++i) 
                    ind[(size_t)r * (g.N + 1) + i] = base + gb.indptr[(size_t)r * (g.N + 1) + i];
                base += total_r;
            }
            g.indptr = ind.data(); 
            g.indices = gb.indices.data();
            
            std::vector<double> h0((size_t)g.N * hidden), m1((size_t)g.N * hidden), 
                               h1((size_t)g.N * hidden), gr(hidden);
            
            double y = rginn_forward(&model, &g, h0.data(), m1.data(), h1.data(), gr.data());
            double loss = (y - ex.target) * (y - ex.target);
            total_val_loss += loss;
            val_batch_count++;
        }
        
        double avg_val_loss = total_val_loss / val_batch_count;
        val_losses.push_back(avg_val_loss);
        
        // Check for improvement
        if (avg_val_loss < best_val_loss) {
            best_val_loss = avg_val_loss;
            best_epoch = ep;
            epochs_without_improvement = 0;
            best_model = model;  // Save best model
        } else {
            epochs_without_improvement++;
        }
        
        // Print progress
        if (ep % 1 == 0 || ep == max_epochs - 1 || epochs_without_improvement >= patience) {
            std::cerr << "Epoch " << std::setw(3) << ep 
                     << ": Train Loss = " << std::fixed << std::setprecision(6) << avg_train_loss 
                     << " (RMSE: " << std::setprecision(4) << std::sqrt(avg_train_loss) << ")"
                     << ", Val Loss = " << std::setprecision(6) << avg_val_loss
                     << " (RMSE: " << std::setprecision(4) << std::sqrt(avg_val_loss) << ")";
            if (ep == best_epoch) {
                std::cerr << " *";
            }
            std::cerr << "\n";
        }
        
        // Early stopping check
        if (epochs_without_improvement >= patience) {
            std::cerr << "\nEarly stopping triggered after " << ep + 1 << " epochs.\n";
            std::cerr << "Best validation loss: " << best_val_loss 
                     << " (RMSE: " << std::sqrt(best_val_loss) << ") at epoch " << best_epoch << "\n";
            model = best_model;  // Restore best model
            break;
        }
    }
    
    // Evaluation on test set
    std::cerr << "\n=== Evaluation on Test Set ===\n";
    
    int true_positive = 0, true_negative = 0;
    int false_positive = 0, false_negative = 0;
    double test_mse = 0.0;
    int test_count = 0;
    
    // For routing analysis
    double optimal_time = 0.0;
    double gnn_routing_time = 0.0;
    double always_postgres_time = 0.0;
    double always_duckdb_time = 0.0;
    
    // Cost threshold analysis
    double cost_threshold_100_time = 0.0;
    double cost_threshold_1000_time = 0.0;
    double cost_threshold_10000_time = 0.0;
    
    // Per-dataset accuracy
    std::unordered_map<std::string, int> dataset_correct;
    std::unordered_map<std::string, int> dataset_total;
    
    for (size_t idx = train_size; idx < all_examples.size(); ++idx) {
        const auto& ex = all_examples[idx];
        
        GraphBuild gb; 
        build_graph_from_plan(ex.plan_json, in_dim, gb);
        if (gb.N == 0) continue;
        
        RGGraph g; 
        g.N = gb.N; 
        g.in_dim = in_dim; 
        g.X = gb.X.data(); 
        g.num_rel = gb.R;
        
        std::vector<int> ind((size_t)g.num_rel * (g.N + 1));
        int base = 0;
        for (int r = 0; r < g.num_rel; ++r){
            int total_r = gb.indptr[(size_t)r * (g.N + 1) + g.N];
            for (int i = 0; i <= g.N; ++i) 
                ind[(size_t)r * (g.N + 1) + i] = base + gb.indptr[(size_t)r * (g.N + 1) + i];
            base += total_r;
        }
        g.indptr = ind.data(); 
        g.indices = gb.indices.data();
        
        std::vector<double> h0((size_t)g.N * hidden), m1((size_t)g.N * hidden), 
                           h1((size_t)g.N * hidden), gr(hidden);
        
        double prediction = rginn_forward(&model, &g, h0.data(), m1.data(), h1.data(), gr.data());
        
        double error = prediction - ex.target;
        test_mse += error * error;
        test_count++;
        
        // Binary classification for routing
        bool actual_duckdb_faster = ex.target > 0;
        bool predicted_duckdb_faster = prediction > 0;
        
        if (actual_duckdb_faster && predicted_duckdb_faster) true_positive++;
        else if (!actual_duckdb_faster && !predicted_duckdb_faster) true_negative++;
        else if (!actual_duckdb_faster && predicted_duckdb_faster) false_positive++;
        else false_negative++;
        
        // Track per-dataset accuracy
        bool correct = (actual_duckdb_faster == predicted_duckdb_faster);
        dataset_total[ex.dataset]++;
        if (correct) dataset_correct[ex.dataset]++;
        
        // Routing time analysis
        if (ex.postgres_time > 0 && ex.duckdb_time > 0) {
            optimal_time += std::min(ex.postgres_time, ex.duckdb_time);
            gnn_routing_time += predicted_duckdb_faster ? ex.duckdb_time : ex.postgres_time;
            always_postgres_time += ex.postgres_time;
            always_duckdb_time += ex.duckdb_time;
            
            // Estimate cost-based routing (using total cost from plan)
            // Extract the root node's total cost
            double total_cost = ex.plan_json.value("Total Cost", 1000.0);
            
            // Cost threshold routing decisions
            cost_threshold_100_time += (total_cost < 100) ? ex.duckdb_time : ex.postgres_time;
            cost_threshold_1000_time += (total_cost < 1000) ? ex.duckdb_time : ex.postgres_time;
            cost_threshold_10000_time += (total_cost < 10000) ? ex.duckdb_time : ex.postgres_time;
        }
    }
    
    // Calculate metrics
    double test_rmse = std::sqrt(test_mse / test_count);
    int total = true_positive + true_negative + false_positive + false_negative;
    double accuracy = (double)(true_positive + true_negative) / total * 100.0;
    double precision = (true_positive > 0) ? 
        (double)true_positive / (true_positive + false_positive) : 0.0;
    double recall = (true_positive > 0) ? 
        (double)true_positive / (true_positive + false_negative) : 0.0;
    double f1_score = (precision + recall > 0) ? 
        2 * precision * recall / (precision + recall) : 0.0;
    
    // Print results
    std::cerr << "\n=== Confusion Matrix ===\n";
    std::cerr << "                 Predicted\n";
    std::cerr << "                 PG    DuckDB\n";
    std::cerr << "Actual PG      [" << std::setw(5) << true_negative 
             << " " << std::setw(5) << false_positive << "]\n";
    std::cerr << "      DuckDB   [" << std::setw(5) << false_negative 
             << " " << std::setw(5) << true_positive << "]\n";
    
    std::cerr << "\n=== Classification Metrics ===\n";
    std::cerr << "Test RMSE: " << std::fixed << std::setprecision(4) << test_rmse << "\n";
    std::cerr << "Accuracy: " << accuracy << "%\n";
    std::cerr << "Precision: " << precision << "\n";
    std::cerr << "Recall: " << recall << "\n";
    std::cerr << "F1 Score: " << f1_score << "\n";
    
    std::cerr << "\n=== Routing Strategy Comparison ===\n";
    std::cerr << "Strategy               | Total Time (s) | Relative to Optimal\n";
    std::cerr << "-----------------------|----------------|-------------------\n";
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "Optimal                | " << std::setw(14) << optimal_time << " | 1.00x\n";
    if (optimal_time > 0) {
        std::cerr << "GNN Routing            | " << std::setw(14) << gnn_routing_time 
                 << " | " << (gnn_routing_time / optimal_time) << "x\n";
        std::cerr << "Cost Threshold (100)   | " << std::setw(14) << cost_threshold_100_time
                 << " | " << (cost_threshold_100_time / optimal_time) << "x\n";
        std::cerr << "Cost Threshold (1000)  | " << std::setw(14) << cost_threshold_1000_time
                 << " | " << (cost_threshold_1000_time / optimal_time) << "x\n";
        std::cerr << "Cost Threshold (10000) | " << std::setw(14) << cost_threshold_10000_time
                 << " | " << (cost_threshold_10000_time / optimal_time) << "x\n";
        std::cerr << "Always PostgreSQL      | " << std::setw(14) << always_postgres_time 
                 << " | " << (always_postgres_time / optimal_time) << "x\n";
        std::cerr << "Always DuckDB          | " << std::setw(14) << always_duckdb_time 
                 << " | " << (always_duckdb_time / optimal_time) << "x\n";
        
        double overhead = gnn_routing_time - optimal_time;
        double overhead_pct = (overhead / optimal_time) * 100;
        std::cerr << "\nGNN Routing Overhead: " << overhead << "s (" 
                 << std::setprecision(1) << overhead_pct << "%)\n";
    }
    
    if (!dataset_total.empty()) {
        std::cerr << "\n=== Per-Dataset Accuracy ===\n";
        for (const auto& pair : dataset_total) {
            double ds_accuracy = (double)dataset_correct[pair.first] / pair.second * 100.0;
            std::cerr << pair.first << ": " << std::setprecision(1) << ds_accuracy 
                     << "% (" << dataset_correct[pair.first] << "/" << pair.second << ")\n";
        }
    }
    
    // Save model
    if (rginn_save(&model, out_path.c_str()) != 0) { 
        std::cerr << "\nFailed to save model.\n"; 
        return 1; 
    }
    
    std::cerr << "\n=== Model Saved ===\n";
    std::cerr << "Trained R-GINN on " << train_size << " examples\n";
    std::cerr << "Model saved to: " << out_path << "\n";
    std::cerr << "Final test accuracy: " << std::setprecision(1) << accuracy << "%\n";
    
    return 0;
}