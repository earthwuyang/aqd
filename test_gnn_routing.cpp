#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <glob.h>
#include <nlohmann/json.hpp>
extern "C" {
#include "postgres_src/src/include/rginn_core.h"
}

using json = nlohmann::json;

static int op_type_id(const std::string& t){
    if (t=="Seq Scan") return 1; 
    if (t=="Index Scan") return 2; 
    if (t=="Bitmap Heap Scan") return 3;
    if (t=="Nested Loop") return 4; 
    if (t=="Merge Join") return 5; 
    if (t=="Hash Join") return 6;
    if (t=="Sort") return 7; 
    if (t=="Aggregate") return 8; 
    if (t=="Group") return 9; 
    return 0;
}

static int rel_id_for_child(const std::string& t){
    int id = op_type_id(t);
    if (id==1 || id==2 || id==3) return 0; // scans
    if (id==4 || id==5 || id==6) return 1; // joins
    return 2; // others
}

struct GraphBuild {
    std::vector<double> X;
    std::vector<int> indptr;
    std::vector<int> indices;
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
    gb.N = (int)nodes.size(); 
    gb.F = in_dim; 
    gb.R = 3; // scans, joins, others
    gb.X.assign((size_t)gb.N * gb.F, 0.0);
    gb.indptr.assign((size_t)gb.R * (gb.N + 1), 0);
    
    // adjacency per relation per node
    std::vector<std::vector<std::vector<int>>> adj(gb.R, std::vector<std::vector<int>>(gb.N));
    
    // features
    const int k = 10;
    for (int i=0;i<gb.N;++i){
        const json& n = nodes[i];
        std::string t = n.value("Node Type", "");
        int id = op_type_id(t);
        if (id>=0 && id<k) gb.X[i*gb.F + id] = 1.0;
        int idx = k;
        double rows = n.value("Plan Rows", 0.0);
        double width = n.value("Plan Width", 0.0);
        double tc = n.value("Total Cost", 0.0);
        // Normalize features
        if (idx<gb.F) gb.X[i*gb.F + idx++] = std::log(rows + 1.0) / 10.0;
        if (idx<gb.F) gb.X[i*gb.F + idx++] = width / 100.0;
        if (idx<gb.F) gb.X[i*gb.F + idx++] = std::log(tc + 1.0) / 10.0;
        // neighbors by relation
        for (int u : children[i]) {
            int r = rel_id_for_child(types[u]);
            if (r < 0 || r >= gb.R) r = 2;
            adj[r][i].push_back(u);
        }
    }
    
    // build CSR per relation
    gb.indices.clear();
    int base = 0;
    for (int r = 0; r < gb.R; ++r) {
        int *ind = &gb.indptr[(size_t)r * (gb.N + 1)];
        ind[0] = base;
        for (int i = 0; i < gb.N; ++i) {
            base += (int)adj[r][i].size();
            ind[i+1] = base;
            for (int dst : adj[r][i]) gb.indices.push_back(dst);
        }
    }
}

int main(int argc, char** argv) {
    std::string model_path = "models/rginn_routing_model.txt";
    std::string test_file = "data/execution_data/all_datasets_unified_training_data.json";
    
    if (argc > 1) test_file = argv[1];
    if (argc > 2) model_path = argv[2];
    
    std::cout << "\n=== GNN Routing Benchmark Test ===\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Test data: " << test_file << "\n\n";
    
    // Initialize GNN model
    if (!rginn_core_init(model_path.c_str())) {
        std::cerr << "ERROR: Failed to initialize GNN model from " << model_path << "\n";
        return 1;
    }
    std::cout << "GNN model loaded successfully.\n\n";
    
    // Load test data
    std::ifstream fin(test_file);
    if (!fin) {
        std::cerr << "ERROR: Failed to open " << test_file << "\n";
        rginn_core_cleanup();
        return 1;
    }
    
    json data;
    try {
        fin >> data;
    } catch(const std::exception& e) {
        std::cerr << "ERROR: Failed to parse JSON: " << e.what() << "\n";
        rginn_core_cleanup();
        return 1;
    }
    
    if (!data.is_array()) {
        std::cerr << "ERROR: Test file is not a JSON array\n";
        rginn_core_cleanup();
        return 1;
    }
    
    std::cout << "Processing " << data.size() << " queries...\n\n";
    
    // Test metrics
    int total_queries = 0;
    int successful_predictions = 0;
    int failed_predictions = 0;
    int correct_routing = 0;
    int incorrect_routing = 0;
    double total_prediction_time_ms = 0.0;
    
    // Process each query
    for (size_t i = 0; i < std::min((size_t)100, data.size()); ++i) {
        const auto& entry = data[i];
        
        if (!entry.contains("postgres_plan_json")) continue;
        const json& plan_array = entry["postgres_plan_json"];
        if (!plan_array.is_array() || plan_array.empty()) continue;
        
        const json& plan = plan_array[0].contains("Plan") ? plan_array[0]["Plan"] : plan_array[0];
        
        // Get ground truth if available
        double pg_time = entry.value("postgres_time", -1.0);
        double duck_time = entry.value("duckdb_time", -1.0);
        bool has_ground_truth = (pg_time > 0 && duck_time > 0);
        bool actual_duck_faster = has_ground_truth ? (duck_time < pg_time) : false;
        
        // Build graph from plan
        GraphBuild gb;
        build_graph_from_plan(plan, 16, gb);
        
        if (gb.N == 0) {
            std::cerr << "WARNING: Empty graph for query " << i << "\n";
            continue;
        }
        
        // Prepare graph structure for prediction
        RGGraph g;
        g.N = gb.N;
        g.in_dim = 16;
        g.X = gb.X.data();
        g.num_rel = gb.R;
        g.indptr = gb.indptr.data();
        g.indices = gb.indices.data();
        
        // Measure prediction time
        auto start = std::chrono::high_resolution_clock::now();
        
        // Make prediction
        double prediction = rginn_core_predict(&g);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double pred_time_ms = duration.count() / 1000.0;
        
        total_queries++;
        total_prediction_time_ms += pred_time_ms;
        
        if (std::isnan(prediction) || std::isinf(prediction)) {
            failed_predictions++;
            std::cerr << "ERROR: Invalid prediction for query " << i << ": " << prediction << "\n";
        } else {
            successful_predictions++;
            
            // Routing decision: positive means DuckDB is faster
            bool predict_duck_faster = (prediction > 0);
            
            if (has_ground_truth) {
                if (predict_duck_faster == actual_duck_faster) {
                    correct_routing++;
                } else {
                    incorrect_routing++;
                }
            }
            
            // Print sample predictions
            if (i < 10) {
                std::cout << "Query " << i << ": "
                          << "Prediction=" << std::fixed << std::setprecision(4) << prediction
                          << " (Route to " << (predict_duck_faster ? "DuckDB" : "PostgreSQL") << ")";
                if (has_ground_truth) {
                    std::cout << " | Actual: " << (actual_duck_faster ? "DuckDB" : "PostgreSQL")
                              << " faster (PG=" << pg_time << "s, Duck=" << duck_time << "s)";
                }
                std::cout << " | Time: " << std::setprecision(2) << pred_time_ms << "ms\n";
            }
        }
    }
    
    // Print summary
    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Total queries processed: " << total_queries << "\n";
    std::cout << "Successful predictions: " << successful_predictions << "\n";
    std::cout << "Failed predictions: " << failed_predictions << "\n";
    
    if (successful_predictions > 0) {
        std::cout << "Average prediction time: " << std::fixed << std::setprecision(3) 
                  << (total_prediction_time_ms / successful_predictions) << " ms\n";
    }
    
    if (correct_routing + incorrect_routing > 0) {
        double accuracy = (double)correct_routing / (correct_routing + incorrect_routing) * 100.0;
        std::cout << "\n=== Routing Accuracy ===\n";
        std::cout << "Correct routing decisions: " << correct_routing << "\n";
        std::cout << "Incorrect routing decisions: " << incorrect_routing << "\n";
        std::cout << "Accuracy: " << std::fixed << std::setprecision(1) << accuracy << "%\n";
    }
    
    // Cleanup
    rginn_core_cleanup();
    
    std::cout << "\n=== Test Completed Successfully ===\n";
    return 0;
}