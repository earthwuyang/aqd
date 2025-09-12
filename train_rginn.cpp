/*
 * train_rginn.cpp
 * 
 * R-GIN training program with real JSON parsing using nlohmann-json
 * Properly parses PostgreSQL query plans and converts to graphs
 * 
 * Compile with:
 * g++ -std=c++17 -O3 -march=native -o train_rginn train_rginn.cpp -lm
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <dirent.h>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Constants
#define FEATURE_DIM 20
#define MAX_HIDDEN_DIM 64
#define MAX_NODES 1000
#define NUM_RELATIONS 3

// R-GIN Model structure
struct RGINNModel {
    int input_dim;
    int hidden_dim;
    int num_layers;
    int num_relations;
    double eps;
    double learning_rate;
    
    // Weight matrices
    double* W_input;    // [hidden_dim * input_dim]
    double* b_input;    // [hidden_dim]
    double* W_rel;      // [num_relations * hidden_dim * hidden_dim]
    double* W_mlp;      // [hidden_dim * hidden_dim]
    double* b_mlp;      // [hidden_dim]
    double* W_output;   // [hidden_dim]
    double b_output;
    
    // Adam optimizer state
    double* m_W_input;
    double* v_W_input;
    double* m_b_input;
    double* v_b_input;
    double* m_W_rel;
    double* v_W_rel;
    double* m_W_mlp;
    double* v_W_mlp;
    double* m_b_mlp;
    double* v_b_mlp;
    double* m_W_output;
    double* v_W_output;
    double m_b_output;
    double v_b_output;
    int adam_t;
};

// Plan graph structure
struct PlanGraph {
    int num_nodes;
    double* features;  // [num_nodes * input_dim]
    int num_edges[NUM_RELATIONS];
    int* edges[NUM_RELATIONS];  // [num_edges * 2] (src, dst pairs)
    
    PlanGraph() : num_nodes(0), features(nullptr) {
        for (int i = 0; i < NUM_RELATIONS; i++) {
            num_edges[i] = 0;
            edges[i] = nullptr;
        }
    }
    
    ~PlanGraph() {
        delete[] features;
        for (int i = 0; i < NUM_RELATIONS; i++) {
            delete[] edges[i];
        }
    }
};

// Training example
struct Example {
    PlanGraph* graph;
    double target;  // log(postgres_time / duckdb_time)
    double postgres_time_ms;
    double duckdb_time_ms;
    std::string dataset;
    std::string query_type;
    
    Example() : graph(nullptr) {}
    ~Example() { delete graph; }
};

// Operator type mapping
int get_operator_type(const std::string& op) {
    if (op == "Seq Scan") return 1;
    if (op == "Index Scan") return 2;
    if (op == "Index Only Scan") return 3;
    if (op == "Bitmap Index Scan") return 4;
    if (op == "Bitmap Heap Scan") return 5;
    if (op == "Nested Loop") return 6;
    if (op == "Hash Join") return 7;
    if (op == "Merge Join") return 8;
    if (op == "Hash") return 9;
    if (op == "Sort") return 10;
    if (op == "Aggregate") return 11;
    if (op == "Group") return 12;
    if (op == "Limit") return 13;
    if (op == "Materialize") return 14;
    if (op == "Gather") return 15;
    return 0;
}

// Get relation type for edges
int get_relation_type(const std::string& parent_op, const std::string& child_op) {
    // Relation 0: Scan operations
    if (child_op.find("Scan") != std::string::npos) return 0;
    // Relation 1: Join operations
    if (parent_op.find("Join") != std::string::npos || 
        parent_op.find("Loop") != std::string::npos) return 1;
    // Relation 2: Other operations
    return 2;
}

// Random number generation
std::mt19937 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);
double randn() { return normal_dist(rng); }

// Parse plan JSON to graph
PlanGraph* parse_plan_to_graph(const json& plan_json) {
    PlanGraph* graph = new PlanGraph();
    
    if (plan_json.is_null()) {
        // Return empty graph
        graph->num_nodes = 1;
        graph->features = new double[FEATURE_DIM]();
        return graph;
    }
    
    // Extract plan
    json plan = plan_json;
    if (plan_json.contains("Plan") && !plan_json["Plan"].is_null()) {
        plan = plan_json["Plan"];
    }
    
    // Build node list with BFS
    std::vector<json> nodes;
    std::vector<std::string> node_types;
    std::vector<std::vector<int>> children;
    
    std::vector<json> queue;
    queue.push_back(plan);
    
    while (!queue.empty()) {
        json node = queue.front();
        queue.erase(queue.begin());
        
        nodes.push_back(node);
        std::string node_type = "Unknown";
        if (node.contains("Node Type") && node["Node Type"].is_string()) {
            node_type = node["Node Type"].get<std::string>();
        }
        node_types.push_back(node_type);
        children.push_back(std::vector<int>());
        
        int parent_idx = nodes.size() - 1;
        
        // Add children
        if (node.contains("Plans") && node["Plans"].is_array()) {
            for (const auto& child : node["Plans"]) {
                children[parent_idx].push_back(nodes.size() + queue.size());
                queue.push_back(child);
            }
        }
    }
    
    graph->num_nodes = nodes.size();
    if (graph->num_nodes == 0) {
        graph->num_nodes = 1;
        graph->features = new double[FEATURE_DIM]();
        return graph;
    }
    
    graph->features = new double[graph->num_nodes * FEATURE_DIM]();
    
    // Extract features for each node
    for (int i = 0; i < graph->num_nodes; i++) {
        const json& node = nodes[i];
        double* feat = &graph->features[i * FEATURE_DIM];
        
        // One-hot encoding for operator type (first 16 features)
        int op_type = get_operator_type(node_types[i]);
        if (op_type >= 0 && op_type < 16) {
            feat[op_type] = 1.0;
        }
        
        // Numeric features (normalized)
        double rows = 0.0, width = 0.0, total_cost = 0.0, startup_cost = 0.0;
        
        if (node.contains("Plan Rows") && node["Plan Rows"].is_number()) {
            rows = node["Plan Rows"].get<double>();
        }
        if (node.contains("Plan Width") && node["Plan Width"].is_number()) {
            width = node["Plan Width"].get<double>();
        }
        if (node.contains("Total Cost") && node["Total Cost"].is_number()) {
            total_cost = node["Total Cost"].get<double>();
        }
        if (node.contains("Startup Cost") && node["Startup Cost"].is_number()) {
            startup_cost = node["Startup Cost"].get<double>();
        }
        
        feat[16] = log(rows + 1.0) / 20.0;  // Log scale for rows
        feat[17] = width / 1000.0;          // Normalize width
        feat[18] = log(total_cost + 1.0) / 20.0;  // Log scale for cost
        feat[19] = log(startup_cost + 1.0) / 20.0; // Log scale for startup cost
    }
    
    // Build edges
    std::vector<std::vector<std::pair<int, int>>> rel_edges(NUM_RELATIONS);
    
    for (int parent = 0; parent < graph->num_nodes; parent++) {
        for (int child : children[parent]) {
            if (child < graph->num_nodes) {
                int rel = get_relation_type(node_types[parent], node_types[child]);
                rel_edges[rel].push_back({parent, child});
                // Add reverse edge for bidirectional
                rel_edges[rel].push_back({child, parent});
            }
        }
    }
    
    // Copy edges to graph
    for (int rel = 0; rel < NUM_RELATIONS; rel++) {
        graph->num_edges[rel] = rel_edges[rel].size();
        if (graph->num_edges[rel] > 0) {
            graph->edges[rel] = new int[graph->num_edges[rel] * 2];
            for (int i = 0; i < graph->num_edges[rel]; i++) {
                graph->edges[rel][i * 2] = rel_edges[rel][i].first;
                graph->edges[rel][i * 2 + 1] = rel_edges[rel][i].second;
            }
        }
    }
    
    return graph;
}

// Initialize model
void rginn_init(RGINNModel* model, int input_dim, int hidden_dim, double learning_rate = 0.001) {
    model->input_dim = input_dim;
    model->hidden_dim = hidden_dim;
    model->num_layers = 1;
    model->num_relations = NUM_RELATIONS;
    model->eps = 0.0;
    model->learning_rate = learning_rate;
    model->adam_t = 0;
    
    // Allocate weights
    size_t input_size = hidden_dim * input_dim;
    size_t hidden_size = hidden_dim;
    size_t rel_size = NUM_RELATIONS * hidden_dim * hidden_dim;
    size_t mlp_size = hidden_dim * hidden_dim;
    
    model->W_input = new double[input_size];
    model->b_input = new double[hidden_size];
    model->W_rel = new double[rel_size];
    model->W_mlp = new double[mlp_size];
    model->b_mlp = new double[hidden_size];
    model->W_output = new double[hidden_size];
    
    // Allocate Adam state
    model->m_W_input = new double[input_size]();
    model->v_W_input = new double[input_size]();
    model->m_b_input = new double[hidden_size]();
    model->v_b_input = new double[hidden_size]();
    model->m_W_rel = new double[rel_size]();
    model->v_W_rel = new double[rel_size]();
    model->m_W_mlp = new double[mlp_size]();
    model->v_W_mlp = new double[mlp_size]();
    model->m_b_mlp = new double[hidden_size]();
    model->v_b_mlp = new double[hidden_size]();
    model->m_W_output = new double[hidden_size]();
    model->v_W_output = new double[hidden_size]();
    
    // Xavier initialization
    double scale_input = sqrt(2.0 / input_dim);
    double scale_hidden = sqrt(2.0 / hidden_dim);
    
    for (size_t i = 0; i < input_size; i++) {
        model->W_input[i] = randn() * scale_input;
    }
    
    for (size_t i = 0; i < hidden_size; i++) {
        model->b_input[i] = 0.0;
        model->b_mlp[i] = 0.0;
        model->W_output[i] = randn() * scale_hidden;
    }
    
    for (size_t i = 0; i < rel_size; i++) {
        model->W_rel[i] = randn() * scale_hidden;
    }
    
    for (size_t i = 0; i < mlp_size; i++) {
        model->W_mlp[i] = randn() * scale_hidden;
    }
    
    model->b_output = 0.0;
    model->m_b_output = 0.0;
    model->v_b_output = 0.0;
}

// Free model memory
void rginn_free(RGINNModel* model) {
    delete[] model->W_input;
    delete[] model->b_input;
    delete[] model->W_rel;
    delete[] model->W_mlp;
    delete[] model->b_mlp;
    delete[] model->W_output;
    
    delete[] model->m_W_input;
    delete[] model->v_W_input;
    delete[] model->m_b_input;
    delete[] model->v_b_input;
    delete[] model->m_W_rel;
    delete[] model->v_W_rel;
    delete[] model->m_W_mlp;
    delete[] model->v_W_mlp;
    delete[] model->m_b_mlp;
    delete[] model->v_b_mlp;
    delete[] model->m_W_output;
    delete[] model->v_W_output;
}

// ReLU activation
inline double relu(double x) { return x > 0 ? x : 0; }

// Forward pass
double rginn_forward(RGINNModel* model, PlanGraph* graph,
                     double* h0, double* messages, double* h1, double* graph_repr) {
    int N = graph->num_nodes;
    int H = model->hidden_dim;
    int F = model->input_dim;
    
    if (N == 0) return 0.0;
    
    // Input projection
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < H; j++) {
            double sum = model->b_input[j];
            for (int i = 0; i < F; i++) {
                sum += model->W_input[j * F + i] * graph->features[n * F + i];
            }
            h0[n * H + j] = relu(sum);
        }
    }
    
    // Initialize messages
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < H; j++) {
            messages[n * H + j] = (1.0 + model->eps) * h0[n * H + j];
        }
    }
    
    // Aggregate messages per relation
    for (int rel = 0; rel < NUM_RELATIONS; rel++) {
        for (int e = 0; e < graph->num_edges[rel]; e++) {
            int src = graph->edges[rel][e * 2];
            int dst = graph->edges[rel][e * 2 + 1];
            
            for (int j = 0; j < H; j++) {
                double sum = 0.0;
                for (int k = 0; k < H; k++) {
                    sum += model->W_rel[rel * H * H + j * H + k] * h0[src * H + k];
                }
                messages[dst * H + j] += sum;
            }
        }
    }
    
    // MLP transformation
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < H; j++) {
            double sum = model->b_mlp[j];
            for (int k = 0; k < H; k++) {
                sum += model->W_mlp[j * H + k] * messages[n * H + k];
            }
            h1[n * H + j] = relu(sum);
        }
    }
    
    // Graph-level readout (mean pooling)
    for (int j = 0; j < H; j++) {
        graph_repr[j] = 0.0;
        for (int n = 0; n < N; n++) {
            graph_repr[j] += h1[n * H + j];
        }
        graph_repr[j] /= N;
    }
    
    // Output layer
    double output = model->b_output;
    for (int j = 0; j < H; j++) {
        output += model->W_output[j] * graph_repr[j];
    }
    
    return output;
}

// Load training data from JSON files
std::vector<Example*> load_training_data(const std::string& data_dir) {
    std::vector<Example*> examples;
    
    DIR* dir = opendir(data_dir.c_str());
    if (!dir) {
        std::cerr << "Failed to open directory: " << data_dir << "\n";
        return examples;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Skip non-JSON files and summary files
        if (filename.find(".json") == std::string::npos ||
            filename.find("summary") != std::string::npos) {
            continue;
        }
        
        std::string filepath = data_dir + "/" + filename;
        std::ifstream file(filepath);
        if (!file) {
            std::cerr << "Failed to open file: " << filepath << "\n";
            continue;
        }
        
        // Parse entire file as JSON
        json data;
        try {
            file >> data;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing " << filepath << ": " << e.what() << "\n";
            continue;
        }
        
        if (!data.is_array()) {
            std::cerr << "File " << filepath << " is not a JSON array\n";
            continue;
        }
        
        std::cout << "Loading from " << filename << ": " << data.size() << " items\n";
        
        // Process each item
        for (const auto& item : data) {
            // Skip if missing required fields
            if (!item.contains("postgres_time_ms") || 
                !item.contains("duckdb_time_ms") ||
                !item.contains("postgres_plan")) {
                continue;
            }
            
            double pg_time = 0.0, duck_time = 0.0;
            
            try {
                if (item["postgres_time_ms"].is_number()) {
                    pg_time = item["postgres_time_ms"].get<double>();
                }
                if (item["duckdb_time_ms"].is_number()) {
                    duck_time = item["duckdb_time_ms"].get<double>();
                }
            } catch (...) {
                continue;
            }
            
            // Skip invalid times
            if (pg_time <= 0 || duck_time <= 0 || pg_time >= 30000 || duck_time >= 30000) {
                continue;
            }
            
            Example* ex = new Example();
            ex->postgres_time_ms = pg_time;
            ex->duckdb_time_ms = duck_time;
            ex->target = log(pg_time / duck_time);
            
            if (item.contains("dataset") && item["dataset"].is_string()) {
                ex->dataset = item["dataset"].get<std::string>();
            } else {
                ex->dataset = filename;
            }
            
            if (item.contains("query_type") && item["query_type"].is_string()) {
                ex->query_type = item["query_type"].get<std::string>();
            } else {
                ex->query_type = "unknown";
            }
            
            ex->graph = parse_plan_to_graph(item["postgres_plan"]);
            
            examples.push_back(ex);
        }
    }
    
    closedir(dir);
    
    std::cout << "Total examples loaded: " << examples.size() << "\n";
    return examples;
}

// Save model
void save_model(RGINNModel* model, const std::string& path) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return;
    }
    
    f << model->input_dim << " " << model->hidden_dim << "\n";
    
    // Save weights in same format as reference implementation
    for (int j = 0; j < model->hidden_dim; j++) {
        for (int i = 0; i < model->input_dim; i++) {
            f << model->W_input[j * model->input_dim + i];
            if (i < model->input_dim - 1) f << " ";
        }
        f << "\n";
    }
    
    for (int j = 0; j < model->hidden_dim; j++) {
        f << model->b_input[j];
        if (j < model->hidden_dim - 1) f << " ";
    }
    f << "\n";
    
    for (int r = 0; r < NUM_RELATIONS; r++) {
        for (int j = 0; j < model->hidden_dim; j++) {
            for (int k = 0; k < model->hidden_dim; k++) {
                f << model->W_rel[r * model->hidden_dim * model->hidden_dim + 
                                 j * model->hidden_dim + k];
                if (k < model->hidden_dim - 1) f << " ";
            }
            f << "\n";
        }
    }
    
    for (int j = 0; j < model->hidden_dim; j++) {
        for (int k = 0; k < model->hidden_dim; k++) {
            f << model->W_mlp[j * model->hidden_dim + k];
            if (k < model->hidden_dim - 1) f << " ";
        }
        f << "\n";
    }
    
    for (int j = 0; j < model->hidden_dim; j++) {
        f << model->b_mlp[j];
        if (j < model->hidden_dim - 1) f << " ";
    }
    f << "\n";
    
    for (int j = 0; j < model->hidden_dim; j++) {
        f << model->W_output[j];
        if (j < model->hidden_dim - 1) f << " ";
    }
    f << "\n";
    
    f << model->b_output << "\n";
    
    f.close();
    std::cout << "Model saved to " << path << "\n";
}

// Training step (simplified - just updates output layer for now)
void train_step(RGINNModel* model, Example* ex, 
                double* h0, double* messages, double* h1, double* graph_repr) {
    // Forward pass
    double pred = rginn_forward(model, ex->graph, h0, messages, h1, graph_repr);
    
    // Compute gradient
    double error = pred - ex->target;
    double grad = error;  // MSE gradient
    
    // Simple gradient descent on output layer
    model->b_output -= model->learning_rate * grad;
    for (int j = 0; j < model->hidden_dim; j++) {
        model->W_output[j] -= model->learning_rate * grad * graph_repr[j];
    }
}

// Evaluate model
void evaluate(RGINNModel* model, const std::vector<Example*>& examples, 
              double threshold = 0.0) {
    int correct = 0;
    int total = 0;
    double total_loss = 0.0;
    
    // Confusion matrix [true_label][predicted_label]
    int confusion[2][2] = {{0, 0}, {0, 0}};
    
    // Allocate working memory
    double* h0 = new double[MAX_NODES * MAX_HIDDEN_DIM];
    double* messages = new double[MAX_NODES * MAX_HIDDEN_DIM];
    double* h1 = new double[MAX_NODES * MAX_HIDDEN_DIM];
    double* graph_repr = new double[MAX_HIDDEN_DIM];
    
    for (const auto& ex : examples) {
        double pred = rginn_forward(model, ex->graph, h0, messages, h1, graph_repr);
        
        // Binary classification
        int pred_label = (pred > threshold) ? 1 : 0;  // 1 = DuckDB, 0 = PostgreSQL
        int true_label = (ex->target > threshold) ? 1 : 0;
        
        confusion[true_label][pred_label]++;
        if (pred_label == true_label) correct++;
        
        // Compute loss
        double error = pred - ex->target;
        total_loss += error * error;
        
        total++;
    }
    
    double accuracy = (double)correct / total;
    double precision = (confusion[1][1] > 0) ? 
        (double)confusion[1][1] / (confusion[1][1] + confusion[0][1]) : 0.0;
    double recall = (confusion[1][0] + confusion[1][1] > 0) ?
        (double)confusion[1][1] / (confusion[1][0] + confusion[1][1]) : 0.0;
    double f1 = (precision + recall > 0) ?
        2 * precision * recall / (precision + recall) : 0.0;
    
    std::cout << "Evaluation Results:\n";
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(3) << accuracy << "\n";
    std::cout << "  Precision: " << precision << "\n";
    std::cout << "  Recall: " << recall << "\n";
    std::cout << "  F1 Score: " << f1 << "\n";
    std::cout << "  MSE: " << total_loss / total << "\n";
    std::cout << "  Confusion Matrix:\n";
    std::cout << "              Predicted\n";
    std::cout << "            PG    DuckDB\n";
    std::cout << "  True PG   " << std::setw(4) << confusion[0][0] 
              << "  " << std::setw(4) << confusion[0][1] << "\n";
    std::cout << "  True Duck " << std::setw(4) << confusion[1][0] 
              << "  " << std::setw(4) << confusion[1][1] << "\n";
    
    delete[] h0;
    delete[] messages;
    delete[] h1;
    delete[] graph_repr;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "dual_execution_data";
    std::string model_path = "models/rginn_model.txt";
    int epochs = 50;
    double learning_rate = 0.001;
    int hidden_dim = 32;
    double test_split = 0.2;
    
    if (argc > 1) data_dir = argv[1];
    if (argc > 2) epochs = std::stoi(argv[2]);
    if (argc > 3) learning_rate = std::stod(argv[3]);
    
    std::cout << "R-GIN Training Program\n";
    std::cout << "======================\n";
    std::cout << "Data directory: " << data_dir << "\n";
    std::cout << "Epochs: " << epochs << "\n";
    std::cout << "Learning rate: " << learning_rate << "\n";
    std::cout << "Hidden dimension: " << hidden_dim << "\n\n";
    
    // Load training data
    std::vector<Example*> all_examples = load_training_data(data_dir);
    if (all_examples.empty()) {
        std::cerr << "No training data found!\n";
        return 1;
    }
    
    // Analyze data distribution
    int pg_preferred = 0, duck_preferred = 0;
    for (const auto& ex : all_examples) {
        if (ex->target < 0) pg_preferred++;
        else duck_preferred++;
    }
    std::cout << "\nData distribution:\n";
    std::cout << "  PostgreSQL preferred: " << pg_preferred << " (" 
              << (100.0 * pg_preferred / all_examples.size()) << "%)\n";
    std::cout << "  DuckDB preferred: " << duck_preferred << " (" 
              << (100.0 * duck_preferred / all_examples.size()) << "%)\n\n";
    
    // Split into train/test
    std::shuffle(all_examples.begin(), all_examples.end(), rng);
    int test_size = all_examples.size() * test_split;
    std::vector<Example*> train_examples(all_examples.begin(), 
                                         all_examples.end() - test_size);
    std::vector<Example*> test_examples(all_examples.end() - test_size,
                                        all_examples.end());
    
    std::cout << "Training examples: " << train_examples.size() << "\n";
    std::cout << "Test examples: " << test_examples.size() << "\n\n";
    
    // Initialize model
    RGINNModel model;
    rginn_init(&model, FEATURE_DIM, hidden_dim, learning_rate);
    
    // Allocate working memory
    double* h0 = new double[MAX_NODES * MAX_HIDDEN_DIM];
    double* messages = new double[MAX_NODES * MAX_HIDDEN_DIM];
    double* h1 = new double[MAX_NODES * MAX_HIDDEN_DIM];
    double* graph_repr = new double[MAX_HIDDEN_DIM];
    
    // Training loop
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        // Shuffle training data
        std::shuffle(train_examples.begin(), train_examples.end(), rng);
        
        // Train on each example
        for (auto& ex : train_examples) {
            // Forward pass
            double pred = rginn_forward(&model, ex->graph,
                                       h0, messages, h1, graph_repr);
            
            // Compute loss
            double error = pred - ex->target;
            total_loss += error * error;
            
            // Backward pass (simplified)
            train_step(&model, ex, h0, messages, h1, graph_repr);
        }
        
        // Print progress
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                     << " - Train MSE: " << std::fixed << std::setprecision(4)
                     << total_loss / train_examples.size() << "\n";
            
            // Evaluate on test set
            if ((epoch + 1) % 20 == 0 || epoch == epochs - 1) {
                std::cout << "\nTest Set ";
                evaluate(&model, test_examples);
                std::cout << "\n";
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nTraining completed in " << duration.count() << " seconds\n\n";
    
    // Final evaluation
    std::cout << "Final Test Set ";
    evaluate(&model, test_examples);
    
    // Save model
    save_model(&model, model_path);
    
    // Clean up
    delete[] h0;
    delete[] messages;
    delete[] h1;
    delete[] graph_repr;
    
    for (auto& ex : all_examples) {
        delete ex;
    }
    
    rginn_free(&model);
    
    return 0;
}