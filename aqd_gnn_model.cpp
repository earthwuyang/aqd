/*
 * aqd_gnn_model.cpp
 *
 * Graph Neural Network implementation for PostgreSQL query plan analysis
 * Captures graph structure of query execution plans for routing decisions
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Graph representation of PostgreSQL query plan
struct PlanNode {
    std::string node_type;          // Node type (SeqScan, HashJoin, etc.)
    double startup_cost;
    double total_cost;
    double plan_rows;
    int plan_width;
    std::vector<int> children;      // Indices of child nodes
    std::unordered_map<std::string, double> features;
};

struct QueryPlanGraph {
    std::vector<PlanNode> nodes;
    int root_node;
    std::string query_hash;
    
    // Graph statistics
    int max_depth;
    int num_joins;
    int num_scans;
    int num_aggregates;
    double total_estimated_cost;
};

class AQDGraphNeuralNetwork {
private:
    struct GNNLayer {
        std::vector<std::vector<double>> weights;     // Weight matrix
        std::vector<double> bias;                     // Bias vector
        std::string activation;                       // Activation function
        int input_dim;
        int output_dim;
        
        GNNLayer(int in_dim, int out_dim, const std::string& act = "relu") 
            : input_dim(in_dim), output_dim(out_dim), activation(act) {
            // Initialize weights with Xavier initialization
            double scale = std::sqrt(6.0 / (in_dim + out_dim));
            weights.resize(out_dim, std::vector<double>(in_dim));
            bias.resize(out_dim);
            
            for (int i = 0; i < out_dim; i++) {
                for (int j = 0; j < in_dim; j++) {
                    weights[i][j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
                }
                bias[i] = 0.0;
            }
        }
    };
    
    struct GraphConvolutionLayer {
        GNNLayer message_passing;
        GNNLayer node_update;
        GNNLayer edge_update;
        
        GraphConvolutionLayer(int node_dim, int hidden_dim) 
            : message_passing(node_dim * 2, hidden_dim, "relu"),
              node_update(hidden_dim, node_dim, "relu"),
              edge_update(node_dim * 2, 1, "sigmoid") {}
    };
    
    // Network architecture
    std::vector<GraphConvolutionLayer> conv_layers;
    std::vector<GNNLayer> fc_layers;
    
    // Model hyperparameters
    int node_feature_dim = 16;      // Dimension of node features
    int hidden_dim = 64;            // Hidden layer dimension
    int num_conv_layers = 3;        // Number of graph convolution layers
    int num_fc_layers = 2;          // Number of fully connected layers
    
    // Training parameters
    double learning_rate = 0.001;
    int max_epochs = 1000;
    double early_stopping_patience = 50;
    
public:
    AQDGraphNeuralNetwork() {
        // Initialize graph convolution layers
        for (int i = 0; i < num_conv_layers; i++) {
            conv_layers.emplace_back(node_feature_dim, hidden_dim);
        }
        
        // Initialize fully connected layers for final prediction
        fc_layers.emplace_back(node_feature_dim, hidden_dim, "relu");
        fc_layers.emplace_back(hidden_dim, hidden_dim / 2, "relu");
        fc_layers.emplace_back(hidden_dim / 2, 1, "linear"); // Output: routing decision
    }
    
    QueryPlanGraph parse_postgres_plan(const json& explain_json) {
        QueryPlanGraph graph;
        
        if (!explain_json.is_array() || explain_json.empty()) {
            return graph;
        }
        
        const json& plan_json = explain_json[0];
        if (!plan_json.contains("Plan")) {
            return graph;
        }
        
        // Parse the plan tree recursively
        graph.root_node = 0;
        parse_plan_node(plan_json["Plan"], graph, -1);
        
        // Calculate graph statistics
        calculate_graph_statistics(graph);
        
        return graph;
    }
    
    std::vector<double> extract_node_features(const PlanNode& node, const QueryPlanGraph& graph) {
        std::vector<double> features(node_feature_dim, 0.0);
        
        // Node type encoding (one-hot-like)
        std::unordered_map<std::string, int> node_type_map = {
            {"SeqScan", 0}, {"IndexScan", 1}, {"BitmapHeapScan", 2},
            {"NestLoop", 3}, {"HashJoin", 4}, {"MergeJoin", 5},
            {"Hash", 6}, {"Sort", 7}, {"Agg", 8}, {"WindowAgg", 9},
            {"Limit", 10}, {"Subquery", 11}, {"Gather", 12}
        };
        
        int node_type_idx = node_type_map.count(node.node_type) ? 
                           node_type_map[node.node_type] : 13;
        if (node_type_idx < node_feature_dim - 3) {
            features[node_type_idx] = 1.0;
        }
        
        // Normalized cost features
        int cost_start_idx = std::min(14, node_feature_dim - 2);
        if (cost_start_idx < node_feature_dim) {
            features[cost_start_idx] = std::log(1.0 + node.total_cost) / 10.0;
        }
        if (cost_start_idx + 1 < node_feature_dim) {
            features[cost_start_idx + 1] = std::log(1.0 + node.plan_rows) / 10.0;
        }
        
        return features;
    }
    
    double predict_routing(const QueryPlanGraph& graph) {
        if (graph.nodes.empty()) {
            return 0.5; // Default to balanced routing
        }
        
        // Extract node features for all nodes
        std::vector<std::vector<double>> node_features;
        for (const auto& node : graph.nodes) {
            node_features.push_back(extract_node_features(node, graph));
        }
        
        // Apply graph convolution layers
        std::vector<std::vector<double>> current_features = node_features;
        
        for (const auto& conv_layer : conv_layers) {
            current_features = apply_graph_convolution(current_features, graph, conv_layer);
        }
        
        // Graph-level pooling (mean pooling)
        std::vector<double> graph_embedding(node_feature_dim, 0.0);
        for (const auto& node_feat : current_features) {
            for (int i = 0; i < node_feature_dim && i < node_feat.size(); i++) {
                graph_embedding[i] += node_feat[i];
            }
        }
        
        if (!current_features.empty()) {
            for (double& val : graph_embedding) {
                val /= current_features.size();
            }
        }
        
        // Apply fully connected layers
        std::vector<double> fc_input = graph_embedding;
        for (const auto& fc_layer : fc_layers) {
            fc_input = apply_linear_layer(fc_input, fc_layer);
        }
        
        // Return routing probability (sigmoid of output)
        double raw_output = fc_input.empty() ? 0.0 : fc_input[0];
        return 1.0 / (1.0 + std::exp(-raw_output)); // Sigmoid
    }
    
    void train_model(const std::vector<std::pair<QueryPlanGraph, double>>& training_data) {
        std::cout << "Training GNN model on " << training_data.size() << " samples..." << std::endl;
        
        double best_loss = std::numeric_limits<double>::max();
        int patience_counter = 0;
        
        for (int epoch = 0; epoch < max_epochs; epoch++) {
            double total_loss = 0.0;
            int num_samples = 0;
            
            // Training loop
            for (const auto& sample : training_data) {
                const QueryPlanGraph& graph = sample.first;
                double target = sample.second;
                
                // Forward pass
                double prediction = predict_routing(graph);
                
                // Calculate loss (mean squared error)
                double loss = (prediction - target) * (prediction - target);
                total_loss += loss;
                num_samples++;
                
                // Backward pass (simplified gradient descent)
                double gradient = 2.0 * (prediction - target);
                update_weights(gradient);
            }
            
            double avg_loss = total_loss / std::max(num_samples, 1);
            
            // Early stopping
            if (avg_loss < best_loss) {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= early_stopping_patience) {
                    std::cout << "Early stopping at epoch " << epoch << std::endl;
                    break;
                }
            }
            
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << avg_loss << std::endl;
            }
        }
        
        std::cout << "Training completed. Best loss: " << best_loss << std::endl;
    }
    
    bool save_model(const std::string& filepath) {
        json model_json;
        
        // Save architecture parameters
        model_json["node_feature_dim"] = node_feature_dim;
        model_json["hidden_dim"] = hidden_dim;
        model_json["num_conv_layers"] = num_conv_layers;
        model_json["num_fc_layers"] = num_fc_layers;
        
        // Save weights for fully connected layers
        model_json["fc_layers"] = json::array();
        for (int i = 0; i < fc_layers.size(); i++) {
            json layer_json;
            layer_json["weights"] = fc_layers[i].weights;
            layer_json["bias"] = fc_layers[i].bias;
            layer_json["activation"] = fc_layers[i].activation;
            layer_json["input_dim"] = fc_layers[i].input_dim;
            layer_json["output_dim"] = fc_layers[i].output_dim;
            model_json["fc_layers"].push_back(layer_json);
        }
        
        // Save conv layers (simplified)
        model_json["conv_layers_info"] = json::object();
        model_json["conv_layers_info"]["count"] = conv_layers.size();
        
        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot save model to " << filepath << std::endl;
            return false;
        }
        
        file << model_json.dump(2) << std::endl;
        std::cout << "GNN model saved to: " << filepath << std::endl;
        
        return true;
    }
    
    bool load_model(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot load model from " << filepath << std::endl;
            return false;
        }
        
        json model_json;
        try {
            file >> model_json;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing model file: " << e.what() << std::endl;
            return false;
        }
        
        // Load architecture parameters
        node_feature_dim = model_json.value("node_feature_dim", 16);
        hidden_dim = model_json.value("hidden_dim", 64);
        num_conv_layers = model_json.value("num_conv_layers", 3);
        num_fc_layers = model_json.value("num_fc_layers", 2);
        
        // Load fully connected layers
        if (model_json.contains("fc_layers")) {
            fc_layers.clear();
            for (const auto& layer_json : model_json["fc_layers"]) {
                int in_dim = layer_json["input_dim"];
                int out_dim = layer_json["output_dim"];
                std::string activation = layer_json["activation"];
                
                GNNLayer layer(in_dim, out_dim, activation);
                layer.weights = layer_json["weights"];
                layer.bias = layer_json["bias"];
                
                fc_layers.push_back(layer);
            }
        }
        
        std::cout << "GNN model loaded from: " << filepath << std::endl;
        return true;
    }

private:
    void parse_plan_node(const json& node_json, QueryPlanGraph& graph, int parent_idx) {
        PlanNode node;
        
        // Extract basic node information
        node.node_type = node_json.value("Node Type", "Unknown");
        node.startup_cost = node_json.value("Startup Cost", 0.0);
        node.total_cost = node_json.value("Total Cost", 0.0);
        node.plan_rows = node_json.value("Plan Rows", 0.0);
        node.plan_width = node_json.value("Plan Width", 0);
        
        // Add additional features
        if (node_json.contains("Actual Time")) {
            node.features["actual_time"] = node_json["Actual Time"];
        }
        if (node_json.contains("Actual Rows")) {
            node.features["actual_rows"] = node_json["Actual Rows"];
        }
        
        int current_idx = graph.nodes.size();
        graph.nodes.push_back(node);
        
        // Update parent's children list
        if (parent_idx >= 0 && parent_idx < graph.nodes.size()) {
            graph.nodes[parent_idx].children.push_back(current_idx);
        }
        
        // Process child nodes
        if (node_json.contains("Plans")) {
            for (const auto& child_json : node_json["Plans"]) {
                parse_plan_node(child_json, graph, current_idx);
            }
        }
    }
    
    void calculate_graph_statistics(QueryPlanGraph& graph) {
        graph.max_depth = 0;
        graph.num_joins = 0;
        graph.num_scans = 0;
        graph.num_aggregates = 0;
        graph.total_estimated_cost = 0.0;
        
        for (const auto& node : graph.nodes) {
            graph.total_estimated_cost += node.total_cost;
            
            if (node.node_type.find("Join") != std::string::npos) {
                graph.num_joins++;
            } else if (node.node_type.find("Scan") != std::string::npos) {
                graph.num_scans++;
            } else if (node.node_type.find("Agg") != std::string::npos) {
                graph.num_aggregates++;
            }
        }
        
        // Calculate depth using DFS
        if (!graph.nodes.empty()) {
            graph.max_depth = calculate_depth(graph, graph.root_node, 0);
        }
    }
    
    int calculate_depth(const QueryPlanGraph& graph, int node_idx, int current_depth) {
        if (node_idx >= graph.nodes.size()) {
            return current_depth;
        }
        
        int max_child_depth = current_depth;
        const PlanNode& node = graph.nodes[node_idx];
        
        for (int child_idx : node.children) {
            int child_depth = calculate_depth(graph, child_idx, current_depth + 1);
            max_child_depth = std::max(max_child_depth, child_depth);
        }
        
        return max_child_depth;
    }
    
    std::vector<std::vector<double>> apply_graph_convolution(
        const std::vector<std::vector<double>>& node_features,
        const QueryPlanGraph& graph,
        const GraphConvolutionLayer& conv_layer) {
        
        std::vector<std::vector<double>> new_features;
        
        for (int i = 0; i < graph.nodes.size(); i++) {
            std::vector<double> aggregated_features(node_feature_dim, 0.0);
            int neighbor_count = 0;
            
            // Aggregate features from neighbors (including self)
            aggregated_features = node_features[i];
            neighbor_count = 1;
            
            // Add features from children
            for (int child_idx : graph.nodes[i].children) {
                if (child_idx < node_features.size()) {
                    for (int j = 0; j < node_feature_dim && j < node_features[child_idx].size(); j++) {
                        aggregated_features[j] += node_features[child_idx][j];
                    }
                    neighbor_count++;
                }
            }
            
            // Average the aggregated features
            if (neighbor_count > 1) {
                for (double& feat : aggregated_features) {
                    feat /= neighbor_count;
                }
            }
            
            // Apply message passing transformation
            std::vector<double> transformed = apply_linear_layer(aggregated_features, conv_layer.message_passing);
            new_features.push_back(transformed);
        }
        
        return new_features;
    }
    
    std::vector<double> apply_linear_layer(const std::vector<double>& input, const GNNLayer& layer) {
        std::vector<double> output(layer.output_dim, 0.0);
        
        // Matrix multiplication: output = weights * input + bias
        for (int i = 0; i < layer.output_dim; i++) {
            for (int j = 0; j < layer.input_dim && j < input.size() && j < layer.weights[i].size(); j++) {
                output[i] += layer.weights[i][j] * input[j];
            }
            output[i] += layer.bias[i];
            
            // Apply activation function
            if (layer.activation == "relu") {
                output[i] = std::max(0.0, output[i]);
            } else if (layer.activation == "sigmoid") {
                output[i] = 1.0 / (1.0 + std::exp(-output[i]));
            } else if (layer.activation == "tanh") {
                output[i] = std::tanh(output[i]);
            }
            // "linear" activation does nothing
        }
        
        return output;
    }
    
    void update_weights(double gradient) {
        // Simplified weight update (in practice, would use proper backpropagation)
        for (auto& layer : fc_layers) {
            for (auto& weight_row : layer.weights) {
                for (double& weight : weight_row) {
                    weight -= learning_rate * gradient * 0.1; // Simplified gradient
                }
            }
            for (double& bias : layer.bias) {
                bias -= learning_rate * gradient * 0.1;
            }
        }
    }
};

// C interface for PostgreSQL integration
extern "C" {
    static AQDGraphNeuralNetwork* gnn_model = nullptr;
    
    void aqd_gnn_initialize() {
        if (!gnn_model) {
            gnn_model = new AQDGraphNeuralNetwork();
        }
    }
    
    void aqd_gnn_cleanup() {
        if (gnn_model) {
            delete gnn_model;
            gnn_model = nullptr;
        }
    }
    
    double aqd_gnn_predict(const char* explain_json_str) {
        if (!gnn_model) {
            return 0.5; // Default routing
        }
        
        try {
            json explain_json = json::parse(explain_json_str);
            QueryPlanGraph graph = gnn_model->parse_postgres_plan(explain_json);
            return gnn_model->predict_routing(graph);
        } catch (const std::exception& e) {
            std::cerr << "Error in GNN prediction: " << e.what() << std::endl;
            return 0.5;
        }
    }
    
    int aqd_gnn_load_model(const char* filepath) {
        if (!gnn_model) {
            aqd_gnn_initialize();
        }
        
        return gnn_model->load_model(std::string(filepath)) ? 1 : 0;
    }
    
    int aqd_gnn_save_model(const char* filepath) {
        if (!gnn_model) {
            return 0;
        }
        
        return gnn_model->save_model(std::string(filepath)) ? 1 : 0;
    }
}

// Standalone training program
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <training_data.json> <output_model.json>" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    std::cout << "AQD Graph Neural Network Trainer" << std::endl;
    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    
    try {
        // Load training data
        std::ifstream file(input_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open input file " << input_file << std::endl;
            return 1;
        }
        
        json training_data_json;
        file >> training_data_json;
        
        // Parse training data
        std::vector<std::pair<QueryPlanGraph, double>> training_data;
        AQDGraphNeuralNetwork gnn;
        
        for (const auto& record : training_data_json) {
            if (record.contains("explain_json") && record.contains("log_time_gap")) {
                QueryPlanGraph graph = gnn.parse_postgres_plan(record["explain_json"]);
                double target = record["log_time_gap"];
                
                // Convert log gap to routing probability
                double routing_prob = 1.0 / (1.0 + std::exp(-target)); // Sigmoid
                
                training_data.emplace_back(graph, routing_prob);
            }
        }
        
        std::cout << "Loaded " << training_data.size() << " training samples" << std::endl;
        
        if (training_data.empty()) {
            std::cerr << "No valid training data found" << std::endl;
            return 1;
        }
        
        // Train the model
        gnn.train_model(training_data);
        
        // Save the trained model
        if (!gnn.save_model(output_file)) {
            return 1;
        }
        
        std::cout << "GNN training completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}