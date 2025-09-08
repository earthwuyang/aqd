#include "lightgbm_inference.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace lightgbm_inference {

bool Tree::load_from_text(const std::string& tree_text) {
    std::istringstream iss(tree_text);
    std::string line;
    
    // Skip header lines
    while (std::getline(iss, line)) {
        if (line.find("Tree=") != std::string::npos) {
            break;
        }
    }
    
    // Read tree properties
    int num_leaves = 0;
    while (std::getline(iss, line)) {
        if (line.find("num_leaves=") != std::string::npos) {
            num_leaves = std::stoi(line.substr(line.find("=") + 1));
        }
        else if (line.find("leaf_value=") != std::string::npos) {
            // For single leaf trees (like our trained model)
            if (num_leaves == 1) {
                root = std::make_shared<TreeNode>();
                root->is_leaf = true;
                root->leaf_value = std::stod(line.substr(line.find("=") + 1));
                return true;
            }
            break;
        }
        else if (line.empty()) {
            break;
        }
    }
    
    return false;
}

double Tree::predict(const std::vector<double>& features) const {
    if (!root) {
        return 0.0;
    }
    return predict_node(root, features);
}

double Tree::predict_node(const std::shared_ptr<TreeNode>& node, const std::vector<double>& features) const {
    if (!node) {
        return 0.0;
    }
    
    if (node->is_leaf) {
        return node->leaf_value;
    }
    
    // For non-leaf nodes (not applicable to our single-leaf model)
    if (node->feature_index >= 0 && node->feature_index < features.size()) {
        if (features[node->feature_index] <= node->threshold) {
            return predict_node(node->left_child, features);
        } else {
            return predict_node(node->right_child, features);
        }
    }
    
    return 0.0;
}

LightGBMPredictor::LightGBMPredictor() {
}

LightGBMPredictor::~LightGBMPredictor() {
}

bool LightGBMPredictor::load_model(const std::string& model_path) {
    std::ifstream file(model_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return false;
    }
    
    std::string line;
    
    // Parse feature names
    while (std::getline(file, line)) {
        if (line.find("feature_names=") != std::string::npos) {
            std::string features_str = line.substr(line.find("=") + 1);
            std::istringstream iss(features_str);
            std::string feature;
            
            feature_names.clear();
            feature_name_to_index.clear();
            
            int index = 0;
            while (iss >> feature) {
                feature_names.push_back(feature);
                feature_name_to_index[feature] = index++;
            }
            break;
        }
    }
    
    // Read the entire file content to extract tree
    file.seekg(0, std::ios::beg);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    
    // Parse trees (in our case, just one tree with one leaf)
    Tree tree;
    if (tree.load_from_text(content)) {
        trees.clear();
        trees.push_back(tree);
        return true;
    }
    
    return false;
}

double LightGBMPredictor::predict(const std::map<std::string, double>& features) const {
    if (trees.empty()) {
        return 0.0;
    }
    
    // Convert feature map to ordered vector
    std::vector<double> feature_vector(feature_names.size(), 0.0);
    
    for (const auto& pair : features) {
        auto it = feature_name_to_index.find(pair.first);
        if (it != feature_name_to_index.end()) {
            feature_vector[it->second] = pair.second;
        }
    }
    
    return predict(feature_vector);
}

double LightGBMPredictor::predict(const std::vector<double>& features) const {
    if (trees.empty()) {
        return 0.0;
    }
    
    double prediction = 0.0;
    for (const auto& tree : trees) {
        prediction += tree.predict(features);
    }
    
    return prediction;
}

} // namespace lightgbm_inference