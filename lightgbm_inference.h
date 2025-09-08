#ifndef LIGHTGBM_INFERENCE_H
#define LIGHTGBM_INFERENCE_H

#include <vector>
#include <string>
#include <map>
#include <memory>

namespace lightgbm_inference {

// Simple tree node structure
struct TreeNode {
    bool is_leaf;
    int feature_index;
    double threshold;
    double leaf_value;
    std::shared_ptr<TreeNode> left_child;
    std::shared_ptr<TreeNode> right_child;
    
    TreeNode() : is_leaf(false), feature_index(-1), threshold(0.0), leaf_value(0.0) {}
};

// Simple tree structure
class Tree {
private:
    std::shared_ptr<TreeNode> root;
    
public:
    Tree() : root(nullptr) {}
    
    bool load_from_text(const std::string& tree_text);
    double predict(const std::vector<double>& features) const;
    
private:
    double predict_node(const std::shared_ptr<TreeNode>& node, const std::vector<double>& features) const;
};

// Main inference engine
class LightGBMPredictor {
private:
    std::vector<Tree> trees;
    std::vector<std::string> feature_names;
    std::map<std::string, int> feature_name_to_index;
    
public:
    LightGBMPredictor();
    ~LightGBMPredictor();
    
    // Load model from text file (the format LightGBM saves to)
    bool load_model(const std::string& model_path);
    
    // Make prediction using feature names
    double predict(const std::map<std::string, double>& features) const;
    
    // Make prediction using feature vector (in correct order)
    double predict(const std::vector<double>& features) const;
    
    // Get feature names
    const std::vector<std::string>& get_feature_names() const { return feature_names; }
    
    // Check if model is loaded
    bool is_loaded() const { return !trees.empty(); }
};

} // namespace lightgbm_inference

#endif // LIGHTGBM_INFERENCE_H