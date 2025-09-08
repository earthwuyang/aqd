#ifndef LIGHTGBM_INFERENCE_H
#define LIGHTGBM_INFERENCE_H

#include "postgres.h"

/* Maximum number of features and trees */
#define LIGHTGBM_MAX_FEATURES 32
#define LIGHTGBM_MAX_TREES 1000

/* Tree node structure */
typedef struct TreeNode
{
    bool is_leaf;
    int feature_index;
    double threshold;
    double leaf_value;
    struct TreeNode *left_child;
    struct TreeNode *right_child;
} TreeNode;

/* Tree structure */
typedef struct Tree
{
    TreeNode *root;
} Tree;

/* Main inference engine */
typedef struct LightGBMPredictor
{
    Tree *trees;
    int num_trees;
    char feature_names[LIGHTGBM_MAX_FEATURES][64];
    int num_features;
    bool is_loaded;
} LightGBMPredictor;

/* Function declarations */

/* Initialize predictor */
extern LightGBMPredictor *lightgbm_create_predictor(void);

/* Free predictor */
extern void lightgbm_free_predictor(LightGBMPredictor *predictor);

/* Load model from text file */
extern bool lightgbm_load_model(LightGBMPredictor *predictor, const char *model_path);

/* Make prediction using feature array (in correct order) */
extern double lightgbm_predict(const LightGBMPredictor *predictor, const double *features);

/* Get feature index by name */
extern int lightgbm_get_feature_index(const LightGBMPredictor *predictor, const char *feature_name);

/* Check if model is loaded */
extern bool lightgbm_is_loaded(const LightGBMPredictor *predictor);

/* Internal functions */
extern TreeNode *lightgbm_create_node(void);
extern void lightgbm_free_tree(TreeNode *node);
extern double lightgbm_predict_tree(const TreeNode *node, const double *features);
extern bool lightgbm_parse_tree_text(Tree *tree, const char *tree_text);

#endif /* LIGHTGBM_INFERENCE_H */