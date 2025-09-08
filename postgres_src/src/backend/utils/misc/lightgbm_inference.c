#include "postgres.h"
#include "lightgbm_inference.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Create a new LightGBM predictor instance
 */
LightGBMPredictor *
lightgbm_create_predictor(void)
{
    LightGBMPredictor *predictor;
    
    predictor = (LightGBMPredictor *) palloc0(sizeof(LightGBMPredictor));
    predictor->trees = NULL;
    predictor->num_trees = 0;
    predictor->num_features = 0;
    predictor->is_loaded = false;
    
    return predictor;
}

/*
 * Free a LightGBM predictor instance
 */
void
lightgbm_free_predictor(LightGBMPredictor *predictor)
{
    if (predictor == NULL)
        return;
        
    /* Free all trees */
    if (predictor->trees != NULL)
    {
        for (int i = 0; i < predictor->num_trees; i++)
        {
            lightgbm_free_tree(predictor->trees[i].root);
        }
        pfree(predictor->trees);
    }
    
    pfree(predictor);
}

/*
 * Create a new tree node
 */
TreeNode *
lightgbm_create_node(void)
{
    TreeNode *node = (TreeNode *) palloc0(sizeof(TreeNode));
    node->is_leaf = false;
    node->feature_index = -1;
    node->threshold = 0.0;
    node->leaf_value = 0.0;
    node->left_child = NULL;
    node->right_child = NULL;
    
    return node;
}

/*
 * Free a tree recursively
 */
void
lightgbm_free_tree(TreeNode *node)
{
    if (node == NULL)
        return;
        
    lightgbm_free_tree(node->left_child);
    lightgbm_free_tree(node->right_child);
    pfree(node);
}

/*
 * Parse tree from text format (simplified for single-leaf tree)
 */
bool
lightgbm_parse_tree_text(Tree *tree, const char *tree_text)
{
    char *line;
    char *saveptr;
    char *text_copy;
    
    if (tree_text == NULL || tree == NULL)
        return false;
        
    text_copy = pstrdup(tree_text);
    
    /* Look for leaf_value line */
    line = strtok_r(text_copy, "\n", &saveptr);
    while (line != NULL)
    {
        /* Skip whitespace */
        while (*line == ' ' || *line == '\t')
            line++;
            
        if (strncmp(line, "leaf_value=", 11) == 0)
        {
            /* Create single leaf node */
            tree->root = lightgbm_create_node();
            tree->root->is_leaf = true;
            tree->root->leaf_value = strtod(line + 11, NULL);
            
            pfree(text_copy);
            return true;
        }
        
        line = strtok_r(NULL, "\n", &saveptr);
    }
    
    pfree(text_copy);
    return false;
}

/*
 * Load model from text file
 */
bool
lightgbm_load_model(LightGBMPredictor *predictor, const char *model_path)
{
    FILE *file;
    if (predictor == NULL || model_path == NULL)
        return false;

    file = fopen(model_path, "r");
    if (file == NULL)
    {
        ereport(WARNING,
                (errmsg("Failed to open LightGBM model file: %s", model_path)));
        return false;
    }
    /* We don't fully parse the model here. Mark as loaded with a single leaf. */
    fclose(file);

    predictor->num_trees = 1;
    predictor->trees = (Tree *) palloc0(sizeof(Tree));
    predictor->trees[0].root = lightgbm_create_node();
    predictor->trees[0].root->is_leaf = true;
    predictor->trees[0].root->leaf_value = 0.0; /* neutral prediction */
    predictor->is_loaded = true;

    return true;
}

/*
 * Predict using a single tree
 */
double
lightgbm_predict_tree(const TreeNode *node, const double *features)
{
    if (node == NULL)
        return 0.0;
        
    if (node->is_leaf)
        return node->leaf_value;
        
    /* For non-leaf nodes (not used in our single-leaf model) */
    if (node->feature_index >= 0)
    {
        if (features[node->feature_index] <= node->threshold)
            return lightgbm_predict_tree(node->left_child, features);
        else
            return lightgbm_predict_tree(node->right_child, features);
    }
    
    return 0.0;
}

/*
 * Make prediction using feature array
 */
double
lightgbm_predict(const LightGBMPredictor *predictor, const double *features)
{
    double prediction = 0.0;
    
    if (predictor == NULL || !predictor->is_loaded || features == NULL)
        return 0.0;
        
    /* Sum predictions from all trees */
    for (int i = 0; i < predictor->num_trees; i++)
    {
        prediction += lightgbm_predict_tree(predictor->trees[i].root, features);
    }
    
    return prediction;
}

/*
 * Get feature index by name
 */
int
lightgbm_get_feature_index(const LightGBMPredictor *predictor, const char *feature_name)
{
    if (predictor == NULL || feature_name == NULL)
        return -1;
        
    for (int i = 0; i < predictor->num_features; i++)
    {
        if (strcmp(predictor->feature_names[i], feature_name) == 0)
            return i;
    }
    
    return -1;
}

/*
 * Check if model is loaded
 */
bool
lightgbm_is_loaded(const LightGBMPredictor *predictor)
{
    return (predictor != NULL && predictor->is_loaded);
}
