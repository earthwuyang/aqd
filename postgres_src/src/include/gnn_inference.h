#ifndef GNN_INFERENCE_H
#define GNN_INFERENCE_H

#include "postgres.h"
#include "nodes/plannodes.h"

/* Simple GNN model for plan trees: two-layer mean-aggregator */

#define GNN_MAX_FEATURES 32
#define GNN_MAX_HIDDEN 32

typedef struct GNNWeights
{
    int in_features;
    int hidden_dim;
    double W1[GNN_MAX_FEATURES][GNN_MAX_HIDDEN];
    double b1[GNN_MAX_HIDDEN];
    double W2[GNN_MAX_HIDDEN]; /* output weights to scalar */
    double b2;                 /* output bias */
} GNNWeights;

typedef struct GNNModel
{
    GNNWeights weights;
    bool loaded;
} GNNModel;

/* API */
extern GNNModel *gnn_create_model(void);
extern void gnn_free_model(GNNModel *model);
extern bool gnn_load_model(GNNModel *model, const char *path);
extern double gnn_predict_plan(GNNModel *model, PlannedStmt *planned_stmt);

#endif /* GNN_INFERENCE_H */

