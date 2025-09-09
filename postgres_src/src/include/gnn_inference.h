#ifndef GNN_INFERENCE_H
#define GNN_INFERENCE_H

#include "postgres.h"
#include "nodes/plannodes.h"
#include "rginn_core.h"

typedef struct GNNModel
{
    RGINNModel core;
    bool loaded;
} GNNModel;

extern GNNModel *gnn_create_model(void);
extern void gnn_free_model(GNNModel *model);
extern bool gnn_load_model(GNNModel *model, const char *path);
extern double gnn_predict_plan(GNNModel *model, PlannedStmt *planned_stmt);

#endif /* GNN_INFERENCE_H */
