#include "postgres.h"
#include "gnn_inference.h"
#include "utils/memutils.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static void relu_vec(double *x, int n)
{
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0) x[i] = 0.0;
}

static int op_type_id(NodeTag tag)
{
    /* Map key node types to ids (0..k); others -> small set */
    switch (tag)
    {
        case T_SeqScan: return 1;
        case T_IndexScan: return 2;
        case T_BitmapHeapScan: return 3;
        case T_NestLoop: return 4;
        case T_MergeJoin: return 5;
        case T_HashJoin: return 6;
        case T_Sort: return 7;
        case T_Agg: return 8;
        case T_Group: return 9;
        default: return 0;
    }
}

static void plan_node_features(Plan *plan, double *feat, int d)
{
    /* Build a small feature vector: [one-hot op, rows, width, cost per row] */
    memset(feat, 0, sizeof(double) * d);
    int k = 10; /* one-hot size */
    int id = op_type_id(nodeTag(plan));
    if (id >= 0 && id < k && id < d)
        feat[id] = 1.0;
    int idx = k;
    if (idx < d) feat[idx++] = (plan->plan_rows);
    if (idx < d) feat[idx++] = (double)plan->plan_width;
    if (idx < d)
    {
        double cpr = (plan->plan_rows > 0) ? (plan->total_cost / plan->plan_rows) : plan->total_cost;
        feat[idx++] = cpr;
    }
    /* zero-fill rest */
}

GNNModel *
gnn_create_model(void)
{
    GNNModel *m = (GNNModel *) palloc0(sizeof(GNNModel));
    m->weights.in_features = 16; /* default in-features */
    m->weights.hidden_dim = 16;
    for (int i = 0; i < m->weights.in_features; i++)
        for (int j = 0; j < m->weights.hidden_dim; j++)
            m->weights.W1[i][j] = 0.01; /* small init */
    for (int j = 0; j < m->weights.hidden_dim; j++)
        m->weights.b1[j] = 0.0;
    for (int j = 0; j < m->weights.hidden_dim; j++)
        m->weights.W2[j] = 0.01;
    m->weights.b2 = 0.0;
    m->loaded = false;
    return m;
}

void
gnn_free_model(GNNModel *model)
{
    if (!model) return;
    pfree(model);
}

bool
gnn_load_model(GNNModel *model, const char *path)
{
    if (!model || !path) return false;
    FILE *f = fopen(path, "r");
    if (!f) return false;
    int fin, hidden;
    if (fscanf(f, "%d %d\n", &fin, &hidden) != 2)
    { fclose(f); return false; }
    if (fin > GNN_MAX_FEATURES || hidden > GNN_MAX_HIDDEN)
    { fclose(f); return false; }
    model->weights.in_features = fin;
    model->weights.hidden_dim = hidden;
    for (int i = 0; i < fin; i++)
        for (int j = 0; j < hidden; j++)
            fscanf(f, "%lf", &model->weights.W1[i][j]);
    for (int j = 0; j < hidden; j++)
        fscanf(f, "%lf", &model->weights.b1[j]);
    for (int j = 0; j < hidden; j++)
        fscanf(f, "%lf", &model->weights.W2[j]);
    fscanf(f, "%lf", &model->weights.b2);
    fclose(f);
    model->loaded = true;
    return true;
}

static void aggregate_node(Plan *plan, double *out_hidden, GNNModel *model)
{
    /* Aggregate mean of children's hidden reps + own projected features */
    int fin = model->weights.in_features;
    int h = model->weights.hidden_dim;
    double x[GNN_MAX_FEATURES];
    plan_node_features(plan, x, fin);
    double h_self[GNN_MAX_HIDDEN];
    for (int j = 0; j < h; j++)
    {
        double s = model->weights.b1[j];
        for (int i = 0; i < fin; i++) s += x[i] * model->weights.W1[i][j];
        h_self[j] = s;
    }

    /* Aggregate children */
    int child_count = 0;
    double h_child_sum[GNN_MAX_HIDDEN];
    for (int j = 0; j < h; j++) h_child_sum[j] = 0.0;
    if (plan->lefttree)
    {
        double h_left[GNN_MAX_HIDDEN];
        aggregate_node(plan->lefttree, h_left, model);
        for (int j = 0; j < h; j++) h_child_sum[j] += h_left[j];
        child_count++;
    }
    if (plan->righttree)
    {
        double h_right[GNN_MAX_HIDDEN];
        aggregate_node(plan->righttree, h_right, model);
        for (int j = 0; j < h; j++) h_child_sum[j] += h_right[j];
        child_count++;
    }
    /* Mean aggregate */
    if (child_count > 0)
        for (int j = 0; j < h; j++) h_child_sum[j] /= (double)child_count;

    /* Combine and activate */
    for (int j = 0; j < h; j++)
        out_hidden[j] = h_self[j] + h_child_sum[j];
    relu_vec(out_hidden, h);
}

double
gnn_predict_plan(GNNModel *model, PlannedStmt *planned_stmt)
{
    if (!model || !model->loaded || !planned_stmt || !planned_stmt->planTree)
        return 0.0;
    int h = model->weights.hidden_dim;
    double h_root[GNN_MAX_HIDDEN];
    aggregate_node(planned_stmt->planTree, h_root, model);
    /* Readout to scalar */
    double y = model->weights.b2;
    for (int j = 0; j < h; j++) y += h_root[j] * model->weights.W2[j];
    return y;
}

