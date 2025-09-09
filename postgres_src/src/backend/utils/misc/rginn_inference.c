/*
 * R-GINN inference in PostgreSQL kernel
 * Integrates the R-GINN (Relational Graph Isomorphism Network) for query routing
 */

#include "postgres.h"
#include "rginn_inference.h"
#include "nodes/plannodes.h"
#include "nodes/pg_list.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* R-GINN Model structure compatible with trained models */
typedef struct {
    int in_dim;
    int hidden_dim;
    int num_layers;
    int num_rel;
    double eps;
    
    /* Model parameters */
    double *Win;    /* Input projection [hidden_dim * in_dim] */
    double *bin;    /* Input bias [hidden_dim] */
    double *Wr;     /* Relation transforms [num_rel * hidden_dim * hidden_dim] */
    double *A;      /* Layer MLP weights [hidden_dim * hidden_dim] */
    double *b;      /* Layer MLP bias [hidden_dim] */
    double *Wout;   /* Output weights [hidden_dim] */
    double bout;    /* Output bias */
    
    bool loaded;
} RGINNModel;

/* Graph structure for plan tree */
typedef struct {
    int num_nodes;
    double *features;  /* [num_nodes * in_dim] */
    int *children;     /* [num_nodes * 2] for binary tree, -1 for no child */
} PlanGraph;

/* Global model instance */
static RGINNModel *global_rginn_model = NULL;

/* Helper functions */
static double relu(double x) {
    return x > 0 ? x : 0;
}

static int op_type_id(NodeTag tag) {
    switch (tag) {
        case T_SeqScan: return 1;
        case T_IndexScan: return 2;
        case T_BitmapHeapScan: return 3;
        case T_NestLoop: return 4;
        case T_MergeJoin: return 5;
        case T_HashJoin: return 6;
        case T_Sort: return 7;
        case T_Agg: return 8;
        case T_Group: return 9;
        case T_Hash: return 10;
        case T_Material: return 11;
        case T_MergeAppend: return 12;
        case T_Result: return 13;
        case T_Append: return 14;
        case T_Limit: return 15;
        default: return 0;
    }
}

/* Extract features from a plan node */
static void extract_node_features(Plan *plan, double *features, int in_dim) {
    int op_id;
    int idx;
    
    memset(features, 0, sizeof(double) * in_dim);
    
    /* One-hot encoding for operation type (first 16 features) */
    op_id = op_type_id(nodeTag(plan));
    if (op_id >= 0 && op_id < 16 && op_id < in_dim) {
        features[op_id] = 1.0;
    }
    
    /* Numeric features (normalized) */
    idx = 16;
    if (idx < in_dim) {
        /* Log of rows */
        features[idx++] = log(plan->plan_rows + 1.0) / 10.0;
    }
    if (idx < in_dim) {
        /* Width normalized */
        features[idx++] = (double)plan->plan_width / 100.0;
    }
    if (idx < in_dim) {
        /* Log of total cost */
        features[idx++] = log(plan->total_cost + 1.0) / 10.0;
    }
}

/* Build graph from plan tree */
static int build_plan_graph_recursive(Plan *plan, PlanGraph *graph, int node_idx, int in_dim) {
    int next_idx;
    
    if (!plan || node_idx >= graph->num_nodes) return -1;
    
    /* Extract features for current node */
    extract_node_features(plan, &graph->features[node_idx * in_dim], in_dim);
    
    /* Initialize children */
    graph->children[node_idx * 2] = -1;
    graph->children[node_idx * 2 + 1] = -1;
    
    next_idx = node_idx + 1;
    
    /* Process left child */
    if (plan->lefttree) {
        graph->children[node_idx * 2] = next_idx;
        next_idx = build_plan_graph_recursive(plan->lefttree, graph, next_idx, in_dim);
    }
    
    /* Process right child */
    if (plan->righttree) {
        graph->children[node_idx * 2 + 1] = next_idx;
        next_idx = build_plan_graph_recursive(plan->righttree, graph, next_idx, in_dim);
    }
    
    return next_idx;
}

/* Count nodes in plan tree */
static int count_plan_nodes(Plan *plan) {
    if (!plan) return 0;
    return 1 + count_plan_nodes(plan->lefttree) + count_plan_nodes(plan->righttree);
}

/* Forward pass through R-GINN */
static double rginn_forward_inference(RGINNModel *model, PlanGraph *graph) {
    int N = graph->num_nodes;
    int H = model->hidden_dim;
    int D = model->in_dim;
    double *h0;
    double *h1;
    int i, j, k;
    
    /* Allocate working memory */
    h0 = (double *)palloc0(sizeof(double) * N * H);
    h1 = (double *)palloc0(sizeof(double) * N * H);
    
    /* Input projection: h0 = ReLU(X @ Win^T + bin) */
    for (i = 0; i < N; i++) {
        for (j = 0; j < H; j++) {
            double sum = model->bin[j];
            for (k = 0; k < D; k++) {
                sum += graph->features[i * D + k] * model->Win[j * D + k];
            }
            h0[i * H + j] = relu(sum);
        }
    }
    
    /* Message passing (bottom-up aggregation for tree) */
    /* Process nodes in reverse order (post-order) for bottom-up */
    for (i = N - 1; i >= 0; i--) {
        int child_count = 0;
        int c;
        
        /* Self features */
        for (j = 0; j < H; j++) {
            h1[i * H + j] = (1 + model->eps) * h0[i * H + j];
        }
        
        /* Aggregate children */
        for (c = 0; c < 2; c++) {
            int child_idx = graph->children[i * 2 + c];
            if (child_idx >= 0 && child_idx < N) {
                for (j = 0; j < H; j++) {
                    h1[i * H + j] += h1[child_idx * H + j];
                }
                child_count++;
            }
        }
        
        /* Mean pooling if there are children */
        if (child_count > 0) {
            for (j = 0; j < H; j++) {
                h1[i * H + j] /= (1 + child_count);
            }
        }
        
        /* Apply MLP: h1 = ReLU(h1 @ A^T + b) */
        {
            double temp[64]; /* Assuming hidden_dim <= 64 */
            for (j = 0; j < H; j++) {
                double sum = model->b[j];
                for (k = 0; k < H; k++) {
                sum += h1[i * H + k] * model->A[j * H + k];
            }
                temp[j] = relu(sum);
            }
            
            /* Copy back */
            for (j = 0; j < H; j++) {
                h1[i * H + j] = temp[j];
            }
        }
    }
    
    /* Readout from root node (index 0) */
    {
        double output = model->bout;
        int jj;
        for (jj = 0; jj < H; jj++) {
            output += h1[jj] * model->Wout[jj];
        }
        
        pfree(h0);
        pfree(h1);
        
        return output;
    }
}

/* Create and initialize R-GINN model */
static RGINNModel *rginn_create_model(void) {
    RGINNModel *model = (RGINNModel *)palloc0(sizeof(RGINNModel));
    model->in_dim = 16;
    model->hidden_dim = 32;
    model->num_layers = 1;
    model->num_rel = 1;
    model->eps = 0.0;
    model->loaded = false;
    
    /* Allocate parameter arrays */
    model->Win = (double *)palloc0(sizeof(double) * model->hidden_dim * model->in_dim);
    model->bin = (double *)palloc0(sizeof(double) * model->hidden_dim);
    model->Wr = (double *)palloc0(sizeof(double) * model->num_rel * model->hidden_dim * model->hidden_dim);
    model->A = (double *)palloc0(sizeof(double) * model->hidden_dim * model->hidden_dim);
    model->b = (double *)palloc0(sizeof(double) * model->hidden_dim);
    model->Wout = (double *)palloc0(sizeof(double) * model->hidden_dim);
    model->bout = 0.0;
    
    return model;
}

/* Free R-GINN model - keep for future use */
#ifdef UNUSED
static void rginn_free_model(RGINNModel *model) {
    if (!model) return;
    
    if (model->Win) pfree(model->Win);
    if (model->bin) pfree(model->bin);
    if (model->Wr) pfree(model->Wr);
    if (model->A) pfree(model->A);
    if (model->b) pfree(model->b);
    if (model->Wout) pfree(model->Wout);
    
    pfree(model);
}
#endif

/* Load R-GINN model from file */
bool rginn_load_model(const char *path) {
    FILE *f;
    RGINNModel *model;
    int H, D;
    
    if (!path) return false;
    
    f = fopen(path, "r");
    if (!f) {
        elog(WARNING, "Failed to open R-GINN model file: %s", path);
        return false;
    }
    
    /* Create model if not exists */
    if (!global_rginn_model) {
        global_rginn_model = rginn_create_model();
    }
    
    model = global_rginn_model;
    
    /* Read model dimensions - format: in_dim hidden_dim */
    if (fscanf(f, "%d %d\n", &model->in_dim, &model->hidden_dim) != 2) {
        fclose(f);
        elog(WARNING, "Failed to read R-GINN model header");
        return false;
    }
    
    /* Set default values for other parameters */
    model->num_layers = 1;
    model->num_rel = 1; 
    model->eps = 0.0;
    
    H = model->hidden_dim;
    D = model->in_dim;
    
    /* Reallocate if dimensions changed */
    pfree(model->Win);
    pfree(model->bin);
    pfree(model->A);
    pfree(model->b);
    pfree(model->Wout);
    
    model->Win = (double *)palloc0(sizeof(double) * H * D);
    model->bin = (double *)palloc0(sizeof(double) * H);
    model->A = (double *)palloc0(sizeof(double) * H * H);
    model->b = (double *)palloc0(sizeof(double) * H);
    model->Wout = (double *)palloc0(sizeof(double) * H);
    
    /* Read Win */
    {
        int i;
        for (i = 0; i < H * D; i++) {
            if (fscanf(f, "%lf", &model->Win[i]) != 1) {
                fclose(f);
                return false;
            }
        }
    }
    
    /* Read bin */
    {
        int i;
        for (i = 0; i < H; i++) {
            if (fscanf(f, "%lf", &model->bin[i]) != 1) {
                fclose(f);
                return false;
            }
        }
    }
    
    /* Read num_rel and skip Wr matrices */
    {
        int num_rel_in_file;
        int i;
        if (fscanf(f, "%d\n", &num_rel_in_file) != 1) {
            fclose(f);
            return false;
        }
        /* Skip relation weight matrices - we don't use them in inference */
        for (i = 0; i < num_rel_in_file * H * H; i++) {
            double dummy;
            (void)fscanf(f, "%lf", &dummy);
        }
    }
    
    /* Read A */
    {
        int i;
        for (i = 0; i < H * H; i++) {
            if (fscanf(f, "%lf", &model->A[i]) != 1) {
                fclose(f);
                return false;
            }
        }
    }
    
    /* Read b */
    {
        int i;
        for (i = 0; i < H; i++) {
            if (fscanf(f, "%lf", &model->b[i]) != 1) {
                fclose(f);
                return false;
            }
        }
    }
    
    /* Read Wout */
    {
        int i;
        for (i = 0; i < H; i++) {
            if (fscanf(f, "%lf", &model->Wout[i]) != 1) {
                fclose(f);
                return false;
            }
        }
    }
    
    /* Read bout */
    if (fscanf(f, "%lf", &model->bout) != 1) {
        fclose(f);
        return false;
    }
    
    fclose(f);
    model->loaded = true;
    
    elog(LOG, "Loaded R-GINN model: in_dim=%d, hidden=%d, layers=%d", 
         model->in_dim, model->hidden_dim, model->num_layers);
    
    return true;
}

/* Main prediction function */
double rginn_predict_plan(PlannedStmt *planned_stmt) {
    int num_nodes;
    PlanGraph graph;
    double prediction;
    
    if (!global_rginn_model || !global_rginn_model->loaded) {
        return 0.0;
    }
    
    if (!planned_stmt || !planned_stmt->planTree) {
        return 0.0;
    }
    
    /* Count nodes in plan tree */
    num_nodes = count_plan_nodes(planned_stmt->planTree);
    if (num_nodes == 0) {
        return 0.0;
    }
    
    /* Build graph from plan tree */
    graph.num_nodes = num_nodes;
    graph.features = (double *)palloc0(sizeof(double) * num_nodes * global_rginn_model->in_dim);
    graph.children = (int *)palloc0(sizeof(int) * num_nodes * 2);
    
    build_plan_graph_recursive(planned_stmt->planTree, &graph, 0, global_rginn_model->in_dim);
    
    /* Run inference */
    prediction = rginn_forward_inference(global_rginn_model, &graph);
    
    /* Cleanup */
    pfree(graph.features);
    pfree(graph.children);
    
    return prediction;
}