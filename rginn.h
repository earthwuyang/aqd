// Simple R-GIN (Relational GIN) core in C for shared trainer/kernel use
#ifndef RGINN_H
#define RGINN_H
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int in_dim;       // input feature dimension
    int hidden_dim;   // hidden dimension
    int num_layers;   // number of GIN layers (currently 1 supported)
    int num_rel;      // number of relation types
    double eps;       // (1+eps) self coefficient
    double lr;        // learning rate
    // Parameters (single layer for now):
    // Input projection
    double *Win;   // [hidden_dim * in_dim]
    double *bin;   // [hidden_dim]
    // Relation transforms Wr[r]
    double *Wr;    // [num_rel * hidden_dim * hidden_dim]
    // Layer MLP: A,b
    double *A;     // [hidden_dim * hidden_dim]
    double *b;     // [hidden_dim]
    // Readout Wout, bout
    double *Wout;  // [hidden_dim]
    double bout;   // scalar
} RGINNModel;

typedef struct {
    int N;                 // number of nodes
    int in_dim;            // input feature dim
    const double *X;       // node features [N * in_dim]
    int num_rel;           // number of relations
    // adjacency per relation: CSR-like
    const int *indptr;     // [num_rel * (N+1)] offsets per relation
    const int *indices;    // concatenated neighbor indices for all relations
} RGGraph;

// API
void rginn_init(RGINNModel *m, int in_dim, int hidden_dim, int num_layers, int num_rel, double lr);
void rginn_free(RGINNModel *m);

// Forward pass: returns predicted scalar; if work buffers provided, also outputs internal activations for backprop
double rginn_forward(const RGINNModel *m, const RGGraph *g,
                     double *h0, double *m1, double *h1, double *greadout);

// Backprop single example with MSE loss (y_pred - y_true)^2 / 2, update in-place (SGD)
void rginn_backward_update(RGINNModel *m, const RGGraph *g,
                           const double *h0, const double *m1, const double *h1, const double *greadout,
                           double y_pred, double y_true);

// Save/load model text format compatible with kernel loader
int rginn_save(const RGINNModel *m, const char *path);

#ifdef __cplusplus
}
#endif

#endif

