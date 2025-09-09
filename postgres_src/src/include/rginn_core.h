#ifndef RGINN_CORE_H
#define RGINN_CORE_H

#include <stddef.h>

typedef struct {
    int in_dim;
    int hidden_dim;
    int num_layers;
    int num_rel;
    double eps;
    double lr;
    double *Win;  // [hidden_dim * in_dim]
    double *bin;  // [hidden_dim]
    double *Wr;   // [num_rel * hidden_dim * hidden_dim]
    double *A;    // [hidden_dim * hidden_dim]
    double *b;    // [hidden_dim]
    double *Wout; // [hidden_dim]
    double bout;  // scalar
} RGINNModel;

typedef struct {
    int N;
    int in_dim;
    const double *X;       // [N * in_dim]
    int num_rel;
    const int *indptr;     // [num_rel * (N+1)] absolute offsets into indices
    const int *indices;    // concatenated neighbor indices for all relations
} RGGraph;

void rginn_init(RGINNModel *m, int in_dim, int hidden_dim, int num_layers, int num_rel, double lr);
void rginn_free(RGINNModel *m);
double rginn_forward(const RGINNModel *m, const RGGraph *g,
                     double *h0, double *m1, double *h1, double *greadout);
void rginn_backward_update(RGINNModel *m, const RGGraph *g,
                           const double *h0, const double *m1, const double *h1, const double *greadout,
                           double y_pred, double y_true);
int rginn_save(const RGINNModel *m, const char *path);
int rginn_load(RGINNModel *m, const char *path);

#endif
