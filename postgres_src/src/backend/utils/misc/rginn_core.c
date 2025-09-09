/* PostgreSQL kernel-compatible R-GINN core implementation */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "rginn_core.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_HIDDEN_DIM 64

static inline double relu(double x) { 
    return x > 0 ? x : 0; 
}

static inline double randn() { 
    double u = (rand() + 1.0) / (RAND_MAX + 2.0); 
    double v = (rand() + 1.0) / (RAND_MAX + 2.0); 
    return sqrt(-2.0 * log(u)) * cos(2 * M_PI * v); 
}

void rginn_init(RGINNModel *m, int in_dim, int hidden_dim, int num_layers, int num_rel, double lr) {
    memset(m, 0, sizeof(*m));
    m->in_dim = in_dim;
    m->hidden_dim = hidden_dim;
    m->num_layers = 1;
    m->num_rel = num_rel;
    m->lr = lr;
    m->eps = 0.0;
    
    m->Win = (double*)calloc((size_t)hidden_dim * in_dim, sizeof(double));
    m->bin = (double*)calloc((size_t)hidden_dim, sizeof(double));
    m->Wr = (double*)calloc((size_t)num_rel * hidden_dim * hidden_dim, sizeof(double));
    m->A = (double*)calloc((size_t)hidden_dim * hidden_dim, sizeof(double));
    m->b = (double*)calloc((size_t)hidden_dim, sizeof(double));
    m->Wout = (double*)calloc((size_t)hidden_dim, sizeof(double));
    
    /* Initialize weights */
    {
        int i, j, r;
        for (i = 0; i < hidden_dim * in_dim; ++i) 
            m->Win[i] = 0.01 * randn();
        for (i = 0; i < hidden_dim * hidden_dim; ++i) 
            m->A[i] = 0.01 * randn();
        for (r = 0; r < num_rel; ++r) {
            for (i = 0; i < hidden_dim * hidden_dim; ++i) 
                m->Wr[r * hidden_dim * hidden_dim + i] = 0.01 * randn();
        }
        for (j = 0; j < hidden_dim; ++j) 
            m->Wout[j] = 0.01 * randn();
    }
}

void rginn_free(RGINNModel *m) {
    if (!m) return;
    free(m->Win);
    free(m->bin);
    free(m->Wr);
    free(m->A);
    free(m->b);
    free(m->Wout);
    memset(m, 0, sizeof(*m));
}

static void proj_in(const RGINNModel *m, const double *x, double *h) {
    int H = m->hidden_dim, F = m->in_dim;
    int j, i;
    for (j = 0; j < H; ++j) {
        double s = m->bin[j];
        const double *wr = &m->Win[j * F];
        for (i = 0; i < F; ++i) 
            s += wr[i] * x[i];
        h[j] = relu(s);
    }
}

double rginn_forward(const RGINNModel *m, const RGGraph *g, double *h0, double *m1, double *h1, double *greadout) {
    int N = g->N, H = m->hidden_dim, F = m->in_dim, R = g->num_rel;
    int n, r, j, k;
    double y;
    double sumH[MAX_HIDDEN_DIM];
    
    (void)F; /* unused */
    
    /* Input projection */
    for (n = 0; n < N; ++n) {
        proj_in(m, &g->X[(size_t)n * F], &h0[(size_t)n * H]);
    }
    
    /* Message passing */
    for (n = 0; n < N; ++n) {
        double *mm = &m1[(size_t)n * H];
        
        /* Self features */
        for (j = 0; j < H; ++j) {
            mm[j] = (1.0 + m->eps) * h0[(size_t)n * H + j];
        }
        
        /* Aggregate from neighbors per relation */
        for (r = 0; r < R; ++r) {
            const int *ind = &g->indptr[(size_t)r * (N + 1)];
            int s = ind[n], e = ind[n + 1];
            int eidx;
            const double *Wr;
            
            /* Clear aggregation buffer */
            for (j = 0; j < H; ++j) 
                sumH[j] = 0.0;
            
            /* Sum neighbor features */
            for (eidx = s; eidx < e; ++eidx) {
                int u = g->indices[eidx];
                const double *hu = &h0[(size_t)u * H];
                for (j = 0; j < H; ++j) 
                    sumH[j] += hu[j];
            }
            
            /* Apply relation transform */
            Wr = &m->Wr[(size_t)r * H * H];
            for (j = 0; j < H; ++j) {
                double acc = 0.0;
                const double *row = &Wr[j * H];
                for (k = 0; k < H; ++k) 
                    acc += row[k] * sumH[k];
                mm[j] += acc;
            }
        }
    }
    
    /* MLP layer */
    for (n = 0; n < N; ++n) {
        const double *mm = &m1[(size_t)n * H];
        double *hh = &h1[(size_t)n * H];
        for (j = 0; j < H; ++j) {
            double s = m->b[j];
            const double *ar = &m->A[j * H];
            for (k = 0; k < H; ++k) 
                s += ar[k] * mm[k];
            hh[j] = relu(s);
        }
    }
    
    /* Readout: sum pooling */
    for (j = 0; j < H; ++j) 
        greadout[j] = 0.0;
    for (n = 0; n < N; ++n) {
        const double *hh = &h1[(size_t)n * H];
        for (j = 0; j < H; ++j) 
            greadout[j] += hh[j];
    }
    
    /* Output scalar */
    y = m->bout;
    for (j = 0; j < H; ++j) 
        y += m->Wout[j] * greadout[j];
    
    return y;
}

void rginn_backward_update(RGINNModel *m, const RGGraph *g, 
                          const double *h0, const double *m1v, 
                          const double *h1, const double *greadout, 
                          double y_pred, double y_true) {
    /* Backpropagation not needed in kernel - placeholder */
    (void)m; (void)g; (void)h0; (void)m1v; (void)h1; 
    (void)greadout; (void)y_pred; (void)y_true;
}

int rginn_save(const RGINNModel *m, const char *path) {
    FILE *f = fopen(path, "w");
    int j, i, k, r;
    if (!f) return -1;
    
    fprintf(f, "%d %d\n", m->in_dim, m->hidden_dim);
    
    /* Save Win */
    for (j = 0; j < m->hidden_dim; ++j) {
        for (i = 0; i < m->in_dim; ++i) {
            fprintf(f, "%g", m->Win[j * m->in_dim + i]);
            if (i + 1 == m->in_dim) fprintf(f, "\n");
            else fprintf(f, " ");
        }
    }
    
    /* Save bin */
    for (j = 0; j < m->hidden_dim; ++j) {
        fprintf(f, "%g", m->bin[j]);
        if (j + 1 == m->hidden_dim) fprintf(f, "\n");
        else fprintf(f, " ");
    }
    
    /* Save num_rel and Wr */
    fprintf(f, "%d\n", m->num_rel);
    for (r = 0; r < m->num_rel; ++r) {
        for (j = 0; j < m->hidden_dim; ++j) {
            for (k = 0; k < m->hidden_dim; ++k) {
                fprintf(f, "%g", m->Wr[r * m->hidden_dim * m->hidden_dim + j * m->hidden_dim + k]);
                if (k + 1 == m->hidden_dim) fprintf(f, "\n");
                else fprintf(f, " ");
            }
        }
    }
    
    /* Save A */
    for (j = 0; j < m->hidden_dim; ++j) {
        for (k = 0; k < m->hidden_dim; ++k) {
            fprintf(f, "%g", m->A[j * m->hidden_dim + k]);
            if (k + 1 == m->hidden_dim) fprintf(f, "\n");
            else fprintf(f, " ");
        }
    }
    
    /* Save b */
    for (j = 0; j < m->hidden_dim; ++j) {
        fprintf(f, "%g", m->b[j]);
        if (j + 1 == m->hidden_dim) fprintf(f, "\n");
        else fprintf(f, " ");
    }
    
    /* Save Wout */
    for (j = 0; j < m->hidden_dim; ++j) {
        fprintf(f, "%g", m->Wout[j]);
        if (j + 1 == m->hidden_dim) fprintf(f, "\n");
        else fprintf(f, " ");
    }
    
    /* Save bout */
    fprintf(f, "%g\n", m->bout);
    
    fclose(f);
    return 0;
}

int rginn_load(RGINNModel *m, const char *path) {
    FILE *f = fopen(path, "r");
    int in_dim, hidden, num_rel;
    int j, i, k, r;
    
    if (!f) return -1;
    
    if (fscanf(f, "%d %d", &in_dim, &hidden) != 2) {
        fclose(f);
        return -1;
    }
    
    /* Initialize model with detected dimensions */
    rginn_init(m, in_dim, hidden, 1, 3, 1e-3);
    
    /* Load Win */
    for (j = 0; j < m->hidden_dim; ++j) {
        for (i = 0; i < m->in_dim; ++i) {
            if (fscanf(f, "%lf", &m->Win[j * m->in_dim + i]) != 1) {
                fclose(f);
                return -1;
            }
        }
    }
    
    /* Load bin */
    for (j = 0; j < m->hidden_dim; ++j) {
        if (fscanf(f, "%lf", &m->bin[j]) != 1) {
            fclose(f);
            return -1;
        }
    }
    
    /* Load num_rel */
    if (fscanf(f, "%d", &num_rel) != 1) {
        fclose(f);
        return -1;
    }
    
    /* If num_rel differs from default, would need to reallocate - for now assume it matches */
    if (num_rel != m->num_rel) {
        /* TODO: Handle different num_rel */
    }
    
    /* Load Wr */
    for (r = 0; r < m->num_rel; ++r) {
        for (j = 0; j < m->hidden_dim; ++j) {
            for (k = 0; k < m->hidden_dim; ++k) {
                if (fscanf(f, "%lf", &m->Wr[r * m->hidden_dim * m->hidden_dim + j * m->hidden_dim + k]) != 1) {
                    fclose(f);
                    return -1;
                }
            }
        }
    }
    
    /* Load A */
    for (j = 0; j < m->hidden_dim; ++j) {
        for (k = 0; k < m->hidden_dim; ++k) {
            if (fscanf(f, "%lf", &m->A[j * m->hidden_dim + k]) != 1) {
                fclose(f);
                return -1;
            }
        }
    }
    
    /* Load b */
    for (j = 0; j < m->hidden_dim; ++j) {
        if (fscanf(f, "%lf", &m->b[j]) != 1) {
            fclose(f);
            return -1;
        }
    }
    
    /* Load Wout */
    for (j = 0; j < m->hidden_dim; ++j) {
        if (fscanf(f, "%lf", &m->Wout[j]) != 1) {
            fclose(f);
            return -1;
        }
    }
    
    /* Load bout */
    if (fscanf(f, "%lf", &m->bout) != 1) {
        fclose(f);
        return -1;
    }
    
    fclose(f);
    return 0;
}