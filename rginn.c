#include "rginn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static inline double randn() {
    // Box-Muller
    double u = (rand() + 1.0) / (RAND_MAX + 2.0);
    double v = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

void rginn_init(RGINNModel *m, int in_dim, int hidden_dim, int num_layers, int num_rel, double lr) {
    memset(m, 0, sizeof(*m));
    m->in_dim = in_dim;
    m->hidden_dim = hidden_dim;
    m->num_layers = 1; // currently single-layer
    m->num_rel = num_rel;
    m->eps = 0.0;
    m->lr = lr;
    m->Win = (double*)calloc((size_t)hidden_dim * in_dim, sizeof(double));
    m->bin = (double*)calloc((size_t)hidden_dim, sizeof(double));
    m->Wr = (double*)calloc((size_t)num_rel * hidden_dim * hidden_dim, sizeof(double));
    m->A = (double*)calloc((size_t)hidden_dim * hidden_dim, sizeof(double));
    m->b = (double*)calloc((size_t)hidden_dim, sizeof(double));
    m->Wout = (double*)calloc((size_t)hidden_dim, sizeof(double));
    m->bout = 0.0;
    // init small randoms for A, Win, Wr, Wout
    for (int i = 0; i < hidden_dim * in_dim; ++i) m->Win[i] = 0.01 * randn();
    for (int i = 0; i < hidden_dim * hidden_dim; ++i) m->A[i] = 0.01 * randn();
    for (int r = 0; r < num_rel; ++r)
        for (int i = 0; i < hidden_dim * hidden_dim; ++i)
            m->Wr[r*hidden_dim*hidden_dim + i] = 0.01 * randn();
    for (int j = 0; j < hidden_dim; ++j) m->Wout[j] = 0.01 * randn();
}

void rginn_free(RGINNModel *m) {
    if (!m) return;
    free(m->Win); free(m->bin); free(m->Wr); free(m->A); free(m->b); free(m->Wout);
    memset(m, 0, sizeof(*m));
}

static inline double relu(double x){ return x>0?x:0; }

double rginn_forward(const RGINNModel *m, const RGGraph *g,
                     double *h0, double *m1, double *h1, double *greadout) {
    const int N = g->N;
    const int F = m->in_dim;
    const int H = m->hidden_dim;
    // h0 = ReLU(Win x + b)
    for (int n = 0; n < N; ++n) {
        const double *x = &g->X[(size_t)n * F];
        double *h = &h0[(size_t)n * H];
        for (int j = 0; j < H; ++j) {
            double s = m->bin[j];
            const double *wrow = &m->Win[j*F];
            for (int i = 0; i < F; ++i) s += wrow[i] * x[i];
            h[j] = relu(s);
        }
    }
    // m1 = (1+eps) h0 + sum_r Wr[r] * sum_{u in N_r(v)} h0_u
    for (int n = 0; n < N; ++n) {
        double *mm = &m1[(size_t)n * H];
        for (int j = 0; j < H; ++j) mm[j] = (1.0 + m->eps) * h0[(size_t)n * H + j];
        for (int r = 0; r < g->num_rel; ++r) {
            const int *indptr = &g->indptr[(size_t)r * (N+1)];
            int start = indptr[n];
            int end = indptr[n+1];
            // sum neighbors
            double *sumH = (double*)calloc(H, sizeof(double));
            for (int e = start; e < end; ++e) {
                int u = g->indices[e];
                const double *hu = &h0[(size_t)u * H];
                for (int j=0;j<H;++j) sumH[j] += hu[j];
            }
            // apply Wr[r]
            const double *Wr = &m->Wr[(size_t)r * H * H];
            for (int j = 0; j < H; ++j) {
                double s = 0.0;
                const double *wrow = &Wr[j*H];
                for (int k = 0; k < H; ++k) s += wrow[k] * sumH[k];
                mm[j] += s;
            }
            free(sumH);
        }
    }
    // h1 = ReLU(A m1 + b)
    for (int n = 0; n < N; ++n) {
        double *h = &h1[(size_t)n * H];
        const double *mm = &m1[(size_t)n * H];
        for (int j = 0; j < H; ++j) {
            double s = m->b[j];
            const double *arow = &m->A[j*H];
            for (int k = 0; k < H; ++k) s += arow[k] * mm[k];
            h[j] = relu(s);
        }
    }
    // readout = sum_n h1[n]
    for (int j=0;j<H;++j) greadout[j]=0.0;
    for (int n=0;n<N;++n) {
        const double *h=&h1[(size_t)n*H];
        for (int j=0;j<H;++j) greadout[j]+=h[j];
    }
    // y = Wout . readout + bout
    double y = m->bout;
    for (int j=0;j<H;++j) y += m->Wout[j]*greadout[j];
    return y;
}

void rginn_backward_update(RGINNModel *m, const RGGraph *g,
                           const double *h0, const double *m1v, const double *h1,
                           const double *greadout,
                           double y_pred, double y_true) {
    const int N = g->N;
    const int H = m->hidden_dim;
    double dy = (y_pred - y_true); // dL/dy
    // grads
    double *gWout = (double*)calloc(H,sizeof(double));
    double gbout = dy;
    for (int j=0;j<H;++j) gWout[j] = dy * greadout[j];
    // dL/d(readout) = dy * Wout
    double *dG = (double*)calloc(H, sizeof(double));
    for (int j=0;j<H;++j) dG[j] = dy * m->Wout[j];
    // distribute to nodes: dL/dh1[n] += dG
    double *dh1 = (double*)calloc((size_t)N*H,sizeof(double));
    for (int n=0;n<N;++n) {
        for (int j=0;j<H;++j) dh1[(size_t)n*H + j] += dG[j];
    }
    // back through ReLU at h1 and linear A,b: z1 = A m1 + b
    double *dz1 = (double*)calloc((size_t)N*H,sizeof(double));
    for (int n=0;n<N;++n) {
        for (int j=0;j<H;++j) {
            double grad = (h1[(size_t)n*H + j] > 0.0) ? dh1[(size_t)n*H + j] : 0.0;
            dz1[(size_t)n*H + j] = grad;
        }
    }
    // grads for A,b and dL/dm1
    double *gA = (double*)calloc((size_t)H*H,sizeof(double));
    double *gb = (double*)calloc(H,sizeof(double));
    double *dm1 = (double*)calloc((size_t)N*H,sizeof(double));
    for (int n=0;n<N;++n) {
        const double *mm = &m1v[(size_t)n*H];
        const double *dz = &dz1[(size_t)n*H];
        for (int j=0;j<H;++j) {
            gb[j] += dz[j];
            // gA[j,:] += dz[j] * mm[:]
            double *gArow = &gA[j*H];
            for (int k=0;k<H;++k) gArow[k] += dz[j]*mm[k];
        }
        // dm1 += A^T dz
        for (int k=0;k<H;++k){
            double s=0.0; for (int j=0;j<H;++j) s += m->A[j*H + k] * dz[j];
            dm1[(size_t)n*H + k] += s;
        }
    }
    // back through m1 = (1+eps)h0 + sum_r Wr[r] sum_{u in N_r(v)} h0[u]
    // grads for Wr and dh0
    double *gWr = (double*)calloc((size_t)m->num_rel*H*H,sizeof(double));
    double *dh0 = (double*)calloc((size_t)N*H,sizeof(double));
    for (int n=0;n<N;++n) {
        const double *dm = &dm1[(size_t)n*H];
        // self
        for (int j=0;j<H;++j) dh0[(size_t)n*H + j] += (1.0 + m->eps) * dm[j];
        for (int r=0;r<g->num_rel;++r){
            const int *indptr = &g->indptr[(size_t)r*(N+1)];
            int start = indptr[n]; int end = indptr[n+1];
            // gWr += outer(dm, sum_h0_neighbors)
            double *sumH = (double*)calloc(H, sizeof(double));
            for (int e=start;e<end;++e){
                int u=g->indices[e];
                const double* hu=&h0[(size_t)u*H];
                for (int j=0;j<H;++j) sumH[j]+=hu[j];
            }
            double *gWr_r = &gWr[(size_t)r*H*H];
            for (int j=0;j<H;++j){
                double *row=&gWr_r[j*H];
                for (int k=0;k<H;++k) row[k]+=dm[j]*sumH[k];
            }
            // propagate to neighbors via Wr^T
            const double *Wr = &m->Wr[(size_t)r*H*H];
            for (int e=start;e<end;++e){
                int u=g->indices[e];
                for (int k=0;k<H;++k){
                    double s=0.0; for (int j=0;j<H;++j) s += Wr[j*H + k]*dm[j];
                    dh0[(size_t)u*H + k] += s;
                }
            }
            free(sumH);
        }
    }
    // back through h0 = ReLU(Win x + b): z0
    double *dz0 = (double*)calloc((size_t)N*H,sizeof(double));
    for (int n=0;n<N;++n){
        for (int j=0;j<H;++j){
            double grad = (h0[(size_t)n*H + j] > 0.0) ? dh0[(size_t)n*H + j] : 0.0;
            dz0[(size_t)n*H + j]=grad;
        }
    }
    double *gWin = (double*)calloc((size_t)H*m->in_dim,sizeof(double));
    double *gbin = (double*)calloc(H,sizeof(double));
    for (int n=0;n<N;++n){
        const double *x = &g->X[(size_t)n * m->in_dim];
        const double *dz = &dz0[(size_t)n*H];
        for (int j=0;j<H;++j){
            gbin[j]+=dz[j];
            double *row=&gWin[j*m->in_dim];
            for (int i=0;i<m->in_dim;++i) row[i]+=dz[j]*x[i];
        }
    }
    // SGD updates
    double lr = m->lr;
    for (int j=0;j<H;++j) m->Wout[j] -= lr * gWout[j]; m->bout -= lr * gbout;
    for (int i=0;i<H*H;++i) m->A[i]-=lr*gA[i]; for (int j=0;j<H;++j) m->b[j]-=lr*gb[j];
    for (int r=0;r<m->num_rel;++r){
        double *wr=&m->Wr[(size_t)r*H*H]; double *gwr=&gWr[(size_t)r*H*H];
        for (int i=0;i<H*H;++i) wr[i]-=lr*gwr[i];
    }
    for (int i=0;i<H*m->in_dim;++i) m->Win[i]-=lr*gWin[i]; for (int j=0;j<H;++j) m->bin[j]-=lr*gbin[j];

    // free
    free(gWout); free(dG); free(dh1); free(dz1); free(gA); free(gb); free(dm1); free(gWr); free(dh0); free(dz0); free(gWin); free(gbin);
}

int rginn_save(const RGINNModel *m, const char *path){
    FILE *f=fopen(path,"w"); if(!f) return -1;
    fprintf(f, "%d %d\n", m->in_dim, m->hidden_dim);
    // Win
    for(int j=0;j<m->hidden_dim;++j){
        for(int i=0;i<m->in_dim;++i){
            fprintf(f, "%g%s", m->Win[j*m->in_dim + i], (i+1==m->in_dim)?"\n":" ");
        }
    }
    // bin
    for(int j=0;j<m->hidden_dim;++j){ fprintf(f, "%g%s", m->bin[j], (j+1==m->hidden_dim)?"\n":" "); }
    // Wr per relation
    fprintf(f, "%d\n", m->num_rel);
    for(int r=0;r<m->num_rel;++r){
        for(int j=0;j<m->hidden_dim;++j){
            for(int k=0;k<m->hidden_dim;++k){
                fprintf(f, "%g%s", m->Wr[r*m->hidden_dim*m->hidden_dim + j*m->hidden_dim + k], (k+1==m->hidden_dim)?"\n":" ");
            }
        }
    }
    // A
    for(int j=0;j<m->hidden_dim;++j){
        for(int k=0;k<m->hidden_dim;++k){ fprintf(f, "%g%s", m->A[j*m->hidden_dim + k], (k+1==m->hidden_dim)?"\n":" "); }
    }
    // b
    for(int j=0;j<m->hidden_dim;++j){ fprintf(f, "%g%s", m->b[j], (j+1==m->hidden_dim)?"\n":" "); }
    // Wout and bout
    for(int j=0;j<m->hidden_dim;++j){ fprintf(f, "%g%s", m->Wout[j], (j+1==m->hidden_dim)?"\n":" "); }
    fprintf(f, "%g\n", m->bout);
    fclose(f);
    return 0;
}

int rginn_load(RGINNModel *m, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    
    int in_dim, hidden_dim;
    if (fscanf(f, "%d %d", &in_dim, &hidden_dim) != 2) {
        fclose(f);
        return -1;
    }
    
    // Free existing model if any and reinitialize
    rginn_free(m);
    
    m->in_dim = in_dim;
    m->hidden_dim = hidden_dim;
    m->Win = (double*)calloc((size_t)hidden_dim * in_dim, sizeof(double));
    m->bin = (double*)calloc((size_t)hidden_dim, sizeof(double));
    
    // Read Win
    for (int j = 0; j < hidden_dim; ++j) {
        for (int i = 0; i < in_dim; ++i) {
            if (fscanf(f, "%lf", &m->Win[j*in_dim + i]) != 1) {
                fclose(f);
                rginn_free(m);
                return -1;
            }
        }
    }
    
    // Read bin
    for (int j = 0; j < hidden_dim; ++j) {
        if (fscanf(f, "%lf", &m->bin[j]) != 1) {
            fclose(f);
            rginn_free(m);
            return -1;
        }
    }
    
    // Read num_rel
    int num_rel;
    if (fscanf(f, "%d", &num_rel) != 1) {
        fclose(f);
        rginn_free(m);
        return -1;
    }
    m->num_rel = num_rel;
    
    m->Wr = (double*)calloc((size_t)num_rel * hidden_dim * hidden_dim, sizeof(double));
    
    // Read Wr
    for (int r = 0; r < num_rel; ++r) {
        for (int j = 0; j < hidden_dim; ++j) {
            for (int k = 0; k < hidden_dim; ++k) {
                if (fscanf(f, "%lf", &m->Wr[r*hidden_dim*hidden_dim + j*hidden_dim + k]) != 1) {
                    fclose(f);
                    rginn_free(m);
                    return -1;
                }
            }
        }
    }
    
    m->A = (double*)calloc((size_t)hidden_dim * hidden_dim, sizeof(double));
    m->b = (double*)calloc((size_t)hidden_dim, sizeof(double));
    
    // Read A
    for (int j = 0; j < hidden_dim; ++j) {
        for (int k = 0; k < hidden_dim; ++k) {
            if (fscanf(f, "%lf", &m->A[j*hidden_dim + k]) != 1) {
                fclose(f);
                rginn_free(m);
                return -1;
            }
        }
    }
    
    // Read b
    for (int j = 0; j < hidden_dim; ++j) {
        if (fscanf(f, "%lf", &m->b[j]) != 1) {
            fclose(f);
            rginn_free(m);
            return -1;
        }
    }
    
    m->Wout = (double*)calloc((size_t)hidden_dim, sizeof(double));
    
    // Read Wout
    for (int j = 0; j < hidden_dim; ++j) {
        if (fscanf(f, "%lf", &m->Wout[j]) != 1) {
            fclose(f);
            rginn_free(m);
            return -1;
        }
    }
    
    // Read bout
    if (fscanf(f, "%lf", &m->bout) != 1) {
        fclose(f);
        rginn_free(m);
        return -1;
    }
    
    // Set default values
    m->eps = 0.0;
    m->lr = 1e-3;
    m->num_layers = 1;
    
    fclose(f);
    return 0;
}
