#include "gnn_c_api.h"
#include "rginn.h"
#include <mutex>
#include <vector>

static std::mutex g_mutex;
static RGINNModel g_model;
static int g_loaded = 0;

extern "C" int aqd_gnn_load_model(const char *model_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loaded) {
        rginn_free(&g_model);
        g_loaded = 0;
    }
    rginn_init(&g_model, 16, 32, 1, 3, 1e-3);
    if (rginn_load(&g_model, model_path) != 0) {
        rginn_free(&g_model);
        return 0;
    }
    g_loaded = 1;
    return 1;
}

extern "C" int aqd_gnn_is_loaded(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_loaded;
}

extern "C" void aqd_gnn_unload(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loaded) {
        rginn_free(&g_model);
        g_loaded = 0;
    }
}

extern "C" double aqd_gnn_predict(int N, int in_dim, const double *X,
                                  int num_rel, const int *indptr, const int *indices) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return 0.0;
    if (in_dim != g_model.in_dim || num_rel != g_model.num_rel) {
        // Basic dim check; return neutral
        return 0.0;
    }
    RGGraph g;
    g.N = N; g.in_dim = in_dim; g.X = X; g.num_rel = num_rel; g.indptr = indptr; g.indices = indices;
    const int H = g_model.hidden_dim;
    std::vector<double> h0((size_t)N*H), m1((size_t)N*H), h1((size_t)N*H), gr(H);
    double y = rginn_forward(&g_model, &g, h0.data(), m1.data(), h1.data(), gr.data());
    return y;
}
