#include "gnn_c_api.h"
#include "rginn.h"
#include <mutex>

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

