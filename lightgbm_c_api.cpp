#include "lightgbm_c_api.h"
#include "lightgbm_inference.h"
#include <mutex>
#include <vector>
#include <string>

using lightgbm_inference::LightGBMPredictor;

static std::mutex g_mutex;
static LightGBMPredictor *g_predictor = nullptr;

extern "C" int aqd_lgb_load_model(const char *model_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_predictor) {
        delete g_predictor;
        g_predictor = nullptr;
    }
    try {
        g_predictor = new LightGBMPredictor();
        if (!g_predictor->load_model(model_path)) {
            delete g_predictor; g_predictor = nullptr; return 0;
        }
        return 1;
    } catch (...) {
        if (g_predictor) { delete g_predictor; g_predictor = nullptr; }
        return 0;
    }
}

extern "C" double aqd_lgb_predict_named(const char **names, const double *values, int n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_predictor || !names || !values || n <= 0) return 0.0;
    std::vector<std::string> fnames;
    fnames.reserve(n);
    std::map<std::string,double> fmap; // rely on predictor’s name->index mapping
    for (int i=0;i<n;++i) {
        if (names[i]) fmap[names[i]] = values[i];
    }
    try {
        return g_predictor->predict(fmap);
    } catch (...) {
        return 0.0;
    }
}

extern "C" void aqd_lgb_unload(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_predictor) { delete g_predictor; g_predictor = nullptr; }
}

