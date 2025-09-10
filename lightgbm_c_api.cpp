// AQD LightGBM inference using official LightGBM C API
#include "lightgbm_c_api.h"
#include <LightGBM/c_api.h>
#include <mutex>
#include <vector>
#include <string>
#include <unordered_map>

static std::mutex g_mutex;
static BoosterHandle g_booster = nullptr;
static std::vector<std::string> g_feature_names;
static std::unordered_map<std::string, int> g_name2idx;
static int g_num_features = 0;

extern "C" int aqd_lgb_load_model(const char *model_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_booster) {
        LGBM_BoosterFree(g_booster);
        g_booster = nullptr;
    }
    if (!model_path) return 0;
    int num_total_model = 0;
    if (LGBM_BoosterCreateFromModelfile(model_path, &num_total_model, &g_booster) != 0 || !g_booster) {
        return 0;
    }
    // Load feature names from booster
    g_feature_names.clear();
    g_name2idx.clear();
    g_num_features = 0;
    if (LGBM_BoosterGetNumFeature(g_booster, &g_num_features) != 0 || g_num_features <= 0) {
        LGBM_BoosterFree(g_booster); g_booster = nullptr; return 0;
    }
    std::vector<size_t> out_buffer_len((size_t)g_num_features, 0);
    const size_t per_name_buf = 256; // should be enough for most names
    std::vector<std::vector<char>> buffers((size_t)g_num_features, std::vector<char>(per_name_buf));
    std::vector<char*> out_strs((size_t)g_num_features);
    for (int i = 0; i < g_num_features; ++i) out_strs[(size_t)i] = buffers[(size_t)i].data();
    int out_len = 0;
    if (LGBM_BoosterGetFeatureNames(g_booster, g_num_features, &out_len, per_name_buf, out_buffer_len.data(), out_strs.data()) != 0 || out_len <= 0) {
        // Fallback: keep empty names but still allow prediction by index
        g_feature_names.resize((size_t)g_num_features);
        for (int i = 0; i < g_num_features; ++i) {
            g_feature_names[(size_t)i] = std::to_string(i);
            g_name2idx[g_feature_names[(size_t)i]] = i;
        }
    } else {
        for (int i = 0; i < out_len; ++i) {
            std::string s(out_strs[(size_t)i]);
            g_feature_names.push_back(s);
            g_name2idx[s] = i;
        }
    }
    return 1;
}

extern "C" double aqd_lgb_predict_named(const char **names, const double *values, int n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_booster || !names || !values || n <= 0 || g_num_features <= 0) return 0.0;
    std::vector<double> row((size_t)g_num_features, 0.0);
    for (int i = 0; i < n; ++i) {
        if (!names[i]) continue;
        auto it = g_name2idx.find(names[i]);
        if (it != g_name2idx.end()) {
            row[(size_t)it->second] = values[i];
        }
    }
    // Single-row prediction (raw score)
    int64_t out_len = 0;
    double out = 0.0;
    if (LGBM_BoosterPredictForMatSingleRow(g_booster,
                                           row.data(),
                                           C_API_DTYPE_FLOAT64,
                                           g_num_features,
                                           /*is_row_major=*/1,
                                           C_API_PREDICT_RAW_SCORE,
                                           /*start_iteration=*/0,
                                           /*num_iteration=*/-1,
                                           /*parameter=*/"",
                                           &out_len,
                                           &out) != 0 || out_len <= 0) {
        return 0.0;
    }
    return out;
}

extern "C" void aqd_lgb_unload(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_booster) {
        LGBM_BoosterFree(g_booster);
        g_booster = nullptr;
    }
    g_feature_names.clear();
    g_name2idx.clear();
    g_num_features = 0;
}
