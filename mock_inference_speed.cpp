/********************************************************************
 * mock_measure_inference_speed.cpp
 *   ─ 测 LightGBM 单条推断耗时（无任何数据库、无 JSON 解析）
 *
 * Build:
 *   g++ -O3 -std=c++17 \
 *       mock_measure_inference_speed.cpp -o measure_speed \
 *       -I$LIGHTGBM/include -L$LIGHTGBM/lib -l_lightgbm -pthread
 *
 * Run (默认 1000 次循环)：
 *   ./measure_speed --model=lgb_model.txt --repeat=5000
 *******************************************************************/
#include <bits/stdc++.h>
#include <LightGBM/c_api.h>

using hrc = std::chrono::high_resolution_clock;

/* ───── CLI helpers ───── */
static std::string get_arg(int argc, char* argv[],
                           const std::string& key,
                           const std::string& def = "")
{
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind(key + '=', 0) == 0) return a.substr(key.size() + 1);
    }
    return def;
}
static void die(const std::string& s){ std::cerr<<"[ERR] "<<s<<'\n'; std::exit(1); }

/* ───── main ───── */
int main(int argc, char* argv[])
{
    const std::string model   = get_arg(argc, argv, "--model",  "lgb_fold1.txt");
    const std::size_t repeat  = std::stoul(get_arg(argc, argv, "--repeat", "100"));

    if (model.empty()) die("usage: --model=<lightgbm_model.txt>");

    /* 1) 载入模型 --------------------------------------------------- */
    BoosterHandle booster = nullptr; int dummy = 0;
    if (LGBM_BoosterCreateFromModelfile(model.c_str(), &dummy, &booster))
        die("cannot load model "+model);

    /* 2) 构造一条虚拟特征向量 -------------------------------------- */
    constexpr int NUM_FEATS = 102;
    float  feat[NUM_FEATS];
    std::iota(feat, feat + NUM_FEATS, 0.f);    // 0,1,2,…  —— 值无所谓
    double pred[1]; int64_t out_len = 0;

    /*   —— 预热一次，排除 lazy init 开销 —— */
    LGBM_BoosterPredictForMat(
        booster, feat, C_API_DTYPE_FLOAT32,
        /*nrow=*/1, NUM_FEATS, /*is_row_major=*/1,
        C_API_PREDICT_NORMAL,
        -1, 0, "", &out_len, pred);

    /* 3) 真正计时 --------------------------------------------------- */
    auto t0 = hrc::now();
    for (std::size_t i = 0; i < repeat; ++i) {
        LGBM_BoosterPredictForMat(
            booster, feat, C_API_DTYPE_FLOAT32,
            1, NUM_FEATS, 1,
            C_API_PREDICT_NORMAL,
            -1, 0, "", &out_len, pred);
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>
              (hrc::now() - t0).count();

    /* 4) 输出 ------------------------------------------------------- */
    double avg_us = double(us) / repeat;
    std::cout << "[INFO] total iterations : " << repeat  << '\n'
              << "► Avg time : " << std::fixed << std::setprecision(2)
              << avg_us << " µs (" << avg_us / 1000.0 << " ms)\n";

    LGBM_BoosterFree(booster);
    return 0;
}
