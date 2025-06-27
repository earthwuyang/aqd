/*  lightgbm_model.cpp  --------------------------------------- */
#include <LightGBM/c_api.h>
#include <cerrno>
#include <iostream>
#include <thread>

#include "model_iface.hpp"

#if __cplusplus < 201402L   // 只有 C++14 之前的标准才进这里
#include <memory>
#include <utility>

namespace std {
    template <typename T, typename... Args>
    inline std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

template <typename T>
inline T clamp_val(const T& v, const T& lo, const T& hi)
{
    return std::max(lo, std::min(v, hi));
}

class LGBModel : public IModel {
    std::string booster_; //"goss" or "rf"
public:
    explicit LGBModel(std::string booster = "goss")
        : booster_(std::move(booster)) {}

    /*  ------------------------------------------------------------
    LGBModel::train  –  re-worked to honour the `va` set
    ------------------------------------------------------------ */
    void train(const std::vector<Sample>& DS,
            const std::vector<Sample>& va,
            const std::string& model_path,
            const TrainOpt&       opt) override
    {
        /* ───── constants ───────────────────────────────────────── */
        constexpr double GAP_THR      = 0.0;
        constexpr float  DECISION     = 0.0f;         // τ*
        constexpr double Q_SMALL      = 1e4;
        constexpr double Q_LARGE      = 5e4;
        constexpr int    EARLY_STOP   = 120;          // rounds w/out improvement
        constexpr int    PRINT_EVERY  = 10;
        int num_threads = 20;
        bool skip_train = opt.skip_train;
        // double sumsample = opt.subsample;
        double colsample = opt.colsample;
        double lr = opt.lr;
        // double max_dpeth = opt.max_depth;
        int num_trees = opt.trees;

        /* ───── build training subset (identical logic) ─────────── */
        std::vector<const Sample*> S;
        for (const auto& s : DS)
            if (s.hybrid_pred != -1 && s.fann_pred != -1 &&
                s.row_t > 0 && s.col_t > 0 &&
                std::fabs(s.row_t - s.col_t) >= GAP_THR)
                S.push_back(&s);

        const int N = static_cast<int>(S.size());
        if (!N) { logE("no sample has full information"); return; }

        if (num_threads <= 0)
            num_threads = std::max(1u, std::thread::hardware_concurrency());

        /* ───── training matrices / label / weight ──────────────── */
        /* ───── training matrices / label / weight ──────────────── */
        std::vector<float> X(size_t(N) * NUM_FEATS), y(N), w(N);
        {
            /* 1. 计算基础正/负样本权；正样本略降，负样本略升  */
            double P = 0, N0 = 0;
            for (auto p : S) p->label ? ++P : ++N0;
            const double w_pos = P  ? 0.90 * N / (2 * P)  : 1.0;
            const double w_neg = N0 ? 1.10 * N / (2 * N0) : 1.0;

            for (int i = 0; i < N; ++i) {
                const Sample& s = *S[i];

                /* ---------- 复制特征 & 标签 ---------- */
                std::copy(s.feat.begin(), s.feat.end(),
                        X.begin() + i * NUM_FEATS);

                const double rt = std::max(s.row_t, EPS_RUNTIME);
                const double ct = std::max(s.col_t, EPS_RUNTIME);
                y[i] = static_cast<float>(std::log(rt) - std::log(ct));

                /* ---------- (A) 基础权重 ---------- */
                const double base = (s.label ? w_pos : w_neg);

                /* 0) Δ 放大（保留） */
                double ln_gap = clamp_val(std::fabs(std::log(rt) - std::log(ct)), 0.0, 2.0);
                double w_i = base * (1.0 + 0.4 * ln_gap);

                /* 1) 提速/减速对称调节 (slow_amp≤2.0) */
                double rel_gain = (s.row_t - s.col_t) / std::max(1e-6, s.row_t);
                if (s.label) {
                    double damp = clamp_val(rel_gain / 0.5, 0.0, 1.0);
                    w_i *= 0.6 + 0.4 * damp;                 // 0.6-1.0×base
                } else {
                    double slow_amp = clamp_val(s.col_t / std::max(1e-6, s.row_t),
                                                1.0, 2.0);
                    w_i *= slow_amp;
                }

                /* 2) 巨型查询只对负样本放大（略降系数） */
                if (!s.label) {
                    if      (s.qcost > 3e7) w_i *= 1.8;
                    else if (s.qcost > 1e7) w_i *= 1.4;
                    else if (s.qcost > 2e6) w_i *= 1.2;
                }

                /* 3) fan / ratio / pc_rc 调整系数整体 ×0.5 */
                double fan   = std::min<double>(3.0, s.feat[22]);
                double ratio = std::min<double>(3.0, s.feat[60]);
                if (fan   > 2.0) w_i += 0.05 * base * (fan   - 2.0);
                if (ratio > 1.2) w_i += 0.04 * base * (ratio - 1.2);

                double pc_rc = std::exp(std::min<double>(2.0, s.feat[25]));
                if (pc_rc > 2.0 && s.qcost < 1e4) w_i += 0.20 * base * (pc_rc - 1.8);

                /* 4) “窄列+点查” 轻量放大 */
                double point_ratio  = s.feat[104];
                double narrow_ratio = s.feat[105];
                if (!s.label && point_ratio > 0.70 && narrow_ratio > 0.80)
                    w_i *= 1.2;

                /* 5) 正样本额外奖励 */
                if (s.label && rel_gain > 0.1) {
                    double bonus = clamp_val(rel_gain / 0.5, 0.0, 1.0);
                    w_i *= 1.0 + 0.3 * bonus;
                }

                /* 6) 软剪裁 hi≤8× */
                const double LO = 0.05 * base, HI = 8.0 * base;
                w_i = clamp_val(w_i, LO, HI);

                w[i] = static_cast<float>(w_i);
            }

            // for (int i = 0; i < N; ++i) {
            //     const Sample& s = *S[i];

            //     /* feature */
            //     std::copy(s.feat.begin(), s.feat.end(), X.begin() + i * NUM_FEATS);

            //     /* regression target Δ = ln(rt/ct) */
            //     double rt = std::max(s.row_t, EPS_RUNTIME);
            //     double ct = std::max(s.col_t, EPS_RUNTIME);
            //     y[i] = float(std::log(rt) - std::log(ct));

            //     /* weight（同旧版） */
            //     double base = (s.label ? w_pos : w_neg);
            //     double q_ln = std::log1p(s.qcost);
            //     double w_i  = base * (1.0 + 0.20 * q_ln);

            //     double fan   = std::min<double>(3.0, s.feat[22]);
            //     double ratio = std::min<double>(3.0, s.feat[60]);
            //     if (fan   > 1.3) w_i += 0.6 * base * (fan   - 1.2);
            //     if (ratio > 1.1) w_i += 0.4 * base * (ratio - 1.0);

            //     double pc_rc = std::exp(std::min<double>(2.0, s.feat[25]));
            //     if (pc_rc > 2.0 && s.qcost < 1e4) w_i += 0.8 * base * (pc_rc - 1.5);
            //     if (fan > 2.5 && s.qcost < 8e3)   w_i += 2.0 * base;

            //     auto rel_gap = [](double fast, double slow) {
            //         double g = std::log(slow / std::max(1e-6, fast));
            //         return clamp(g / 3.0, 0.0, 1.0);
            //     };
            //     if (s.label == 1 && s.qcost < Q_SMALL)
            //         w_i *= (1.0 + rel_gap(s.col_t, s.row_t));
            //     if (s.label == 0 && s.qcost > Q_LARGE)
            //         w_i *= (1.0 + rel_gap(s.row_t, s.col_t));

            //     w[i] = static_cast<float>(w_i);
            // }
        }


        /* ───── validation matrices (simpler weighting = 1) ─────── */
        std::vector<float> X_val, y_val;
        if (!va.empty()) {
            const int Nv = static_cast<int>(va.size());
            X_val.resize(size_t(Nv) * NUM_FEATS);
            y_val.resize(Nv);

            for (int i = 0; i < Nv; ++i) {
                const Sample& s = va[i];
                std::copy(s.feat.begin(), s.feat.end(),
                        X_val.begin() + i * NUM_FEATS);

                double rt = std::max(s.row_t, EPS_RUNTIME);
                double ct = std::max(s.col_t, EPS_RUNTIME);
                y_val[i]  = float(std::log(rt) - std::log(ct));
            }
        }

        /* ───── LightGBM handles ─────────────────────────────────── */
        DatasetHandle dtrain = nullptr, dvalid = nullptr;
        BoosterHandle booster = nullptr;

        /* ---------------------------------------------------------- */
        if (!skip_train) {
            /* create train set */
            if (LGBM_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT32,
                                        N, NUM_FEATS, 1,
                                        "", nullptr, &dtrain))
            { logE("DatasetCreate failed"); return; }

            LGBM_DatasetSetField(dtrain, "label",  y.data(), N, C_API_DTYPE_FLOAT32);
            LGBM_DatasetSetField(dtrain, "weight", w.data(), N, C_API_DTYPE_FLOAT32);

            /* optional valid set */
            if (!X_val.empty()) {
                const int Nv = static_cast<int>(va.size());
                if (LGBM_DatasetCreateFromMat(X_val.data(), C_API_DTYPE_FLOAT32,
                                            Nv, NUM_FEATS, 1,
                                            "", dtrain, &dvalid))
                { logE("DatasetCreate(valid) failed"); return; }
                LGBM_DatasetSetField(dvalid, "label", y_val.data(),
                                    Nv, C_API_DTYPE_FLOAT32);
            }

            std::string param = "boosting=" + booster_; 
            if (booster_ == "goss") {
                param += " top_rate=0.2 other_rate=0.05";
            }
            else if (booster_ == "rf") {
                double bag_frac = (opt.subsample > 0.0 && opt.subsample < 1.0)
                                    ? opt.subsample : 0.63;
                param += " bagging_fraction=" + std::to_string(bag_frac)
                    + " bagging_freq=1";              // must be >0
            }
            // param +=
            //     " objective=fair fair_c=1.2 metric=l1 max_depth=18"
            //     " max_bin=127 num_leaves=756 min_data_in_leaf=40"
            //     " learning_rate="    + std::to_string(lr * 0.75) +
            //     " feature_fraction=" + std::to_string(colsample) +
            //     " lambda_l2=1.0 num_threads=" + std::to_string(num_threads) +
            //     " verbosity=-1";

            param +=
                " objective=regression_l2 metrics=l2 max_depth=22"
                " max_bin=127 num_leaves=1536 min_data_in_leaf=40"
                " learning_rate="    + std::to_string(lr * 0.75) +
                " feature_fraction=" + std::to_string(colsample) +
                " lambda_l2=1.0 num_threads=" + std::to_string(num_threads) +
                " verbosity=-1";

            /* monotone constraints (same as before) */
            {
                std::string mono; mono.reserve(NUM_FEATS * 2);
                for (int i = 0; i < NUM_FEATS; ++i) {
                    int m = 0;
                    if (i == 22 || i == 23 || i == 40 || i == 60 || i == 101)  m = +1; // 正相关
                    if (i == 65 || i == 97 || i == 98 || i == 99 || i == 100   // 旧负相关
                        || i == 104 || i == 105)          // 新负相关
                        m = -1;
                    mono += std::to_string(m) + (i + 1 < NUM_FEATS ? "," : "");
                }
                param += " monotone_constraints=" + mono;
            }

            


            /* create booster */
            if (LGBM_BoosterCreate(dtrain, param.c_str(), &booster))
            { logE("BoosterCreate failed"); return; }

            if (dvalid)
                LGBM_BoosterAddValidData(booster, dvalid);   // returns 0 on success

            /* ------------ training loop with early-stop ------------ */
            const int MAX_ITERS = num_trees;
            double best_metric  = std::numeric_limits<double>::max();
            int    best_iter    = -1;

            for (int it = 0, fin; it < MAX_ITERS; ++it) {
                LGBM_BoosterUpdateOneIter(booster, &fin);

                if (dvalid && (it + 1) % PRINT_EVERY == 0) {
                    /* metric on the first valid dataset */
                    int      n_eval = 0;
                    LGBM_BoosterGetEvalCounts(booster, &n_eval);

                    std::vector<double> res(n_eval);
                    int out_len = 0;
                    LGBM_BoosterGetEval(booster, 0, &out_len, res.data());

                    double l1 = res.empty() ? 0.0 : res[0];
                    progress("l1=" + std::to_string(l1), it + 1, MAX_ITERS);

                    if (l1 < best_metric - 1e-7) {      // tiny epsilon
                        best_metric = l1; best_iter = it;
                    } else if (it - best_iter >= EARLY_STOP) {
                        logI("Early stop at iter " + std::to_string(it + 1) +
                            "  best=" + std::to_string(best_iter + 1));
                        break;
                    }
                }
            }
            std::cerr << '\n';

            /* save model */
            LGBM_BoosterSaveModel(booster, 0, -1, 0, model_path.c_str());
            logI("Model saved → " + model_path);
        }
        else {
            /* inference-only path */
            int iters = 0;
            if (LGBM_BoosterCreateFromModelfile(model_path.c_str(),
                                                &iters, &booster))
            { logE("model load failed"); return; }
        }

        /* ───── quick evaluation on training subset ─────────────── */
        {
            std::vector<Sample> eval_set;
            eval_set.reserve(N);
            for (auto p : S) eval_set.push_back(*p);

            std::vector<int> pred = predict(model_path, eval_set, DECISION);

            long    TP=0, FP=0, TN=0, FN=0;
            double  r_row=0, r_col=0, r_opt=0,
                    r_lgb=0, r_fann=0, r_hyp=0, r_costthr=0;   // ← 新名字

            for (int i = 0; i < N; ++i) {
                const Sample& s = eval_set[i];

                /* ---------- LightGBM ---------- */
                bool col_lgb = pred[i];
                r_lgb += col_lgb ? s.col_t : s.row_t;

                /* ---------- Hybrid Opt -------- */
                bool col_hyp = (s.hybrid_pred == 1);
                r_hyp += col_hyp ? s.col_t : s.row_t;

                /* ---------- FANN -------------- */
                bool col_fann = (s.fann_pred == 1);
                r_fann += col_fann ? s.col_t : s.row_t;

                /* ---------- Cost-Threshold ---- */
                bool use_row = (s.qcost < COST_THR);           // ← 关键比较
                r_costthr += use_row ? s.row_t : s.col_t;

                /* ---------- 统计准确率 --------- */
                bool col_lbl = s.label;                        // 1 = 列更快
                (col_lgb ? (col_lbl?++TP:++FP) : (col_lbl?++FN:++TN));

                r_row += s.row_t;  r_col += s.col_t;
                r_opt += std::min(s.row_t, s.col_t);
            }

            auto safe_div = [](double a, double b){
                return b ? a / b : 0.0;
            };

            auto avg = [&](double v){ return v / N; };
            std::cout << "*** EVALUATED ON " << N << " SAMPLES ***\n"
                    << "Acc="      << double(TP+TN)/N
                    << "  Precision=" << safe_div(TP,TP+FP)
                    << "  Recall="    << safe_div(TP,TP+FN)
                    << "  BalAcc=" << 0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
                    << "  F1="     << (TP?2.0*TP/(2*TP+FP+FN):0.) << '\n'
                    << "Row-only:        " << avg(r_row)      << '\n'
                    << "Col-only:        " << avg(r_col)      << '\n'
                    << "Cost-Threshold:  " << avg(r_costthr)  << '\n'
                    << "HybridOpt:       " << avg(r_hyp)      << '\n'
                    << "FANN:            " << avg(r_fann)     << '\n'
                    << "LightGBM:        " << avg(r_lgb)      << '\n'
                    << "Oracle (min):    " << avg(r_opt)      << '\n';
        }

        /* ───── cleanup ───────────────────────────────────────────── */
        if (!skip_train) {
            LGBM_DatasetFree(dtrain);
            if (dvalid) LGBM_DatasetFree(dvalid);
        }
        LGBM_BoosterFree(booster);
    }


    std::vector<int> predict(const std::string& path,
                             const std::vector<Sample>& DS,
                             float tau) const override
    {
        const int N = DS.size();
        // for (int i=0;i<N;i++) {
        //     for (int j=0;j<NUM_FEATS;j++)   
        //         printf("feat[%d]=%f\n",j,DS[i].feat[j]);
        // }
        vector<float> X(size_t(N) * NUM_FEATS);
        for (int i = 0; i < N; ++i)
            std::copy(DS[i].feat.begin(), DS[i].feat.end(),
                    X.begin() + i * NUM_FEATS);

        BoosterHandle booster = nullptr; int iters = 0;
        int err = LGBM_BoosterCreateFromModelfile(path.c_str(),
                                            &iters, &booster);

        if (err != 0 || booster == nullptr) {
            logE("LightGBM load failed: " + std::string(LGBM_GetLastError()));
            throw std::runtime_error("cannot load model");
        }

        vector<double> pred(N); int64_t out_len = 0;
        LGBM_BoosterPredictForMat(
            booster, X.data(), C_API_DTYPE_FLOAT32,
            N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
            -1, 0, "", &out_len, pred.data());
        LGBM_BoosterFree(booster);

        vector<int> bin(N);
        for (int i = 0; i < N; ++i) bin[i] = pred[i] > tau;

        return bin;
    }

    double bal_acc(const std::string& model_path,
                   const std::vector<Sample>& DS,
                   float tau) const override
    {
        if (DS.empty()) return 0.0;

        std::vector<int> pred = predict(model_path, DS, tau);

        /* 2) 统计混淆矩阵 */
        long TP = 0, TN = 0, FP = 0, FN = 0;
        for (size_t i = 0; i < DS.size(); ++i) {
            bool col = pred[i];            // 1 ⇒ 预测列存
            bool gt  = DS[i].label;        // 1 ⇒ 列存更快 (真实标签)

            if      ( col &&  gt) ++TP;
            else if ( col && !gt) ++FP;
            else if (!col &&  gt) ++FN;
            else                   ++TN;
        }

        /* 3) 计算 Balanced Accuracy = (TPR + TNR) / 2 */
        double tpr = (TP + FN) ? double(TP) / (TP + FN) : 0.0;
        double tnr = (TN + FP) ? double(TN) / (TN + FP) : 0.0;
        return 0.5 * (tpr + tnr);
    }
};

/* factory – so main() can remain agnostic */
std::unique_ptr<IModel> make_lightgbm(const std::string& booster) { 
    return std::make_unique<LGBModel>(booster); 
}
