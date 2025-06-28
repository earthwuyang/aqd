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
            const TrainOpt&       opt, const std::unordered_map<std::string,double>& DIR_W) override
    {
        /* ───── constants ───────────────────────────────────────── */
        constexpr double GAP_THR      = 0.0;
        constexpr float  DECISION     = 0.0f;         // τ*
        constexpr double Q_SMALL      = 1e4;
        constexpr double Q_LARGE      = 5e4;
        constexpr int    EARLY_STOP   = 100;          // rounds w/out improvement
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
        std::vector<const Sample*> S_focus; // 二次训练用焦点样本
        
        for (const auto& s : DS) {
            bool in_main = false;
            if (s.hybrid_pred != -1 && s.fann_pred != -1 &&
                s.row_t > 0 && s.col_t > 0 &&
                std::fabs(s.row_t - s.col_t) >= GAP_THR) {
                    S.push_back(&s);
                    in_main = true;
                }
                
            if (in_main) {
                /* ------------ 把“难点子集”另外收集 ------------ */
                bool col_faster =  s.col_t < s.row_t;   // 正样本
                if ( (s.qcost < Q_SMALL &&  col_faster) ||
                    (s.qcost > Q_LARGE && !col_faster) )
                    S_focus.push_back(&s);
            }
            
        }
            

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
            double w_pos = 1.30 * N / (2 * P);   // ↑ 1.3
            double w_neg = 0.90 * N / (2 * N0);  // ↓ 0.9

            for (int i = 0; i < N; ++i) {
                const Sample& s = *S[i];

                /* ---------- 复制特征 & 标签 ---------- */
                std::copy(s.feat.begin(), s.feat.end(),
                        X.begin() + i * NUM_FEATS);

                const double rt = std::max(s.row_t, EPS_RUNTIME);
                const double ct = std::max(s.col_t, EPS_RUNTIME);
                // y[i] = static_cast<float>(std::log(rt) - std::log(ct));
                double diff = std::log(rt) - std::log(ct);
                const double EPS = 0.05;           // 约 5 % 差距阈值
                if (std::fabs(diff) < EPS) diff *= 0.3;   // 压缩梯度
                y[i] = static_cast<float>(diff);

                /* ---------- (A) 基础正负样本权重 -------------------------------- */                
                const double base = s.label ? w_pos : w_neg;

                /* ---------- (B) Δ(gap) 放大 ---------------------------------- */
                double ln_gap = clamp_val(std::fabs(std::log(rt) - std::log(ct)), 0.0, 2.0);
                double w_i    = base * (1.0 + 0.4 * ln_gap);

                /* ---------- (C) 数据集级 √N 反比权重 -------------------------- */
                w_i *= DIR_W.at(s.dir_tag);

                // /* ---------- (D) 提速 / 减速 非对称调节 ----------------------- */
                // double rel_gap = (s.row_t - s.col_t) / std::max(1e-6, std::min(rt, ct));
                // if (s.label) {                                    // 列更快
                //     w_i *= 1.0 + 0.4 * clamp_val(rel_gap / 0.5, 0.0, 1.0);
                // } else {                                          // 行更快
                //     double slow_ratio = clamp_val(ct / std::max(1e-6, rt), 1.0, 3.0);
                //     w_i *= 1.0 + 0.5 * (slow_ratio - 1.0);
                // }

                // /* ---------- (E) qcost 调节（2 档，只奖励正样本） -------------- */
                // if (s.label) {
                //     if      (s.qcost > 3e7) w_i *= 1.25;
                //     else if (s.qcost > 1e7) w_i *= 1.10;
                // }

                // /* ---------- (F) fan / ratio / pc_rc 微调 ---------------------- */
                // double fan   = std::min<double>(3.0, s.feat[22]);
                // double ratio = std::min<double>(3.0, s.feat[60]);
                // if (fan   > 2.0) w_i *= 1.00 + 0.02 * (fan   - 2.0);
                // if (ratio > 1.2) w_i *= 1.00 + 0.02 * (ratio - 1.2);

                // /* ========== ❶ 新 G：行窄 + 点查（Row-win 信号） =============== */
                // double point_ratio  = s.feat[104];   // ref + eq_ref 占比
                // double narrow_ratio = s.feat[105];   // ≤16 B 列占比
                // bool   point_and_narrow = (point_ratio > 0.70 && narrow_ratio > 0.80);
                // if (!s.label && point_and_narrow)
                //     w_i *= 1.8;                      // 原 1.6 → 1.8，放大一点

                // /* ========== ❷ 新 H：中等规模但 Row 更快 → 软放大 =============== *
                // * 逻辑：行存实际更快，且 qcost 不到 3 M（典型 OLTP / 点查）。     */
                // if (!s.label) {
                //     /* gap_lt0 为负值，-gap_lt0 越大说明 Row 优势越明显 */
                //     double gap_lt0 = std::log(rt) - std::log(ct);   // <0 ⇒ Row faster
                //     if (gap_lt0 < -0.2) {
                //         double bonus = clamp_val((-gap_lt0) / 1.5, 0.0, 2.0); // 上限 ×3
                //         w_i *= 100.0 + bonus;
                //     }
                // }

                // /* ---------- (L) 早期行数很小 + fan-out 介于 5-20 → 行存往往获胜 ---- */
                // double outer_rows_norm = s.feat[63];   // 已做 log 缩放
                // double fanout          = s.feat[22];   // 原始 fan-out_max
                // bool   late_fan        = (s.feat[97] > 0.4);   // 累计 fan-out 信号
                // bool   small_core      = (outer_rows_norm < 0.05);   // ≈ outerRows ≤ 3
                // bool   mid_fan_range   = (fanout > 5.0 && fanout < 20.0);

                // if (!s.label && small_core && mid_fan_range && late_fan) {
                //     w_i *= 10.0;            // 适度放大到够用即可
                // }


                // /* ---------- (G2) 稀有场景加权：多表但行更快 ------------------ */
                // int tbl_cnt = int(s.feat[107]);          // 你在 plan2feat 里写入的位置
                // if (!s.label && tbl_cnt > 20)            // 行存赢且表数>20
                //     w_i *= 1.30;

                // /* ---------- (H2) MIN/MAX 无 GROUP 且行存快 ------------------- */
                // bool has_agg_no_grp = (s.feat[108] > 0.5);   // 0/1 布尔
                // if (!s.label && has_agg_no_grp)
                //     w_i *= 1.40;

                // /* ---------- (I2) rows_probe / rows_outer 失衡 ---------------- */
                // double probe_ratio = s.feat[109];            // 已是 log_tanh 或原值
                // if (!s.label && probe_ratio < 0.5)           // 行快却探测行很少
                //     w_i *= 1.20;
                // if ( s.label && probe_ratio > 10.0)          // 列快却探测行很多
                //     w_i *= 1.20;


                // /* --- (J) fan-out 很大且列更快 → 放大列样本 -------------------- */
                // double fan_big = s.feat[22];      // fanout_max
                // if (s.label && fan_big > 30.0) {
                //     double bump = clamp_val((fan_big - 30.0) / 170.0, 0.0, 1.0); // 30→200 线性
                //     w_i *= 1.0 + bump;            // 最高 ×2
                // }

                /* ========== ❸ 新 I：统一软剪裁 (0.03–20 × base) =============== */
                // const double LO = 0.03 * base;
                // const double HI = 100.0 * base;
                // w_i = clamp_val(w_i, LO, HI);
                double log_lo = std::log(0.05 * (s.label ? w_pos : w_neg));
                double log_hi = std::log(20.0 * (s.label ? w_pos : w_neg));
                w_i = std::exp(clamp_val(std::log(w_i), log_lo, log_hi));


                w[i] = static_cast<float>(w_i);

                
            }
            for (size_t j = 0; j < w.size(); ++j)
                if (!std::isfinite(w[j]) || w[j] <= 0)
                    throw std::runtime_error("bad weight at "+std::to_string(j));
           
        }

        std::vector<float> Xf, yf, wf;
        const int NF = static_cast<int>(S_focus.size());
        if (NF > 0) {
            Xf.resize(size_t(NF) * NUM_FEATS);
            yf.resize(NF);
            wf.resize(NF);

            for (int i = 0; i < NF; ++i) {
                const Sample& s = *S_focus[i];
                std::copy(s.feat.begin(), s.feat.end(),
                        Xf.begin() + i * NUM_FEATS);

                double rt = std::max(s.row_t, EPS_RUNTIME);
                double ct = std::max(s.col_t, EPS_RUNTIME);
                yf[i] = float(std::log(rt) - std::log(ct));

                auto it = std::find(S.begin(), S.end(), S_focus[i]);
                if (it == S.end())
                    throw std::logic_error("focus sample not in main sample set");
                wf[i] = 2.0f * w[std::distance(S.begin(), it)];
            }
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
        DatasetHandle dfocus = nullptr;

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
                                    ? opt.subsample : 0.8;
                param += " bagging_fraction=" + std::to_string(bag_frac)
                    + " bagging_freq=5";              // must be >0
            }
            // param +=
            //     " objective=fair fair_c=1.2 metric=l1 max_depth=18"
            //     " max_bin=127 num_leaves=756 min_data_in_leaf=40"
            //     " learning_rate="    + std::to_string(lr * 0.75) +
            //     " feature_fraction=" + std::to_string(colsample) +
            //     " lambda_l2=1.0 num_threads=" + std::to_string(num_threads) +
            //     " verbosity=-1";

            param +=
                " objective=regression_l2 metrics=l2 max_depth=18"
                " max_bin=127 num_leaves=1024 min_data_in_leaf=1"
                " learning_rate="    + std::to_string(lr * 0.75) +
                " feature_fraction=" + std::to_string(colsample) +
                " lambda_l1=0.2 lambda_l2=3.0 num_threads=" + std::to_string(num_threads) +
                " verbosity=-1 min_sum_hessian_in_leaf=1e-6";

            /* monotone constraints (same as before) */
            {
                std::string mono; mono.reserve(NUM_FEATS * 2);
                for (int i = 0; i < NUM_FEATS; ++i) {
                    int m = 0;
                    if (i == 22 || i == 23 || i == 40 || i == 101)  m = +1; // 正相关
                    if (i == 65 || i == 99 || i == 100   // 旧负相关
                        || i == 104 || i == 105 || i==106)          // 新负相关
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
            LGBM_BoosterFree(booster);
            LGBM_DatasetFree(dtrain);
            if (dfocus) LGBM_DatasetFree(dfocus);
            if (dvalid) LGBM_DatasetFree(dvalid);
            
        }
        else {
            LGBM_BoosterFree(booster);
        }
        
    }


    std::vector<int> predict(const std::string& path,
                             const std::vector<Sample>& DS,
                             float tau) const override
    {
        const int N = DS.size();
        
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
