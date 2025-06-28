/*  lightgbm_model.cpp  --------------------------------------- */
#include <LightGBM/c_api.h>
#include <cerrno>
#include <iostream>
#include <thread>
#include <cmath>
#include <unordered_map>

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

extern std::unordered_map<std::string,double> g_db_size_gb;


template <typename T>
inline T clamp_val(const T& v, const T& lo, const T& hi)
{
    return std::max(lo, std::min(v, hi));
}

/* ★ util – sigmoid(raw_score) → [0,1] 概率 */
inline double sigmoid(double z){ return 1.0 / (1.0 + std::exp(-z)); }

/* ★ util – 断言简化 */
inline void chk(bool ok,const char*msg){
    if(!ok){ std::cerr<<"[FATAL] "<<msg<<'\n'; std::abort(); }
}

/* ---------- 自定义损失: 期望运行时间平方误差 ---------- *
 * L = ( p·ct + (1-p)·rt – g )²
 *   其中 p = σ(score),  g = min(rt,ct)
 * ---------------------------------------------------- */
struct TimeObjData {
    const double* rt;   // 行存真实时间 (len = num_data)
    const double* ct;   // 列存真实时间
};

int time_obj_grad_hess(const double* score,        /* raw scores      */
                       const double* /*label*/,    /* unused          */
                       int64_t        num_data,
                       double*        grad,        /* ← write here    */
                       double*        hess,        /* ← write here    */
                       void*          user_data)   /* ← TimeObjData*  */
{
    const auto* d = static_cast<const TimeObjData*>(user_data);

    /* ---- numeric safety constants --------------------------------- */
    constexpr double SCALE  = 20.0;   // magnify g & h (empirically 10-40)
    constexpr double G_CLIP = 50.0;   // final gradient clipping range
    constexpr double H_MIN  = 1e-1;   // Hessian floor  (post-scale)
    constexpr double H_MAX  = 1e3;    // Hessian ceiling

    for (int64_t i = 0; i < num_data; ++i) {

        /* --------- local shorthand ---------------------------------- */
        const double rt    = d->rt[i];
        const double ct    = d->ct[i];
        const double delta = ct - rt;                     // ct – rt

        const double s  = score[i];
        const double p  = 1.0 / (1.0 + std::exp(-s));     // σ(s)
        const double sp = p * (1.0 - p);                  // σ'(s)

        const double f_hat = rt + p * delta;              // model output
        const double target = std::min(rt, ct);           // g
        const double diff   = f_hat - target;             // (f – g)

        /* ------------- gradient ------------------------------------- */
        double g = 2.0 * diff * delta * sp;

        /* ------------- Hessian  (complete) -------------------------- */
        double h = 2.0 * delta * delta * sp * sp               // always ≥ 0
                 + 2.0 * diff  * delta * sp * (1.0 - 2.0 * p); // mixed term

        /* --------- numeric scaling & guards ------------------------- */
        g *= SCALE;
        h  = std::fabs(h) * SCALE;            // keep non-negative

        /* floors / caps */
        h  = std::max(H_MIN, std::min(H_MAX, h));
        g  = std::max(-G_CLIP, std::min(G_CLIP, g));

        grad[i] = g;
        hess[i] = h;
    }
    return 0;   // success
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
        constexpr int    EARLY_STOP   = 30;          // rounds w/out improvement
        constexpr int    PRINT_EVERY  = 10;
        int num_threads = 20;
        const int MAX_EPOCH = 1;

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
        
            
        for(auto p:S) const_cast<Sample*>(p)->prev_prob = 0.5;


        for (int cur_epoch = 0; cur_epoch < MAX_EPOCH; ++cur_epoch)
        {
             /* 1. 计算基础正/负样本权；正样本略降，负样本略升  */
            double P = 0, N0 = 0;
            for (auto p : S) p->label ? ++P : ++N0;
            double w_pos = 1.30 * N / (2 * P);   // ↑ 1.3
            double w_neg = 0.90 * N / (2 * N0);  // ↓ 0.9

            for (int i = 0; i < N; ++i) {
                const Sample& s = *S[i];

                // ✔ 直接写成普通判断
                if (s.feat.size() != NUM_FEATS) {
                    std::cerr << "[BUG] dir=" << s.dir_tag
                            << "  sample#" << i
                            << "  feat.size=" << s.feat.size()
                            << "  NUM_FEATS=" << NUM_FEATS << '\n';
                    std::abort();
                }

                /* ---------- 复制特征 & 标签 ---------- */
                std::copy(s.feat.begin(), s.feat.end(),
                        X.begin() + i * NUM_FEATS);

                const double rt = std::max(s.row_t, EPS_RUNTIME);
                const double ct = std::max(s.col_t, EPS_RUNTIME);
                // y[i] = static_cast<float>(std::log(rt) - std::log(ct));
                double diff = std::log(rt) - std::log(ct) ;
                // auto it = g_db_size_gb.find(s.dir_tag);             // ② 按 dir_tag 查
                // double gb = (it != g_db_size_gb.end()) ? it->second : 1.0;
                // diff /= std::max(1e-6, gb); 

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

                /* ---------- (D) 代价敏感放大  ←📍就在这里加 ---------- */
                w_i *= std::pow(std::fabs(rt - ct), 1.3); 

                /* (E)  focal-loss 难例权重 */
                w_i *= std::pow(1.0-std::fabs(s.prev_prob-0.5)*2,2.0);

                /* (F)  统一软剪裁：随 epoch 略放宽 */
                double epoch_scale = 1.0 + cur_epoch/200.0;
                double log_lo = std::log(0.05*base);
                double log_hi = std::log(20.0*base*epoch_scale);
                w_i = std::exp(clamp_val(std::log(w_i),log_lo,log_hi));

                double slow = std::pow(std::min(rt,ct) /*秒*/ , 0.8);  // or log1p(rt+ct)
                w_i *= 1.0 + 0.5 * slow;      // 系数 0.5~1.0 可 grid-search


                w[i] = static_cast<float>(w_i);

            }
            /* ========== 1. 重新计算 focal-weight (上面已算好 w[]) ========== */
            for (size_t j = 0; j < w.size(); ++j)           // 放进临时数组 w_tmp
                w[j] *= std::pow(1.0 - std::fabs(S[j]->prev_prob - 0.5) * 2, 2.0);

            /* ========== 2. 创建 dtrain / dvalid / booster ========== */
            DatasetHandle dtrain = nullptr, dvalid = nullptr;
            chk(!LGBM_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT32,
                                        N, NUM_FEATS, 1, "",
                                        nullptr, &dtrain),
                "DatasetCreate failed");
            LGBM_DatasetSetField(dtrain, "label",  y.data(), N, C_API_DTYPE_FLOAT32);
            LGBM_DatasetSetField(dtrain, "weight", w.data(), N, C_API_DTYPE_FLOAT32);

            if (!va.empty()) {                          // 你的原来的验证集构造逻辑
                const int Nv = static_cast<int>(va.size());
                std::vector<float> Xv(size_t(Nv)*NUM_FEATS), yv(Nv);
                for (int i = 0; i < Nv; ++i) {
                    std::copy(va[i].feat.begin(), va[i].feat.end(),
                            Xv.begin() + i*NUM_FEATS);
                    double rt = std::max(va[i].row_t, EPS_RUNTIME);
                    double ct = std::max(va[i].col_t, EPS_RUNTIME);
                    yv[i] = float(std::log(rt) - std::log(ct));
                    // auto it = g_db_size_gb.find(va[i].dir_tag);             // ② 按 dir_tag 查
                    // double gb = (it != g_db_size_gb.end()) ? it->second : 1.0;
                    // yv[i] /= std::max(1e-6, gb); 
                }
                chk(!LGBM_DatasetCreateFromMat(Xv.data(), C_API_DTYPE_FLOAT32,
                                            Nv, NUM_FEATS, 1, "",
                                            dtrain, &dvalid),
                    "DatasetCreate(valid) failed");
                LGBM_DatasetSetField(dvalid, "label", yv.data(), Nv, C_API_DTYPE_FLOAT32);
            }

            /* ------ 训练前准备 rt / ct 数组 ------ */
            std::vector<double> rt_d(N), ct_d(N);
            for (int i = 0; i < N; ++i) {
                rt_d[i] = S[i]->row_t;
                ct_d[i] = S[i]->col_t;
            }
            TimeObjData obj_data{ rt_d.data(), ct_d.data() };


            /* --- 参数字符串 param 与你旧代码完全一样 --- */
            std::string param = "boosting=" + booster_
                  + " objective=none metrics=l2"
                  + " learning_rate=" + std::to_string(lr*0.75)
                  + " num_leaves=1024 max_depth=18 max_bin=127"
                  + " feature_fraction=" + std::to_string(colsample)
                  + " lambda_l1=0 lambda_l2=0.1"
                  + " min_data_in_leaf=5"      // ★ 新增：允许小叶子
                  + " min_split_gain=0"        // ★ 新增：取消最小增益门槛
                  + " num_threads=" + std::to_string(num_threads)
                  + " verbosity=-1";
            
            if (booster_ == "rf") {
                // ① 先取 CLI 里传进来的 --subsample；否则默认 0.8
                double frac = (opt.subsample > 0.0 && opt.subsample < 1.0)
                                ? opt.subsample : 0.8;

                // ② rf 还要求 bagging_freq>0；设为 1 就行
                param += " bagging_fraction=" + std::to_string(frac)
                    +  " bagging_freq=1";
            }

            BoosterHandle booster = nullptr;
            chk(!LGBM_BoosterCreate(dtrain, param.c_str(), &booster),
                "BoosterCreate failed");
            // LGBM_BoosterSetObjective(booster, my_grad_hess_cb, nullptr);
            if (dvalid) LGBM_BoosterAddValidData(booster, dvalid);

            /* ========== 3. 训练循环（带进度条 + 早停） ========== */
            double best_metric = std::numeric_limits<double>::max();
            int    best_iter   = -1;

            double last_l2 = std::numeric_limits<double>::quiet_NaN();
            // LGBM_BoosterUpdateOneIter(booster, &fin);
            for (int it = 0, fin; it < num_trees; ++it) {
                /* 1. 先拿当前 raw score */
                std::vector<double> raw(N);
                int64_t out_len = 0;
                LGBM_BoosterPredictForMat(
                    booster, X.data(), C_API_DTYPE_FLOAT32,
                    N, NUM_FEATS, 1, C_API_PREDICT_RAW_SCORE,
                    -1, 0, "", &out_len, raw.data());

                /* 2. 算 grad / hess */
                std::vector<double> gD(N), hD(N);
                time_obj_grad_hess(raw.data(), nullptr, N,
                                gD.data(), hD.data(), &obj_data);

                std::vector<float> grad(N), hess(N);
                for (int i = 0; i < N; ++i) {        // 转 float
                    grad[i] = static_cast<float>(gD[i]);
                    hess[i] = static_cast<float>(hD[i]);
                }

                /* 3. 更新一棵树 */
                LGBM_BoosterUpdateOneIterCustom(
                    booster, grad.data(), hess.data(), &fin);            

                // if (it == 0 && cur_epoch == 0) {
                //     int sz = 0;
                //     LGBM_BoosterNumberOfTotalModel(booster, &sz);     // sz=1 就是首棵树
                    // double gain = 0;
                    // LGBM_BoosterFeatureImportance(booster, 0, 0, &gain);
                    // std::cerr<<"[DBG] first-tree gain = "<<gain<<"\n";
                // }

                 /* === 每 PRINT_EVERY 轮在验证集上评估一次 === */
                if (dvalid && (it + 1) % PRINT_EVERY == 0) {
                    int n = 0;  LGBM_BoosterGetEvalCounts(booster, &n);
                    std::vector<double> eval(n); int out_len=0;
                    LGBM_BoosterGetEval(booster, 0, &out_len, eval.data());

                    double l2 = eval.empty() ? 0.0 : eval[0];
                    last_l2   = l2;                         // ① 先更新 last_l2

                    // ② 早停判定
                    if (l2 < best_metric - 1e-7) { 
                        best_metric = l2;  best_iter = it; 
                    } else if (it - best_iter >= EARLY_STOP) {
                        break;                              // 提前结束本 epoch
                    }
                }

                /* === 打印进度条（用刚刚更新的 last_l2） === */
                std::ostringstream tag;
                tag << "ep" << cur_epoch
                    << " l2=" << std::fixed << std::setprecision(4)
                    << (std::isnan(last_l2) ? 0.0 : last_l2);

                progress(tag.str(), it + 1, num_trees);
            }
            std::cerr << '\n';

            /* ========== 4. 预测训练集，写回 prev_prob ========== */
            std::vector<double> raw(N); int64_t out_len = 0;
            chk(!LGBM_BoosterPredictForMat(
                    booster, X.data(), C_API_DTYPE_FLOAT32,
                    N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
                    -1, 0, "", &out_len, raw.data()),
                "PredictForMat failed");
            for (size_t i = 0; i < S.size(); ++i)
                const_cast<Sample*>(S[i])->prev_prob = sigmoid(raw[i]);

            /* ========== 5. 仅最后一轮落盘 ========== */
            if (cur_epoch == MAX_EPOCH - 1)
                LGBM_BoosterSaveModel(booster, 0, -1, 0, model_path.c_str());

            /* ========== 6. 清理，下一轮重建 ========== */
            LGBM_BoosterFree(booster);
            LGBM_DatasetFree(dtrain);
            if (dvalid) LGBM_DatasetFree(dvalid);
        }   /* <<< end for cur_epoch */
        

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
