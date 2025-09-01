/*  lightgbm_model.cpp  --------------------------------------- */
#include <LightGBM/c_api.h>
#include <cerrno>
#include <iostream>
#include <thread>
#include <cmath>
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <omp.h>

#include "model_iface.hpp"
#include "vib.hpp"
#include "shap_util.hpp"

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

constexpr char ROW_SUFFIX[] = "_row";      // row_fold0.txt …
constexpr char COL_SUFFIX[] = "_col";      // col_fold0.txt …
constexpr char FINAL_ROW [] = "lightgbm_row_best.txt";
constexpr char FINAL_COL [] = "lightgbm_col_best.txt";
constexpr char FEAT_FILE [] = ".feat";

inline double   log1p_safe(double x) { return std::log1p(std::max(x,1e-9)); }

template <typename T>
inline T clamp_val(const T& v, const T& lo, const T& hi)
{
    return std::max(lo, std::min(v, hi));
}

/* ★ util – sigmoid(raw_score) → [0,1] 概率 */
inline double sigmoid(double z){ return 1.0 / (1.0 + std::exp(-z)); }



/* ---------- Regret-Aware BCE 目标 ------------------------------
 *  grad = (p-y) * regret
 *  hess = p(1-p) * regret
 *  其中 regret = |rt-ct|  (秒)
 * ------------------------------------------------------------- */
struct RegretObjData {
    const double* rt;   // 行存真实耗时
    const double* ct;   // 列存真实耗时
};

int regret_obj_grad_hess(const double* score, const double* /*label*/,
                         int64_t num_data,
                         double* grad, double* hess,
                         void* user_data)
{
    const auto* d = static_cast<const RegretObjData*>(user_data);
    constexpr double G_CLIP = 50.0;       // 和你原来的保持一致

    for (int64_t i = 0; i < num_data; ++i) {
        double rt = d->rt[i], ct = d->ct[i];
        double y  = (ct < rt) ? 1.0 : 0.0;           // 1 ⇒ 列更快
        double r  = std::fabs(rt - ct);              // regret (秒)

        double p = 1.0 / (1.0 + std::exp(-score[i]));
        double g = (p - y) * r;                      // 见上式
        double h = std::max(1e-6, p * (1.0 - p)) * r;

        /* 与旧实现一致的裁剪 / 地板 */
        g = std::max(-G_CLIP, std::min(G_CLIP, g));
        h = std::max(1e-4, h);
        grad[i] = g;
        hess[i] = h;
    }
    return 0;
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

static bool copy_file_bin(const std::string& src,
                          const std::string& dst)
{
    std::ifstream fin(src, std::ios::binary);
    if (!fin) return false;
    std::ofstream fout(dst, std::ios::binary);
    if (!fout) return false;
    fout << fin.rdbuf();          // 一行把整文件搬过去
    return (bool)fout;
}


static void report_qerr(const char* tag,
                        BoosterHandle booster,
                        const std::vector<float>& X,
                        const std::vector<Sample>& samp,
                        bool is_row)
{
    const int64_t n   = static_cast<int64_t>(samp.size());
    const int64_t F   = NUM_FEATS;
    std::vector<double> pred_log(n);
    int64_t out_len = 0;

    // ---- 1. 预测（仍返回 log1p 值）----
    chk(!LGBM_BoosterPredictForMat(
            booster,                    // handle
            X.data(),                   // data ptr
            C_API_DTYPE_FLOAT32,        // float32
            n, F,                       // nrow, ncol
            1,                          // row-major
            C_API_PREDICT_NORMAL,       // same as RAW for regression
            -1,                         // all trees
            0,                          // start_iteration
            "",                         // extra params
            &out_len,                   // returned length
            pred_log.data()),           // output buffer
        "Predict");

    // ---- 2. 计算 qerror ----
    std::vector<double> qerr(n);
    for (int64_t i = 0; i < n; ++i) {
        double pred = std::expm1(pred_log[i]);
        double trg  = is_row ? samp[i].row_t : samp[i].col_t;
        pred = std::max(pred, 1e-9);
        trg  = std::max(trg , 1e-9);
        qerr[i] = std::max(pred / trg, trg / pred);
    }

    std::sort(qerr.begin(), qerr.end());
    const double mn   = qerr.front();
    const double mx   = qerr.back();
    const double mean = std::accumulate(qerr.begin(), qerr.end(), 0.0) / n;
    const double med  = qerr[n / 2];
    const double p95  = qerr[ static_cast<size_t>(0.95*(n-1)) ];

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s qErr  min=%.3g  max=%.3g  mean=%.3g  median=%.3g  p95=%.3g",
        tag, mn, mx, mean, med, p95);
    logI(buf);
}

class LGBModel : public IModel {
    std::string booster_; //"goss" or "rf"
    bool        use_custom_loss_; 
public:
    explicit LGBModel(std::string booster = "goss")
        : booster_(std::move(booster)),
        use_custom_loss_(booster_ != "rf") {}  //rf doesn't use custom objective

    void train(const std::vector<Sample>  &tr,
            const std::vector<Sample>  &va,
            const std::string          &model_path,     // 不带后缀
            const TrainOpt             &opt,
            const std::unordered_map<std::string,double>& /*DIR_W: 已弃用*/)
    {
        if (tr.empty()) { logE("train(): empty training set"); return; }

        /* ---------- 0. 选特征 (如需 SHAP/VIB 可填 feat_keep) -------------- */
        std::vector<int> feat_keep;                // 为空==保留全部
        const int F = NUM_FEATS;
        const int N = static_cast<int>(tr.size());

        /* ---------- 1. 生成 X / y_row / y_col ----------------------------- */
        std::vector<float> X(static_cast<size_t>(N)*F),
                        y_row(N),
                        y_col(N);

        for (int i = 0; i < N; ++i) {
            const Sample &s = tr[i];
            if (feat_keep.empty())
                std::copy(s.feat.begin(), s.feat.end(), X.begin() + size_t(i)*F);
            else
                for (int k = 0; k < F; ++k)
                    X[size_t(i)*F + k] = s.feat[ feat_keep[k] ];

            y_row[i] = float(log1p_clip(s.row_t));
            y_col[i] = float(log1p_clip(s.col_t));
        }

        /* ---------- 2. 构建验证集 (可选) ---------------------------------- */
        const int Nv = static_cast<int>(va.size());
        std::vector<float> Xv, y_row_v, y_col_v;
        DatasetHandle dval_row = nullptr, dval_col = nullptr;

        if (Nv) {
            Xv.resize(size_t(Nv)*F);
            y_row_v.resize(Nv);
            y_col_v.resize(Nv);

            for (int i = 0; i < Nv; ++i) {
                const Sample &s = va[i];
                if (feat_keep.empty())
                    std::copy(s.feat.begin(), s.feat.end(), Xv.begin() + size_t(i)*F);
                else
                    for (int k = 0; k < F; ++k)
                        Xv[size_t(i)*F + k] = s.feat[ feat_keep[k] ];

                y_row_v[i] = float(log1p_clip(s.row_t));
                y_col_v[i] = float(log1p_clip(s.col_t));
            }
        }

        /* ---------- 3. 构建训练 Dataset（无权重） ------------------------- */
        auto build_ds = [&](const std::vector<float>& y,
                            DatasetHandle& out) {
            chk(!LGBM_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT32,
                                        N, F, 1, "", nullptr, &out), "Dataset");
            LGBM_DatasetSetField(out, "label", y.data(), N, C_API_DTYPE_FLOAT32);
        };
        DatasetHandle dtr_row = nullptr, dtr_col = nullptr;
        build_ds(y_row, dtr_row);
        build_ds(y_col, dtr_col);

        auto build_val = [&](const std::vector<float>& y,
                            DatasetHandle tr, DatasetHandle& out,
                            const std::vector<float>& Xbuf) {
            chk(!LGBM_DatasetCreateFromMat(Xbuf.data(), C_API_DTYPE_FLOAT32,
                                        Nv, F, 1, "", tr, &out), "Dataset(val)");
            LGBM_DatasetSetField(out, "label", y.data(), Nv, C_API_DTYPE_FLOAT32);
        };
        if (Nv) {
            build_val(y_row_v, dtr_row, dval_row, Xv);
            build_val(y_col_v, dtr_col, dval_col, Xv);
        }

        /* ---------- 4. 通用 Booster 参数 ---------------------------------- */
        const std::string PARAM =
            "objective=regression_l2 metric=l2 "
            "num_leaves=512 max_depth=-1 "
            "learning_rate="   + std::to_string(opt.lr)        + ' ' +
            "feature_fraction="+ std::to_string(opt.colsample) + ' ' +
            "bagging_fraction=0.8 bagging_freq=1 "
            "min_data_in_leaf=50 verbosity=-1 num_threads=" +
            std::to_string(std::max(1u, std::thread::hardware_concurrency()));

        /* ---------- 5. 单模型训练函数 (带 Early-Stopping) ------------------ */
        auto train_one = [&](DatasetHandle dtr,
                            DatasetHandle dval,
                            const std::string& out_path)
        {
            BoosterHandle booster = nullptr;
            chk(!LGBM_BoosterCreate(dtr, PARAM.c_str(), &booster), "Create");
            if (dval) LGBM_BoosterAddValidData(booster, dval);

            const int eval_freq = 10;                 // 每 10 轮评估一次
            double best_l2   = 1e100;
            int    best_iter = -1;
            int    no_imp    = 0;
            int    it        = 0;

            for (; it < opt.trees; ++it) {
                int fin = 0;
                LGBM_BoosterUpdateOneIter(booster, &fin);
                progress("iter", it + 1, opt.trees);           // 实时进度

                if (dval && (it + 1) % eval_freq == 0) {
                    double l2 = 0.0;
                    int dummy = 0;                                 // int 才符合签名
                    LGBM_BoosterGetEval(booster, 0, &dummy, &l2);  // OK

                    if (l2 + 1e-12 < best_l2) {
                        best_l2   = l2;
                        best_iter = it;
                        no_imp    = 0;
                    } else {
                        no_imp += eval_freq;
                        if (no_imp >= opt.early_round) {
                            logI("Early-stop at iter=" + std::to_string(it+1));
                         
                            ++it;                                 // 让进度条结束
                            break;
                        }
                    }
                }
            }
            std::fprintf(stderr, "\n");

            /* 保存最佳模型（若未触发早停则保存全部） */
            const int num_model = (best_iter >= 0 ? best_iter + 1 : it);
            chk(!LGBM_BoosterSaveModel(
                    booster, 0, num_model, 0, out_path.c_str()), "Save");


            if (!X.empty())  // 使用训练集；若有验证集更想看 va -> 传 Xbuf & va
            {
                const bool is_row = (out_path.find(ROW_SUFFIX) != std::string::npos);
                report_qerr(is_row ? "[row]" : "[col]",
                            booster, X, tr, is_row);
            }
            LGBM_BoosterFree(booster);
        };

        /* ---------- 6. 并行训练 row / col ------------------------------- */
    #pragma omp parallel sections
        {
    #pragma omp section
            { train_one(dtr_row, dval_row, model_path + ROW_SUFFIX); }
    #pragma omp section
            { train_one(dtr_col, dval_col, model_path + COL_SUFFIX); }
        }

        /* ---------- 7. 备份特征列清单 (若做了特征筛) -------------------- */
        if (!feat_keep.empty()) {
            std::ofstream ft(model_path + FEAT_FILE);
            for (size_t k = 0; k < feat_keep.size(); ++k)
                ft << (k ? " " : "") << feat_keep[k];
        }

        /* ---------- 8. 释放句柄 ---------------------------------------- */
        LGBM_DatasetFree(dtr_row); LGBM_DatasetFree(dtr_col);
        if (dval_row) LGBM_DatasetFree(dval_row);
        if (dval_col) LGBM_DatasetFree(dval_col);
    }


    /* ------------------------------------------------------------
     * predict()
     *   model_path 既可以是：
     *      · 训练阶段生成的 stub（两行 ROW=…/COL=…）
     *      · 或直接给出 *.txt 时，按 “去扩展名 + _row.txt/_col.txt” 寻找
     * ------------------------------------------------------------ */
    std::vector<int>
    predict(const std::string&        root_path,   // 不带后缀
                    const std::vector<Sample>& DS,
                    float /*tau_unused*/) const override
    {
        const int N = int(DS.size());
        if (!N) return {};

        /* —— 读取特征列清单（若存在） —— */
        std::vector<int> keep;
        {
            std::ifstream fin(root_path + FEAT_FILE);
            int id; while (fin >> id) keep.push_back(id);
        }
        const int F = keep.empty() ? NUM_FEATS : int(keep.size());

        /* —— 加载两个模型 —— */
        auto load_model = [&](const std::string& fp)->BoosterHandle {
            BoosterHandle h=nullptr; int _;
            if (LGBM_BoosterCreateFromModelfile(fp.c_str(), &_, &h)!=0 || !h)
                throw std::runtime_error("cannot load "+fp);
            return h;
        };
        BoosterHandle bro = load_model(root_path+ROW_SUFFIX);
        BoosterHandle bco = load_model(root_path+COL_SUFFIX);

        /* —— 构造特征矩阵 —— */
        std::vector<float> X(size_t(N)*F);
        for (int i=0;i<N;++i)
            if (keep.empty())
                std::copy(DS[i].feat.begin(), DS[i].feat.end(), X.begin()+size_t(i)*F);
            else
                for (int k=0;k<F;++k) X[size_t(i)*F+k] = DS[i].feat[ keep[k] ];

        /* —— 预测 —— */
        auto run_pred = [&](BoosterHandle h, std::vector<double>& out){
            out.resize(N); int64_t _=0;
            LGBM_BoosterPredictForMat(
                h, X.data(), C_API_DTYPE_FLOAT32,
                N, F, 1, C_API_PREDICT_NORMAL,
                -1, 0, "", &_, out.data());
        };
        std::vector<double> pr_row, pr_col;
        run_pred(bro, pr_row);
        run_pred(bco, pr_col);
        LGBM_BoosterFree(bro); LGBM_BoosterFree(bco);

        /* —— 逐条比较 —— */
        std::vector<int> choose(N);
        for (int i=0;i<N;++i) {
            double t_row = std::exp(pr_row[i]) - 1.0;   // inverse log1p
            double t_col = std::exp(pr_col[i]) - 1.0;
            choose[i] = (t_col < t_row);                // 1 = 列更快
        }
        return choose;
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

private:
    /* 训练单个回归模型，返回验证 l2   *
     * is_row==true  :  label = log1p(row_t)
     * is_row==false :  label = log1p(col_t)                               */
    static double train_one(const std::vector<Sample>& DS,
                            const std::vector<float>&  X,
                            const std::vector<int>&   tr,
                            const std::vector<int>&   va,
                            bool      is_row,
                            const std::string& out_path,
                            int       num_trees,
                            int&      dummy /*for GetEval*/)
    {
        const int F = NUM_FEATS;
        /* -------- build train set -------- */
        std::vector<float> y_tr(tr.size());
        std::vector<float> y_va(va.size());

        for (size_t i=0;i<tr.size();++i){
            double t = is_row ? DS[tr[i]].row_t : DS[tr[i]].col_t;
            y_tr[i]  = float(log1p_safe(t));
        }
        for (size_t i=0;i<va.size();++i){
            double t = is_row ? DS[va[i]].row_t : DS[va[i]].col_t;
            y_va[i]  = float(log1p_safe(t));
        }

        DatasetHandle dtr=nullptr,dva=nullptr;
        chk(!LGBM_DatasetCreateFromMat(
                /*data*/X.data()+size_t(tr.front())*F*sizeof(float), // pointer OK – LightGBM 会 copy
                C_API_DTYPE_FLOAT32, tr.size(), F, 1, "", nullptr, &dtr),
            "DatasetCreate tr");

        chk(!LGBM_DatasetCreateFromMat(
                /*data*/X.data()+size_t(va.front())*F*sizeof(float),
                C_API_DTYPE_FLOAT32, va.size(), F, 1, "", dtr, &dva),
            "DatasetCreate va");

        LGBM_DatasetSetField(dtr,"label",y_tr.data(),y_tr.size(),C_API_DTYPE_FLOAT32);
        LGBM_DatasetSetField(dva,"label",y_va.data(),y_va.size(),C_API_DTYPE_FLOAT32);

        const std::string param =
            "objective=regression_l2 boosting=goss "
            "num_leaves=256 learning_rate=0.05 "
            "feature_fraction=0.8 "
            // "bagging_fraction=0.8 bagging_freq=1 "
            "verbosity=-1 num_threads=8";

        BoosterHandle booster=nullptr;
        chk(!LGBM_BoosterCreate(dtr,param.c_str(),&booster),"BoosterCreate");
        LGBM_BoosterAddValidData(booster,dva);

        double best_l2 = 1e100; int best_it = -1;

        for (int it=0,fin;it<num_trees;++it){
            LGBM_BoosterUpdateOneIter(booster,&fin);
            if((it+1)%20==0){
                double l2=0;
                LGBM_BoosterGetEval(booster,0,&dummy,&l2);
                if(l2<best_l2){best_l2=l2;best_it=it;}
            }
        }
        LGBM_BoosterSaveModel(booster,0,best_it+1,0,out_path.c_str());
        LGBM_BoosterFree(booster);
        LGBM_DatasetFree(dtr); LGBM_DatasetFree(dva);
        logI("  saved "+out_path+"  (best l2="+std::to_string(best_l2)+")");
        return best_l2;
    }

    /* 运行单次预测 (C_API_PREDICT_NORMAL) -------------------------- */
    static void run_pred(const std::string& model_file,
                         const std::vector<float>& X,
                         int N,int F,
                         std::vector<double>& out)
    {
        BoosterHandle booster=nullptr; int iters=0;
        chk(!LGBM_BoosterCreateFromModelfile(model_file.c_str(),
                                             &iters,&booster),
            "load model");
        int64_t olen=0;
        out.resize(N);
        chk(!LGBM_BoosterPredictForMat(
                booster,X.data(),C_API_DTYPE_FLOAT32,
                N,F,1,C_API_PREDICT_NORMAL,
                -1,0,"",&olen,out.data()),
            "predict");
        LGBM_BoosterFree(booster);
    }
};

/* factory – so main() can remain agnostic */
std::unique_ptr<IModel> make_lightgbm(const std::string& booster) { 
    return std::make_unique<LGBModel>(booster); 
}
