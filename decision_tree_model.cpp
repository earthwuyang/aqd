/**********************************************************************
 * decision_tree_model.cpp
 *
 * A light-weight wrapper that exposes a single-CART classifier through
 * the generic IModel interface.
 *
 *  ┌─ IModel ------------------------------------------------────────┐
 *  │ virtual void   train (…)                                     │
 *  │ virtual std::vector<int> predict (…) const                  │
 *  │ virtual double bal_acc  (…) const                           │
 *  └───────────────────────────────────────────────────────────────┘
 *
 * Build (example):
 *   g++ -O3 -std=c++17 -fopenmp \
 *       decision_tree_model.cpp -o decision_tree_model \
 *       -I.                              # for common.hpp / json.hpp
 *********************************************************************/
#include "common.hpp"          // NUM_FEATS, Sample, helpers, logging …
#include "model_iface.hpp"     // IModel declaration
#include "json.hpp"
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

using json = nlohmann::json;

/* ------------------------------------------------------------------ */
/* 1.  Decision-tree wrapper (updated to TrainOpt & DS_val)           */
/* ------------------------------------------------------------------ */
class DTreeModel : public IModel
{
public:
    /* hyper-parameters – taken from TrainOpt each time train() runs */
    int    max_depth   = 15;
    int    min_samples = 40;
    double min_gain    = 0.0005;
    float  threshold   = 0.0f;      //  ŷ > τ ⇒ choose column-store

    /* ------------ train / fine-tune -------------------------------- */
    void train(const std::vector<Sample>& DS_tr,
               const std::vector<Sample>& /*DS_val – optional, unused*/,
               const std::string&         model_path,
               const TrainOpt&            opt) override
    {
        /* honour TrainOpt on each invocation */
        max_depth   = opt.max_depth  > 0   ? opt.max_depth   : max_depth;
        min_samples = opt.hidden1    > 0   ? opt.hidden1     : min_samples;
        min_gain    = opt.lr > 0.0         ? opt.lr * 1e-2   : min_gain; // tiny hack

        /* ------------------------------------------------------------------
         *  If the caller only wants inference we just lazy-load the model
         * ------------------------------------------------------------------ */
        if (opt.skip_train) {
            ensure_tree_loaded(model_path);
            return;
        }

        /* ---------- assemble weighted matrices from DS_tr ---------- */
        std::vector<std::array<float,NUM_FEATS>> X;
        std::vector<int>         y;
        std::vector<float>       w;
        X.reserve(DS_tr.size());  y.reserve(DS_tr.size()); w.reserve(DS_tr.size());

        std::size_t P = 0, N = 0;
        for (const auto& s : DS_tr) (s.label ? ++P : ++N);

        const float w_pos = P ? float(P + N) / (2.f * P) : 1.f;
        const float w_neg = N ? float(P + N) / (2.f * N) : 1.f;

        for (const auto& s : DS_tr) {
            X.push_back(s.feat);
            y.push_back(s.label);
            w.push_back(s.label ? w_pos : w_neg);
        }

        /* ---------- learn the tree ---------- */
        tree_ = DecisionTree(max_depth, min_samples, min_gain);
        tree_.fit(X, y, w);
        loaded_ = true;                 // model is resident in memory

        /* ---------- persist ---------- */
         json j;
         j["nodes"] = tree_.to_json();
         j["meta"]  = {{"max_depth",   max_depth},
                     {"min_samples", min_samples},
                     {"min_gain",    min_gain},
                     {"features",    NUM_FEATS}};
         std::ofstream(model_path) << j.dump(2);
         logI("Decision-tree model saved → " + model_path);

         /* ---------- quick evaluation on DS_tr ---------------------- */
         {
            // 1. 预测当前树
            std::vector<int> pred = predict(model_path, DS_tr, /*τ*/0.0f);

            long   TP=0, FP=0, TN=0, FN=0;
            double r_row=0, r_col=0, r_opt=0,
                  r_costthr=0, r_hyp=0, r_fann=0, r_dtree=0;

            for (size_t i = 0; i < DS_tr.size(); ++i) {
               const Sample& s = DS_tr[i];

               bool col_dtree = pred[i];               // 本决策树
               bool col_hyp   = (s.hybrid_pred == 1);  // Hybrid Optimizer
               bool col_fann  = (s.fann_pred   == 1);  // FANN baseline
               bool use_row   = (s.qcost < COST_THR);  // cost threshold

               /* runtime 累加 */
               r_row     += s.row_t;
               r_col     += s.col_t;
               r_opt     += std::min(s.row_t, s.col_t);
               r_costthr += use_row   ? s.row_t : s.col_t;
               r_hyp     += col_hyp   ? s.col_t : s.row_t;
               r_fann    += col_fann  ? s.col_t : s.row_t;
               r_dtree   += col_dtree ? s.col_t : s.row_t;

               /* 仅对决策树统计分类指标 */
               (col_dtree ? (s.label?++TP:++FP)
                           : (s.label?++FN:++TN));
            }

            auto avg = [&](double v){ return v / DS_tr.size(); };
            std::cout << std::fixed << std::setprecision(6)
                     << "\n*** Decision-Tree evaluation on " << DS_tr.size()
                     << " samples ***\n"
                     << "Acc="      << double(TP+TN)/DS_tr.size()
                     << "  BalAcc=" << 0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
                     << "  F1="     << (TP?2.0*TP/(2*TP+FP+FN):0) << '\n'
                     << "Row only        : " << avg(r_row)     << '\n'
                     << "Col only        : " << avg(r_col)     << '\n'
                     << "Cost-Threshold  : " << avg(r_costthr) << '\n'
                     << "HybridOpt       : " << avg(r_hyp)     << '\n'
                     << "FANN baseline   : " << avg(r_fann)    << '\n'
                     << "DecisionTree    : " << avg(r_dtree)   << '\n'
                     << "Oracle (min)    : " << avg(r_opt)     << "\n\n";
         }
    }

    /* ------------ predict ---------------------------------------- */
    std::vector<int> predict(const std::string&         model_path,
                             const std::vector<Sample>& DS,
                             float                      tau = 0.0f) const override
    {
        ensure_tree_loaded(model_path);

        std::vector<int> out;
        out.reserve(DS.size());
        for (const auto& s : DS)
            out.push_back(tree_.predict(s.feat.data()) > tau);
        return out;
    }

    /* ------------ balanced accuracy ------------------------------ */
    double bal_acc(const std::string&         model_path,
                   const std::vector<Sample>& DS,
                   float                      tau = 0.0f) const override
    {
        auto pred = predict(model_path, DS, tau);
        long TP=0,TN=0,FP=0,FN=0;
        for (std::size_t i=0;i<DS.size();++i){
            bool col = pred[i];
            bool gt  = DS[i].label;
            (col ? (gt?++TP:++FP) : (gt?++FN:++TN));
        }
        double tpr = (TP+FN) ? double(TP)/(TP+FN) : 0.0;
        double tnr = (TN+FP) ? double(TN)/(TN+FP) : 0.0;
        return 0.5*(tpr+tnr);
    }

private:
    mutable DecisionTree tree_{max_depth, min_samples, min_gain};
    mutable bool loaded_ = false;

    /* ---------- lazy loader for inference-only paths ------------- */
    void ensure_tree_loaded(const std::string& model_path) const
    {
        if (loaded_) return;
        std::ifstream in(model_path);
        if (!in) { logE("cannot open model file "+model_path); std::exit(1); }
        json j; in >> j;
        tree_.from_json(j.at("nodes"));
        loaded_ = true;
    }
};

/* factory – called by the main driver once it parses --model=dtree */
std::unique_ptr<IModel> make_dtree()
{
    return std::make_unique<DTreeModel>();
}
