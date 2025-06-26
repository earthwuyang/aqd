/**********************************************************************
 *  random_forest_model.cpp  –  Random-Forest gap-regression router
 *
 *  • 训练目标:  Δ = row_t − col_t   (负数 ⇒ 列存储更快)
 *  • 预测规则:  if Δ < 0 → column, else row    (τ ≡ 0)
 *  • 并行建树  :  OpenMP (#pragma omp parallel for)
 *
 *  build 例:
 *      g++ -O3 -std=c++17 -fopenmp \
 *          random_forest_model.cpp -o rf_model \
 *          -I.                     # common.hpp / json.hpp / …
 *********************************************************************/
#include "common.hpp"          // NUM_FEATS, Sample, logI/logW/logE
#include "model_iface.hpp"
#include "json.hpp"
#include <omp.h>
#include <random>
#include <numeric>
#include <iomanip>

using json = nlohmann::json;

/* ──────────────────────────────────────────────────────────── */
/*                 1.  轻量级 CART 回归树                      */
/* ──────────────────────────────────────────────────────────── */
struct RegNode {
    int feat = -1;                // -1 ⇒ leaf
    float thr = 0.f;
    int left = -1, right = -1;    // child indices
    double value = 0.0;           // 叶结点存均值 Δ
};

class CartRegTree {
    std::vector<RegNode> nodes_;
    int    max_depth_;
    int    min_samples_;
    double min_gain_;
    /* 误差度量: Σ(y−μ)² */
    static double sse(const std::vector<double>& y,
                      const std::vector<float>&  w,
                      const std::vector<int>&    idx)
    {
        double sw = 0, sy = 0;
        for (int i:idx) { sw += w[i]; sy += w[i]*y[i]; }
        if (sw == 0) return 0;
        double mu = sy / sw;
        double err = 0;
        for (int i:idx) {
            double d = y[i] - mu;
            err += w[i] * d * d;
        }
        return err;
    }

    int build(const std::vector<std::array<float,NUM_FEATS>>& X,
              const std::vector<double>&                     y,
              const std::vector<float>&                      w,
              std::vector<int>&                              idx,
              int depth)
    {
        /* 计算当前结点信息 */
        double sw = 0, sy = 0;
        for (int i:idx) { sw += w[i]; sy += w[i]*y[i]; }
        double mu = (sw? sy/sw : 0);

        RegNode node;
        node.value = mu;

        /* 停止条件 */
        if (depth >= max_depth_ || idx.size() <= min_samples_) {
            nodes_.push_back(node);
            return int(nodes_.size()) - 1;
        }

        /* 穷举特征 & 若干候选阈值, 选 ΔSSE 最大的分裂 */
        double best_gain = 0;
        int    best_feat = -1;
        float  best_thr  = 0.f;
        std::vector<int> best_left, best_right;

        for (int f = 0; f < NUM_FEATS; ++f) {
            /* 取得唯一值，采样 ≤16 个阈值 */
            std::vector<float> vals;
            vals.reserve(idx.size());
            for (int i:idx) vals.push_back(X[i][f]);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(),vals.end()), vals.end());
            if (vals.size() < 2) continue;

            int step = std::max<size_t>(1, vals.size()/16);
            for (size_t k=1; k<vals.size(); k+=step) {
                float thr = (vals[k-1] + vals[k]) * 0.5f;
                std::vector<int> L, R;
                for (int i:idx)
                    (X[i][f] < thr ? L : R).push_back(i);
                if (L.size() < min_samples_ || R.size() < min_samples_) continue;

                double err_tot = sse(y,w,idx);
                double gain = err_tot - sse(y,w,L) - sse(y,w,R);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feat = f;
                    best_thr  = thr;
                    best_left.swap(L);
                    best_right.swap(R);
                }
            }
        }

        if (best_gain < min_gain_ || best_feat == -1) {      // 无可行分裂
            nodes_.push_back(node);
            return int(nodes_.size())-1;
        }

        /* 创建内部结点并递归 */
        int self = int(nodes_.size());
        node.feat = best_feat;
        node.thr  = best_thr;
        nodes_.push_back(node);         // 占位

        int left  = build(X,y,w,best_left ,depth+1);
        int right = build(X,y,w,best_right,depth+1);
        nodes_[self].left  = left;
        nodes_[self].right = right;
        return self;
    }

public:
    CartRegTree(int md,int ms,double mg)
        : max_depth_(md), min_samples_(ms), min_gain_(mg) {}

    void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
             const std::vector<double>&                     y,
             const std::vector<float>&                      w)
    {
        nodes_.clear();
        std::vector<int> idx(X.size());
        std::iota(idx.begin(), idx.end(), 0);
        build(X,y,w,idx,0);
    }

    double predict(const float* x) const {
        int id = 0;
        while (nodes_[id].feat != -1) {
            id = (x[nodes_[id].feat] < nodes_[id].thr)
                 ? nodes_[id].left : nodes_[id].right;
        }
        return nodes_[id].value;
    }

    json to_json() const {
        json arr = json::array();
        for (auto& n: nodes_) arr.push_back({
            {"feat",n.feat},{"thr",n.thr},
            {"left",n.left},{"right",n.right},
            {"val",n.value}});
        return arr;
    }
    void from_json(const json& arr) {
        nodes_.clear();
        for (auto& e:arr) {
            RegNode n;
            n.feat  = e.at("feat").get<int>();
            n.thr   = e.at("thr") .get<float>();
            n.left  = e.at("left").get<int>();
            n.right = e.at("right").get<int>();
            n.value = e.at("val") .get<double>();
            nodes_.push_back(n);
        }
    }
};

/* ──────────────────────────────────────────────────────────── */
/*                     2.  Random-Forest 回归                   */
/* ──────────────────────────────────────────────────────────── */
class RandomForestReg {
    std::vector<CartRegTree> trees_;
    double sampleRatio_;
public:
    RandomForestReg(int n,int md,int ms,double mg,double sr)
        : sampleRatio_(sr)
    {
        trees_.reserve(n);
        for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
    }

    void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
             const std::vector<double>&                     y,
             const std::vector<float>&                      w,
             std::mt19937& rng)
    {
        const std::size_t N = X.size();
        std::uniform_int_distribution<int> uni(0, int(N-1));

        // #pragma omp parallel for schedule(dynamic)
        for (int t=0; t<int(trees_.size()); ++t)
        {
            std::vector<std::array<float,NUM_FEATS>> BX;
            std::vector<double> By;
            std::vector<float>  Bw;

            std::mt19937 rng_local(rng() + t);
            std::uniform_int_distribution<int> uni_loc(0, int(N-1));

            std::size_t m = std::size_t(sampleRatio_ * N);
            BX.reserve(m); By.reserve(m); Bw.reserve(m);
            for (std::size_t i=0;i<m;++i)
            {
                int j = uni_loc(rng_local);
                BX.push_back(X[j]);
                By.push_back(y[j]);
                Bw.push_back(w[j]);
            }
            trees_[t].fit(BX,By,Bw);

            progress("RFreg-train", t, trees_.size());
        }
    }

    double predict(const float* x) const {
        double s = 0;
        for (auto& tr:trees_) s += tr.predict(x);
        return s / trees_.size();
    }

    json to_json() const {
        json arr = json::array();
        for (auto& tr:trees_) arr.push_back(tr.to_json());
        return arr;
    }
    void from_json(const json& arr) {
        trees_.clear();
        for (auto& jt:arr) {
            CartRegTree tr(0,0,0);
            tr.from_json(jt);
            trees_.push_back(std::move(tr));
        }
    }
};

/* ──────────────────────────────────────────────────────────── */
/*                 3.  RFModel   (IModel 实现)                  */
/* ──────────────────────────────────────────────────────────── */
class RFModel : public IModel
{
public:
    int    nTrees      = 400;
    int    maxDepth    = 10;
    int    minSamples  = 40;
    double minGain     = 1e-4;
    double sampleRatio = 0.63;

    /* =======================  Train  ======================= */
    void train(const std::vector<Sample>& DS_tr,
               const std::vector<Sample>& DS_val,
               const std::string&        path,
               const TrainOpt&           opt) override
    {
        /* 1. skip_train ⇒ load  */
        if (opt.skip_train) { ensure_loaded(path); return; }

        /* 2. prepare matrices */
        std::vector<std::array<float,NUM_FEATS>> X;
        std::vector<double>  Y;          // Δ
        std::vector<float>   W;          // 可按 gap or cost 加权

        constexpr double GAP_EMPH = 0.25;
        for (auto& s:DS_tr)
        {
            X.push_back(s.feat);
            double gap = std::log((s.col_t + 1e-9) / (s.row_t+ 1e-9) );
            Y.push_back(gap);

            double wgap  = 1.0 + GAP_EMPH*std::abs(gap)/(s.row_t+s.col_t+1e-6);
            double wcost = 1.0 + std::min(s.qcost,1e6)/1e5;
            W.push_back(float(wgap*wcost));
        }

        /* 3. grow forest */
        rf_ = RandomForestReg(nTrees,maxDepth,minSamples,minGain,sampleRatio);
        std::mt19937 rng(opt.threads?opt.threads:42);
        rf_.fit(X,Y,W,rng);
        logI("Random-Forest(gap) trained with " + std::to_string(nTrees) +
             " trees – saved → " + path);

        /* 4. persist */
        json j;
        j["forest"] = rf_.to_json();
        j["meta"]   = {
            {"trees",nTrees},{"depth",maxDepth},{"min_samples",minSamples},
            {"min_gain",minGain},{"sample_ratio",sampleRatio}};
        std::ofstream(path) << j.dump(2);

        loaded_ = true;

        /* 5. quick eval on DS_tr (avg runtime) -------------------- */
        eval_and_report(DS_tr, "training");
    }

    /* =======================  Predict  ====================== */
    std::vector<int> predict(const std::string& path,
                             const std::vector<Sample>& DS,
                             float /*tau unused*/ = 0) const override
    {
        ensure_loaded(path);
        std::vector<int> out; out.reserve(DS.size());
        for (auto& s:DS) {
            double d = rf_.predict(s.feat.data());   // 预测 gap
            out.push_back(d < 0);                    // 列更快?
        }
        return out;
    }

    double bal_acc(const std::string& path,
                   const std::vector<Sample>& DS,
                   float tau) const override
    {
        auto pr = predict(path,DS);
        long TP=0,TN=0,FP=0,FN=0;
        for (size_t i=0;i<DS.size();++i){
            bool col=pr[i], gt=DS[i].label;
            if(col&&gt)++TP; else if(col&&!gt)++FP;
            else if(!col&&gt)++FN; else ++TN;
        }
        double tpr=(TP+FN)?double(TP)/(TP+FN):0;
        double tnr=(TN+FP)?double(TN)/(TN+FP):0;
        return 0.5*(tpr+tnr);
    }

private:
    mutable RandomForestReg rf_{nTrees,maxDepth,minSamples,minGain,sampleRatio};
    mutable bool loaded_ = false;

    /* ------------ load-on-demand ---------- */
    void ensure_loaded(const std::string& p) const
    {
        if (loaded_) return;
        std::ifstream in(p);
        if (!in){ logE("cannot open model "+p); std::exit(1); }
        json j; in>>j;
        rf_.from_json(j.at("forest"));
        loaded_ = true;
    }

    /* ------------ runtime metrics ---------- */
    void eval_and_report(const std::vector<Sample>& DS,
                         const std::string& tag) const
    {
        double row_sum=0,col_sum=0,opt_sum=0,rf_sum=0,cost_sum=0,hyb_sum=0,fann_sum=0;
        long TP=0,FP=0,TN=0,FN=0;

        for (auto& s:DS)
        {
            double d = rf_.predict(s.feat.data());
            bool col_fast = (d<0);

            row_sum += s.row_t;
            col_sum += s.col_t;
            opt_sum += std::min(s.row_t,s.col_t);
            rf_sum  += col_fast ? s.col_t : s.row_t;
            cost_sum+= (s.qcost<COST_THR)? s.row_t : s.col_t;
            hyb_sum += (s.hybrid_pred==1)?s.col_t:s.row_t;
            fann_sum+= (s.fann_pred  ==1)?s.col_t:s.row_t;

            (col_fast ? (s.label?++TP:++FP)
                       : (s.label?++FN:++TN));
        }
        auto avg=[&](double v){ return v/DS.size(); };
        double bal=0.5*((TP+FN)?double(TP)/(TP+FN):0 +
                        (TN+FP)?double(TN)/(TN+FP):0);

        std::cout<<std::fixed<<std::setprecision(6)
                 <<"\n*** RF-Gap "<<tag<<" report ("<<DS.size()<<" samples) ***\n"
                 <<"BalAcc="<<bal<<"  Acc="<<double(TP+TN)/DS.size()<<"\n"
                 <<"Row only        : "<<avg(row_sum)<<"\n"
                 <<"Col only        : "<<avg(col_sum)<<"\n"
                 <<"Cost-Threshold  : "<<avg(cost_sum)<<"\n"
                 <<"HybridOpt       : "<<avg(hyb_sum)<<"\n"
                 <<"FANN baseline   : "<<avg(fann_sum)<<"\n"
                 <<"AI model (RF)   : "<<avg(rf_sum)<<"\n"
                 <<"Oracle (min)    : "<<avg(opt_sum)<<"\n";
    }
};

/* factory */
std::unique_ptr<IModel> make_rf() { return std::make_unique<RFModel>(); }
