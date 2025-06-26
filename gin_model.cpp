#include "common.hpp"          // Sample, log helpers
#include <Eigen/Dense>
#include <unordered_map>
#include "model_iface.hpp"
#include <Eigen/Dense>
#include <random>




/* ─────────  tiny, header-only 2-layer MLP using Eigen  ────────── */
class EigenMLP
{
    /* network dims */
    int in_, hid_, out_;
    double lr_;                        // learning-rate

    /* parameters */
    Eigen::MatrixXd     W1;            // (hid_ × in_)
    Eigen::VectorXd     b1;            // (hid_)
    Eigen::RowVectorXd  W2;            // (1 × hid_)
    double              b2;            // scalar bias

    /* running loss for getMSE() */
    double last_mse_{};

public:
    /* sigmoid helpers */
    static double sigmoid(double z)  { return 1.0 / (1.0 + std::exp(-z)); }
    static double dsigmoid(double a) { return a * (1.0 - a); }            // derivative


    /* ------------ ctor with random weight initialisation ------------- */
    EigenMLP(int n_in, int n_hid, int n_out, double lr)
        : in_(n_in), hid_(n_hid), out_(n_out), lr_(lr),
          W1(n_hid, n_in), b1(n_hid), W2(n_hid), b2(0.0)
    {
        /* capture-less lambda for Eigen::NullaryExpr (avoids functor error) */
        auto randInit = []() {
            static thread_local std::mt19937 gen{std::random_device{}()};
            static thread_local std::uniform_real_distribution<double> dist(-0.1, 0.1);
            return dist(gen);
        };

        W1 = Eigen::MatrixXd ::NullaryExpr(hid_, in_,  randInit);
        b1 = Eigen::VectorXd ::NullaryExpr(hid_,       randInit);
        W2 = Eigen::RowVectorXd::NullaryExpr(hid_,     randInit);
        /* b2 already zero */
    }

    /* ---------- light-weight getters / setters ---------- */
    Eigen::MatrixXd&        W1m()       { return W1; }
    const Eigen::MatrixXd&  W1m() const { return W1; }

    Eigen::VectorXd&        b1v()       { return b1; }
    const Eigen::VectorXd&  b1v() const { return b1; }

    Eigen::RowVectorXd&     W2m()       { return W2; }
    const Eigen::RowVectorXd& W2m() const { return W2; }

    double&                 b2s()       { return b2; }
    double                  b2s() const { return b2; }

    int     in_dim()  const { return in_;  }
    int     hid_dim() const { return hid_; }
    double  lr()      const { return lr_;  }

    /* ------------------------- training step ------------------------- */
    struct Sample { const std::vector<float>& x; float y; };   // lightweight view

    void train(const std::vector<Sample>& batch)
    {
        for (const auto& s : batch)
        {
            /* ---- forward ---- */
            Eigen::VectorXd x(in_);
            for (int i = 0; i < in_; ++i) x(i) = s.x[i];

            Eigen::VectorXd z1 = W1 * x + b1;                 // (hid)
            Eigen::VectorXd a1 = z1.unaryExpr(&sigmoid);      // (hid)
            double          z2 = W2.dot(a1) + b2;             // scalar
            double a2=z2;

            double err  = a2 - s.y;
            last_mse_   = err * err;                          // for getMSE()

            /* ---- backward ---- */
            double delta2 = err * dsigmoid(a2);               // output delta

            Eigen::RowVectorXd gradW2 = delta2 * a1.transpose();   // (1×hid)
            double             gradb2 = delta2;

            Eigen::VectorXd delta1 = (W2.transpose() * delta2)
                                     .cwiseProduct(a1.unaryExpr(&dsigmoid));  // (hid)

            Eigen::MatrixXd gradW1 = delta1 * x.transpose();  // (hid×in)
            Eigen::VectorXd gradb1 = delta1;

            /* ---- SGD update ---- */
            W2 -= lr_ * gradW2;
            b2 -= lr_ * gradb2;
            W1 -= lr_ * gradW1;
            b1 -= lr_ * gradb1;
        }
    }

    /* --------------------------- inference --------------------------- */
    float run(const std::vector<float>& in) const
    {
        Eigen::VectorXd x(in_);
        for (int i = 0; i < in_; ++i) x(i) = in[i];

        Eigen::VectorXd a1 = (W1 * x + b1).unaryExpr(&sigmoid);
        double out = W2.dot(a1) + b2;
        return static_cast<float>(out);
    }

    double getMSE() const { return last_mse_; }

    /* ------------------------- save / load -------------------------- */
    void save(const std::string& path) const
    {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot write " + path);
        f << in_ << ' ' << hid_ << ' ' << out_ << '\n';
        for (int r = 0; r < hid_; ++r) for (int c = 0; c < in_; ++c) f << W1(r,c) << ' ';
        f << '\n';
        for (int i = 0; i < hid_; ++i) f << b1(i) << ' ';     f << '\n';
        for (int i = 0; i < hid_; ++i) f << W2(i) << ' ';     f << '\n';
        f << b2 << '\n';
    }

    void load(const std::string& path)
    {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot read " + path);
        int ni, nh, no; f >> ni >> nh >> no;
        if (ni != in_ || nh != hid_ || no != out_)
            throw std::runtime_error("Shape mismatch in " + path);

        for (int r = 0; r < hid_; ++r) for (int c = 0; c < in_; ++c) f >> W1(r,c);
        for (int i = 0; i < hid_; ++i) f >> b1(i);
        for (int i = 0; i < hid_; ++i) f >> W2(i);
        f >> b2;
    }
};



/* ─────────  GIN encoder  (exactly your original struct)  ──────── */


// ──────────────────────  GIN encoder (with backprop!)  ──────────────────────
struct GIN {
    int in_dim;            // = 2 + 8  (2 numeric node features + 8‐dim op embedding)
    int hidden;
    int n_ops;

    Eigen::MatrixXd W1;    // (hidden × in_dim)
    Eigen::MatrixXd b1;    // (hidden × 1)
    Eigen::MatrixXd W2;    // (hidden × hidden)
    Eigen::MatrixXd b2;    // (hidden × 1)
    Eigen::MatrixXd opEmb; // (8 × n_ops)

    std::mt19937               rng;
    std::normal_distribution<> nd;

    GIN(int n_ops_, int hidden_)
      : in_dim(2+8), hidden(hidden_), n_ops(n_ops_),
        W1(hidden, in_dim), b1(hidden,1),
        W2(hidden, hidden), b2(hidden,1),
        opEmb(8, n_ops_),
        rng(123), nd(0.0, 0.1)
    {
        auto init = [&](Eigen::MatrixXd &m){
            for (int i=0;i<m.size();++i) m.data()[i] = nd(rng);
        };
        init(W1); init(b1);
        init(W2); init(b2);
        init(opEmb);
    }

    // ReLU
    static Eigen::VectorXd relu(const Eigen::VectorXd &v) {
        return v.cwiseMax(0.0);
    }

    // forward only
    Eigen::VectorXd encode(const Graph &g) const {
        if (g.N==0) return Eigen::VectorXd::Zero(hidden);

        std::vector<Eigen::VectorXd> z1(g.N), h(g.N);
        Eigen::VectorXd out = Eigen::VectorXd::Zero(hidden);

        // 1) per-node first layer
        for (int i=0;i<g.N;++i) {
            Eigen::VectorXd xin(in_dim);
            xin.head(2)        = g.x[i];
            xin.tail(8)        = opEmb.col(g.op[i]);
            z1[i] = relu(W1*xin + b1.col(0));
        }
        // 2) message‐passing + second layer
        for (int i=0;i<g.N;++i) {
            Eigen::VectorXd agg = Eigen::VectorXd::Zero(hidden);
            for (int nb:g.adj[i]) agg += z1[nb];
            h[i] = relu(W2*agg + b2.col(0));
            out += h[i];
        }
        return out / double(g.N);
    }

    // backprop + sgd update.  d_out = ∂L/∂(GIN_embedding), lr = learning rate
    void backward_and_update(const Graph &g,
                             const Eigen::VectorXd &d_out,
                             double lr)
    {
        if (g.N==0) return;

        // re-compute and stash all forward intermediates
        std::vector<Eigen::VectorXd> xin(g.N), z1(g.N), agg(g.N), pre2(g.N), h(g.N);
        for (int i=0;i<g.N;++i) {
            xin[i].resize(in_dim);
            xin[i].head(2)  = g.x[i];
            xin[i].tail(8)  = opEmb.col(g.op[i]);
            z1[i]   = relu(W1*xin[i] + b1.col(0));
            // will fill agg & pre2 below
        }
        for (int i=0;i<g.N;++i) {
            agg[i]  = Eigen::VectorXd::Zero(hidden);
            for (int nb:g.adj[i]) agg[i] += z1[nb];
            pre2[i] = W2*agg[i] + b2.col(0);
            h[i]    = relu(pre2[i]);
        }

        // allocate grads
        Eigen::MatrixXd gW2=Eigen::MatrixXd::Zero(W2.rows(),W2.cols()),
                        gb2=Eigen::MatrixXd::Zero(b2.rows(),b2.cols()),
                        gW1=Eigen::MatrixXd::Zero(W1.rows(),W1.cols()),
                        gb1=Eigen::MatrixXd::Zero(b1.rows(),b1.cols()),
                        gOp=Eigen::MatrixXd::Zero(opEmb.rows(),opEmb.cols());

        // gradient at each h[i] = d_out / N
        Eigen::VectorXd dh = d_out / double(g.N);

        // backprop nodewise
        for (int i=0;i<g.N;++i) {
            // d_pre2 = dh * relu'(pre2)
            Eigen::VectorXd relu_mask2 = (pre2[i].array()>0.0)
                                          .template cast<double>()
                                          .matrix();
            Eigen::VectorXd d_pre2 = dh.cwiseProduct(relu_mask2);

            // w2,b2 grads
            gW2 += d_pre2 * agg[i].transpose();
            gb2.col(0) += d_pre2;

            // back to agg → distribute to each z1[nb]
            Eigen::VectorXd d_agg = W2.transpose()*d_pre2;
            for (int nb:g.adj[i]) {
                // backprop z1[nb]
                Eigen::VectorXd relu_mask1 = ((W1*xin[nb] + b1.col(0)).array()>0.0)
                                              .template cast<double>()
                                              .matrix();
                Eigen::VectorXd d_pre1 = d_agg.cwiseProduct(relu_mask1);

                gW1 += d_pre1 * xin[nb].transpose();
                gb1.col(0) += d_pre1;

                // back to xin[nb] → split: tail(8) → opEmb; head(2) ignored
                Eigen::VectorXd dxin = W1.transpose()*d_pre1;
                gOp.col(g.op[nb]) += dxin.tail(8);
            }
        }

        // SGD step
        W2    -= lr * gW2;
        b2    -= lr * gb2;
        W1    -= lr * gW1;
        b1    -= lr * gb1;
        opEmb -= lr * gOp;
    }
};

// /* ---------- tiny helpers ---------- */
// static std::vector<float>
// make_feature(const Sample& s, const GIN& gin, bool use_col)
// {
//     std::vector<float> feat(s.feat.begin(), s.feat.begin()+8);
//     if (use_col && s.colGraph.N) {
//         Eigen::VectorXd emb = gin.encode(s.colGraph);
//         for (int i = 0; i < emb.size(); ++i) feat.push_back(float(emb[i]));
//     }
//     return feat;
// }



/* ─────────────────────────  IModel wrapper  ───────────────────── */
class GINModel : public IModel
{
public:                       /* ---- IModel API ---- */
    void train (const std::vector<Sample>& DS_tr,
                const std::vector<Sample>& DS_val,
                const std::string&        model_path,
                const TrainOpt&           opt) override;

    std::vector<int> predict(const std::string&        model_path,
                             const std::vector<Sample>& DS,
                             float                     tau = 0.0f) const override;

    double bal_acc (const std::string&         model_path,
                    const std::vector<Sample>& DS,
                    float                      tau = 0.0f) const override;

private:
    /* runtime state (kept only while process lives) */
    mutable GIN       gin_{1, 128};   // will re-init in train/load
    mutable EigenMLP   mlp_{8, 128, 1, 0.001};
    mutable bool      loaded_ = false;

    /* helpers */
    void save(const std::string& path) const;
    void load(const std::string& path) const;
    static std::vector<float>
        make_feature(const Sample& s, const GIN& gin, bool USE_COL);
};

std::vector<float>
GINModel::make_feature(const Sample& s, const GIN& gin, bool USE_COL)
{
    std::vector<float> x(s.feat.begin(), s.feat.end());   // 102-d base
    if (USE_COL && s.colGraph.N) {
        Eigen::VectorXd emb = gin.encode(s.colGraph);
        x.reserve(x.size() + emb.size());
        for (int i = 0; i < emb.size(); ++i) x.push_back(float(emb[i]));
    }
    return x;
}

/* ---- 1. 训练：你之前的“真实反传”版本即可，这里省略 ... --------------- */
/*  .. 采用你上一次给出的 train() 正文。不需要再动，它只会用到下面的 save(). */

/* ---- 2. 保存 ----------------------------------------------------------- */
void GINModel::save(const std::string& p) const
{
    json j;
    /* meta */
    j["meta"] = { {"hid", gin_.hidden}, {"ops", gin_.n_ops} };

    /* gin weights ── 一律扁平化存 */
    auto dump = [](const auto& M) {
        return std::vector<double>(M.data(), M.data() + M.size());
    };
    j["gin"] = {
        {"W1", dump(gin_.W1)}, {"b1", dump(gin_.b1)},
        {"W2", dump(gin_.W2)}, {"b2", dump(gin_.b2)},
        {"op", dump(gin_.opEmb)}
    };

    /* mlp weights + dims */
    j["mlp"] = {
        {"in",  mlp_.in_dim()}, {"hid", mlp_.hid_dim()}, {"lr", mlp_.lr()},
        {"W1", dump(mlp_.W1m())},
        {"b1", dump(mlp_.b1v())},
        {"W2", dump(mlp_.W2m())},
        {"b2", mlp_.b2s()}
    };

    std::ofstream(p) << j.dump(2);
}

/* ---- 3. 载入 ----------------------------------------------------------- */
void GINModel::load(const std::string& p) const
{
    if (loaded_) return;
    json j; std::ifstream(p) >> j;

    /* ---------- GIN ---------- */
    int hid = j["meta"]["hid"], nOps = j["meta"]["ops"];
    const_cast<GIN&>(gin_) = GIN(nOps, hid);

    auto fillM = [](auto& M, const std::vector<double>& v){
        std::copy(v.begin(), v.end(), M.data());
    };
    fillM(gin_.W1 , j["gin"]["W1"].get<std::vector<double>>());
    fillM(gin_.b1 , j["gin"]["b1"].get<std::vector<double>>());
    fillM(gin_.W2 , j["gin"]["W2"].get<std::vector<double>>());
    fillM(gin_.b2 , j["gin"]["b2"].get<std::vector<double>>());
    fillM(gin_.opEmb , j["gin"]["op"].get<std::vector<double>>());

    /* ---------- MLP ---------- */
    int in  = j["mlp"]["in"],  h = j["mlp"]["hid"];
    double lr = j["mlp"]["lr"];
    const_cast<EigenMLP&>(mlp_) = EigenMLP(in, h, 1, lr);

    fillM(mlp_.W1m(), j["mlp"]["W1"].get<std::vector<double>>());
    fillM(mlp_.b1v(), j["mlp"]["b1"].get<std::vector<double>>());
    fillM(mlp_.W2m(), j["mlp"]["W2"].get<std::vector<double>>());
    const_cast<double&>(mlp_.b2s()) =
        j["mlp"]["b2"].get<double>();

    loaded_ = true;
}

/*───────────────────────────────────────────────────────────────*/
/*  GINModel::train – full implementation with “real” GIN-BP     */
/*───────────────────────────────────────────────────────────────*/
void GINModel::train(const std::vector<Sample>& DS_tr,
                     const std::vector<Sample>& DS_val,
                     const std::string&        path,
                     const TrainOpt&           opt)
{
    /* 1) ── 仅加载 ───────────────────────────────────────────────*/
    if (opt.skip_train) {                // --skip_train ⇒ 直接 restore
        load(path);
        return;
    }

    /* 2) ── run-time shape init  (hidden / #ops) ────────────────*/
    int maxOp = 0;
    for (auto& s : DS_tr)
        for (int id : s.colGraph.op) maxOp = std::max(maxOp, id);

    const int nOps   = maxOp + 1;
    const int hidden = (opt.hidden1 > 0) ? opt.hidden1 : 128;
    const bool USE_COL = true;                       // 始终用列图
    const int  inDim   = USE_COL ? (8 + hidden) : 8;

    gin_ = GIN(nOps, hidden);
    mlp_ = EigenMLP(inDim, hidden, 1, opt.lr);

    /* 3) ── 简单 class-balancing 采样器 ─────────────────────────*/
    std::vector<size_t> P, N;
    for (size_t i = 0; i < DS_tr.size(); ++i)
        (DS_tr[i].label ? P : N).push_back(i);

    std::mt19937 rng(123);
    std::uniform_int_distribution<> up(0, int(P.size()) - 1),
                                    un(0, int(N.size()) - 1);

    /* 4) ── epoch 训练循环  + 真实 GIN 反传 ─────────────────── */
    double best = 1e30;
    int    wait = 0;

    for (int ep = 1; ep <= opt.epochs; ++ep)
    {
        double train_mse = 0.0;
        int train_cnt = 0;
        const size_t steps = DS_tr.size();

        const double CLIP = 1e3;      // 1. 设置裁剪阈值

        for (size_t st = 0; st < steps; ++st)
        {
            progress("train-gin: ", st, steps);
            /* ---------- 取样本 ---------- */
            const size_t idx  = (st & 1) ? P[up(rng)] : N[un(rng)];
            const auto&  samp = DS_tr[idx];

            /* ---------- 4.1 拼输入向量 x ---------- */
            Eigen::VectorXd emb = USE_COL ? gin_.encode(samp.colGraph)
                                        : Eigen::VectorXd::Zero(0);

            Eigen::VectorXd x(inDim);
            for (int i = 0; i < 8; ++i) {
                float v = samp.feat[i];
                x(i) = std::isfinite(v) ? v : 0.0f;           // 缺失值→0
            }
            if (USE_COL)
                for (int i = 0; i < hidden; ++i) x(8 + i) = emb[i];

            /* ---------- 4.2 前向 ---------- */
            Eigen::VectorXd z1 = mlp_.W1m() * x + mlp_.b1v();   // hidden pre-act
            Eigen::VectorXd a1 = z1.unaryExpr(&EigenMLP::sigmoid);         // hidden act
            double z2 = mlp_.W2m().dot(a1) + mlp_.b2s();        // output pre-act
            double a2 = z2;                                     // 纯回归，无 sigmoid

            /* ---------- 4.2 计算安全标签 ---------- */
            auto safe_log1p = [](double v){
                return (v > -1.0 && std::isfinite(v)) ? std::log1p(v) : NAN;
            };
            double y = safe_log1p(samp.col_t) - safe_log1p(samp.row_t);
            if (!std::isfinite(y)) continue;        // 跳过脏样本

            /* ---------- 4.3 误差与在线均值 ---------- */
            double err = a2 - y;
            if (!std::isfinite(err)) continue;      // 额外保护

            train_mse += ((err*err) - train_mse) / double(++train_cnt);

            /* ---------- 4.4 反向 ---------- */
            /* 用 max/min 做梯度裁剪，等价于 clamp(err, -CLIP, CLIP) */
            double delta2 = std::max(-CLIP, std::min(err, CLIP));

            mlp_.W2m() -= opt.lr * (delta2 * a1.transpose());
            mlp_.b2s() -= opt.lr *  delta2;

            Eigen::VectorXd sigma_prime = a1.array() * (1.0 - a1.array());
            Eigen::VectorXd delta1 =
                    (mlp_.W2m().transpose() * delta2).cwiseProduct(sigma_prime);

            mlp_.W1m() -= opt.lr * (delta1 * x.transpose());
            mlp_.b1v() -= opt.lr *  delta1;

            /* ---------- 4.5 反传至 GIN ---------- */
            if (USE_COL && samp.colGraph.N) {
                Eigen::VectorXd d_emb = delta1.tail(hidden);
                gin_.backward_and_update(samp.colGraph, d_emb, opt.lr);
            }
        }

        /* 5) ── 每个 epoch 做一次验证 + Early-Stopping ───────── */
        double val_mse = 0.0; int cnt = 0;
        for (auto& s : DS_val)
        {
            if (s.label < 0) continue;
            auto feat = make_feature(s, gin_, USE_COL);
            double p  = mlp_.run(feat);
            // double t  = s.label ? 1.0 : 0.0;
            double t = log1p(s.col_t) - log1p(s.row_t);
            double err = p - t;
            val_mse  += ((err*err) - val_mse) / (++cnt);
            
            progress("valid", cnt, DS_val.size());
        }

        logI("GIN-EP " + std::to_string(ep) + " train-MSE=" + std::to_string(train_mse) +
             "  val-MSE=" + std::to_string(val_mse));

        /* 保存最优 – 简单 patience 早停 */
        if (val_mse < best - 1e-6) { best = val_mse; wait = 0; save(path); }
        else if (++wait >= opt.patience) {
            logI("early-stopped at epoch " + std::to_string(ep));
            break;
        }
    }


    /* ---------- 6. 训练结束后 : 多策略平均运行时 ---------------- */
    auto predict_col = [&](const Sample& s)->bool {
        auto feat = make_feature(s, gin_, USE_COL);
        float p   = mlp_.run(feat);          // GIN-MLP 概率
        return p > 0.0f;                     // τ=0 ⇒ 最大化 BalAcc
    };

    long   TP=0, FP=0, TN=0, FN=0;
    double r_row=0, r_col=0, r_opt=0,
           r_costthr=0, r_hopt=0, r_fann=0, r_gin=0;

    for (const auto& s : DS_tr)
    {
        bool col_gin  = predict_col(s);        // ← 当前 GIN 模型
        bool col_hopt = (s.hybrid_pred == 1);  // Hybrid-Opt baseline
        bool col_fann = (s.fann_pred   == 1);  // “fann_use_imci”
        bool use_row  = (s.qcost < COST_THR);  // cost threshold

        r_row     += s.row_t;
        r_col     += s.col_t;
        r_opt     += std::min(s.row_t, s.col_t);
        r_costthr += use_row   ? s.row_t : s.col_t;
        r_hopt    += col_hopt  ? s.col_t : s.row_t;
        r_fann    += col_fann  ? s.col_t : s.row_t;
        r_gin     += col_gin   ? s.col_t : s.row_t;

        /* 仅对 GIN 统计分类指标 */
        (col_gin ? (s.label?++TP:++FP)
                 : (s.label?++FN:++TN));
    }

    auto avg = [&](double v){ return v / DS_tr.size(); };
    std::cout << std::fixed << std::setprecision(6)
              << "\n*** GIN evaluation on " << DS_tr.size()
              << " samples ***\n"
              << "Acc="      << double(TP+TN)/DS_tr.size()
              << "  BalAcc=" << 0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
              << "  F1="     << (TP?2.0*TP/(2*TP+FP+FN):0) << '\n'
              << "Row only        : " << avg(r_row)     << '\n'
              << "Col only        : " << avg(r_col)     << '\n'
              << "Cost-Threshold  : " << avg(r_costthr) << '\n'
              << "HybridOpt       : " << avg(r_hopt)    << '\n'
              << "FANN baseline   : " << avg(r_fann)    << '\n'
              << "AI Model (GIN)  : " << avg(r_gin)     << '\n'
              << "Oracle (min)    : " << avg(r_opt)     << "\n\n";
}

/* --------------------------------------------------- */
std::vector<int>
GINModel::predict(const std::string&        path,
                  const std::vector<Sample>&DS,
                  float                     tau) const
{
    if (!loaded_) load(path);
    const bool USE_COL = true;
    std::vector<int> out; out.reserve(DS.size());
    for (auto& s: DS){
        auto x = make_feature(s, gin_, USE_COL);
        // float p = mlp_.run(x);
        // out.push_back(p > tau);
        double g = mlp_.run(x);
        out.push_back(g<0);
    }
    return out;
}

double GINModel::bal_acc(const std::string&         model_path,
                         const std::vector<Sample>& DS, float tau) const
{
    if (!loaded_) load(model_path);           // lazy restore

    const bool USE_COL = true;
    long TP=0, TN=0, FP=0, FN=0;

    for (const auto& s : DS)
    {
        /* ① 拼特征 → 预测 gap̂ */
        auto feat   = make_feature(s, gin_, USE_COL);
        double gap = mlp_.run(feat);         // >0 ⇒ row faster, <0 ⇒ column faster

        bool col_pred  = (gap < 0.0);        // 我们的决策
        bool col_truth = s.label;             // 真实标签

        (col_pred ? (col_truth?++TP:++FP)
                  : (col_truth?++FN:++TN));
    }
    double tpr = (TP+FN)? double(TP)/(TP+FN) : 0.0;
    double tnr = (TN+FP)? double(TN)/(TN+FP) : 0.0;
    return 0.5*(tpr + tnr);                   // Balanced Accuracy
}


std::unique_ptr<IModel> make_gin()
{
    return std::make_unique<GINModel>();
}