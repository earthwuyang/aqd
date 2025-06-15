/************************************************************************************
 * Row-/Column-routing MLP implemented with Eigen (no FANN)
 *
 *   – Reads multiple “dataDir” directories and builds one dataset
 *   – 2-layer MLP  (8 → hidden → 1) with sigmoid activations
 *   – Online SGD, early-stopping (patience = 20) on val MSE
 *   – Saves / loads plain-text weights (compatible across platforms)
 *   – Reports precision / recall / AUC and runtime metrics
 *   – "--skip_train" loads an existing model and only evaluates
 *
 * Compile:
 *   g++ -std=c++11 -O2 -I/path/to/eigen3 -I/path/to/json/single_include \
 *       -o rowmlp_eigen rowmlp_eigen.cpp
 *
 * Example:
 *   ./rowmlp_eigen \
 *        --data_dirs=tpch_sf1_templates \
 *        --data_dirs=tpch_sf1_templates_index \
 *        --epochs=1000 --hidden_neurons=128 --lr=0.001 \
 *        --best_model_path=checkpoints/best_mlp.txt
 ************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include <dirent.h>
#include <sys/stat.h>
#include <unordered_map>
#include <set>
#include <unordered_set>  // for global operator collection

#include <Eigen/Dense>            //  ←— Eigen replaces FANN
#include "json.hpp"               //  ←— nlohmann/json single-include

using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/* ───────────────────────────  simple logging  ─────────────────────────── */
static void logInfo (const std::string &m){ std::cout  << "[INFO]  " << m << '\n'; }
static void logWarn (const std::string &m){ std::cerr  << "[WARN]  " << m << '\n'; }
static void logError(const std::string &m){ std::cerr  << "[ERROR] " << m << '\n'; }

/* ────────────────────────  misc filesystem helpers  ───────────────────── */
bool isDirectory(const std::string &p){ struct stat s; return (stat(p.c_str(),&s)==0)&&S_ISDIR(s.st_mode); }
bool fileExists (const std::string &p){ std::ifstream f(p); return f.good(); }

/* ─────────────────────────────  progress bar  ─────────────────────────── */
void showProgressBar(size_t cur,size_t tot,size_t w=50){
    double frac = (tot==0)?1.0:double(cur)/tot;
    size_t filled = size_t(frac*w);
    std::cout << "\r[";
    for(size_t i=0;i<filled;++i) std::cout << '=';
    for(size_t i=filled;i<w;++i) std::cout << ' ';
    std::cout << "] " << int(frac*100) << "% ("<<cur<<"/"<<tot<<")";
    if(cur>=tot) std::cout << '\n';
    std::fflush(stdout);
}

/* ─────────────────────────────  utilities  ────────────────────────────── */
double parsePossibleNumber(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string())               return std::stod(j[k].get<std::string>());
        if(j[k].is_number_float()||
           j[k].is_number_integer())       return j[k].get<double>();
    }catch(...){ logWarn("Invalid number for key: "+k);}
    return 0.0;
}
double convert_data_size_to_numeric(std::string s){
    if(s.empty()) return 0.0;
    s.erase(std::find_if(s.rbegin(),s.rend(),
         [](unsigned char c){return !std::isspace(c);} ).base(),s.end());
    char suff=s.back(); double f=1.0;
    if(suff=='G'){f=1e9; s.pop_back();}
    else if(suff=='M'){f=1e6; s.pop_back();}
    else if(suff=='K'){f=1e3; s.pop_back();}
    try{ return std::stod(s)*f; }catch(...){ return 0.0;}
}
inline double safeLog1p(double v){ return std::log1p(std::max(0.0,v)); }

/* ──────────────────  row-plan JSON → 8-D feature vector  ───────────────── */
bool parseRowPlanJSON(const std::string &path,float out[8],double &query_cost){
    std::fill(out,out+8,0.f);
    std::ifstream ifs(path); if(!ifs){ logWarn("Missing plan "+path); return false; }
    json j; try{ ifs>>j;}catch(...){ logWarn("JSON error "+path); return false; }
    if(!j.contains("query_block")){ logWarn("No query_block "+path); return false;}
    struct FC{ double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0; int n=0;} fc;
    std::function<void(const json&)> rec=[&](const json& b){
        if(b.is_object()){
            if(b.contains("table")){
                const auto&t=b["table"];
                double re=parsePossibleNumber(t,"rows_examined_per_scan");
                double rp=parsePossibleNumber(t,"rows_produced_per_join");
                double f =parsePossibleNumber(t,"filtered");
                double rc=0,ec=0,pc=0,dr=0;
                if(t.contains("cost_info")){
                    const auto&ci=t["cost_info"];
                    rc=parsePossibleNumber(ci,"read_cost");
                    ec=parsePossibleNumber(ci,"eval_cost");
                    pc=parsePossibleNumber(ci,"prefix_cost");
                    dr=convert_data_size_to_numeric(ci.value("data_read_per_join","0"));
                }
                if(re>=0&&rp>=0){ fc.re+=re; fc.rp+=rp; fc.f+=f;
                                  fc.rc+=rc; fc.ec+=ec; fc.pc+=pc; fc.dr+=dr; ++fc.n; }
            }
            for(const auto&kv:b.items()) rec(kv.value());
        }else if(b.is_array()) for(const auto&v:b) rec(v);
    };
    rec(j["query_block"]);
    query_cost=0.0;
    if(j["query_block"].contains("cost_info"))
        query_cost=parsePossibleNumber(j["query_block"]["cost_info"],"query_cost");
    if(fc.n==0) return true;
    auto avg=[&](double x){ return safeLog1p(x/fc.n); };
    out[0]=float(avg(fc.re)); out[1]=float(avg(fc.rp)); out[2]=float(avg(fc.f));
    out[3]=float(avg(fc.rc)); out[4]=float(avg(fc.ec)); out[5]=float(avg(fc.pc));
    out[6]=float(avg(fc.dr)); out[7]=float(safeLog1p(query_cost));
    return true;
}

// ───────────────── column-plan statistics (cached per dataset) ───────────
struct ColStats {
    double rows_c = 0, rows_s = 1, cost_c = 0, cost_s = 1;
    std::unordered_map<std::string,int> op2id;  // local operator IDs
};

static ColStats buildColStats(const std::string& dir) {
    ColStats st;
    std::string cache = dir + "/column_plan_statistics.json";
    if (fileExists(cache)) {
        try {
            json j; std::ifstream(cache) >> j;
            if (j.contains("op2id") && j["op2id"].size() > 0) {
                st.rows_c = j["rows_c"]; st.rows_s = j["rows_s"];
                st.cost_c = j["cost_c"]; st.cost_s = j["cost_s"];
                for (auto& kv : j["op2id"].items())
                    st.op2id[kv.key()] = kv.value();
                return st;
            } else {
                logWarn("Empty op2id in cache, rebuilding stats: " + cache);
            }
        } catch (...) {
            logWarn("Bad stats cache, rebuilding: " + cache);
        }
    }
    // scan column plans to compute medians and IQRs + collect op names
    logInfo("Scanning column plans in " + dir);
    // first collect files
    std::vector<std::string> files;
    std::string cdir = dir + "/column_plans";
    DIR* dp = opendir(cdir.c_str());
    if (!dp) { logWarn("No directory " + cdir); return st; }
    struct dirent* de;
    while ((de = readdir(dp))) {
        std::string fn = de->d_name;
        if (fn.size()>5 && fn.substr(fn.size()-5)==".json")
            files.push_back(fn);
    }
    closedir(dp);
    size_t total = files.size();
    logInfo("Found " + std::to_string(total) + " column plan files");

    std::vector<double> rows, cost;
    std::set<std::string> ops;
    // show progress bar while scanning
    for (size_t i = 0; i < total; ++i) {
        showProgressBar(i+1, total);
        const std::string& fn = files[i];
        std::ifstream f(cdir + "/" + fn);
        if (!f) continue;
        json j; try { f >> j; } catch(...) { continue; }
        if (!j.contains("plan") || !j["plan"].is_array()) continue;
        std::vector<json> stack(j["plan"].begin(), j["plan"].end());
        while (!stack.empty()) {
            json node = stack.back(); stack.pop_back();
            std::string opName = node.value("operator","UNK");
            ops.insert(opName);
            rows.push_back(safeLog1p(parsePossibleNumber(node, "esti_rows")));
            cost.push_back(safeLog1p(parsePossibleNumber(node, "esti_cost")));
            if (node.contains("children"))
                for (auto& c : node["children"]) stack.push_back(c);
        }
    }
    if (!rows.empty()) {
        auto mid = [](std::vector<double>& v){ std::nth_element(v.begin(), v.begin()+v.size()/2, v.end()); return v[v.size()/2]; };
        auto iqr = [](std::vector<double>& v){ size_t n=v.size(); std::nth_element(v.begin(), v.begin()+n/4, v.end()); double q1=v[n/4]; std::nth_element(v.begin()+n/4, v.begin()+3*n/4, v.end()); double q3=v[3*n/4]; return (q3-q1)/2.0; };
        st.rows_c = mid(rows);  st.cost_c = mid(cost);
        st.rows_s = std::max(1e-6, iqr(rows));
        st.cost_s = std::max(1e-6, iqr(cost));
        int id = 0;
        for (const auto& op : ops) st.op2id[op] = id++;
    }
    // cache to disk
    json dump{{"rows_c",st.rows_c},{"rows_s",st.rows_s},{"cost_c",st.cost_c},{"cost_s",st.cost_s},{"op2id",st.op2id}};
    std::ofstream(cache) << dump.dump(2);
    return st;
}

/* ─────────────── column-plan JSON → lightweight graph ------------------- */
struct Graph {
    int N=0;                     // #nodes
    std::vector<Eigen::Vector2d> x;        // (rows,cost) normed
    std::vector<std::vector<int>> adj;     // adjacency
    std::vector<int> op;                   // operator id
};
// parseColPlan now uses per-dir stats for numeric norms and a global op2id map
static Graph parseColPlan(const json& j,
                          const ColStats& st,
                          const std::unordered_map<std::string,int>& globalOp2id) {
    Graph g;
    if (!j.contains("plan") || !j["plan"].is_array()) return g;
    std::vector<const json*> stack{&j["plan"][0]};
    std::vector<std::pair<int,int>> edges;
    while (!stack.empty()) {
        const json* cur = stack.back(); stack.pop_back();
        int idx = g.N++;
        double rawR = safeLog1p(parsePossibleNumber(*cur, "esti_rows"));
        double rawC = safeLog1p(parsePossibleNumber(*cur, "esti_cost"));
        double nr = (rawR - st.rows_c) / st.rows_s;
        double nc = (rawC - st.cost_c) / st.cost_s;
        g.x.emplace_back(nr, nc);
        std::string opName = cur->value("operator", "UNK");
        auto it = globalOp2id.find(opName);
        int gid = (it != globalOp2id.end() ? it->second : globalOp2id.at("UNK"));
        g.op.push_back(gid);
        g.adj.emplace_back();
        if (cur->contains("children") && cur->at("children").is_array()) {
            for (const auto& c : (*cur)["children"]) {
                stack.push_back(&c);
                edges.emplace_back(idx, g.N);
                edges.emplace_back(g.N, idx);
            }
        }
    }
    for (auto& e : edges)
        if (e.first < g.N && e.second < g.N)
            g.adj[e.first].push_back(e.second);
    return g;
}

/* ────────────────────────  dataset definitions  ──────────────────────── */
struct Sample{
    float  feats[8];
    Graph colGraph;   
    int    label;          // 0,1,-1
    double row_time,col_time;
    double query_cost;
    int    hybrid_use_imci;
    int fann_use_imci;
};

// ExecutionPlanDataset now builds global operator map and loads per-dir stats
class ExecutionPlanDataset {
    std::vector<Sample> data_;
    std::unordered_map<std::string,int> globalOp2id_;
public:
    ExecutionPlanDataset(const std::vector<std::string>& dirs, double timeout) {
        // 1) build per-directory stats and collect all operator names
        std::vector<ColStats> statsList;
        std::unordered_set<std::string> allOps;
        for (const auto& d : dirs) {
            std::string dir = "/home/wuy/query_costs/" + d;
            ColStats cst = buildColStats(dir);
            statsList.push_back(cst);
            for (auto& kv : cst.op2id) allOps.insert(kv.first);
        }
        allOps.insert("UNK");
        // 2) build global operator ID map
        int oid = 0;
        for (const auto& op : allOps) globalOp2id_[op] = oid++;
        // 3) load samples, parsing both row and column plans
        int skip = 0, val = 0;
        for (size_t di = 0; di < dirs.size(); ++di) {
            const auto& d = dirs[di];
            std::string dir = "/home/wuy/query_costs/" + d;
            std::string csv = dir + "/query_costs.csv";
            if (!fileExists(csv)) { logWarn("Missing " + csv); continue; }
            // count lines
            size_t lines = 0;
            { std::ifstream f(csv); std::string l; while (std::getline(f, l)) ++lines; }
            std::ifstream in(csv);
            std::string header; std::getline(in, header);
            size_t cur = 0, tot = (lines ? lines-1 : 0);
            const ColStats& cst = statsList[di];

            std::string rpdir = dir + "/row_plans";
            std::string cpdir = dir + "/column_plans";
            std::string line;
            while (std::getline(in, line)) {
                ++cur; showProgressBar(cur, tot);
                if (line.empty()) continue;
                std::stringstream ss(line);
                std::string qid,lbl,rt,ct,hy,fy;
                std::getline(ss,qid,','); std::getline(ss,lbl,',');
                std::getline(ss,rt,',');  std::getline(ss,ct,','); std::getline(ss,hy,','); std::getline(ss,fy,',');
                if (qid.empty()||lbl.empty()) { ++skip; continue; }
                Sample rs{};
                rs.label           = std::stoi(lbl);
                rs.row_time        = rt.empty()? timeout : std::stod(rt);
                rs.col_time        = ct.empty()? timeout : std::stod(ct);
                if (std::fabs(rs.row_time - rs.col_time) < 1e-3) rs.label = -1;
                rs.hybrid_use_imci = hy.empty()?0:std::stoi(hy);
                rs.fann_use_imci = fy.empty()?0:std::stoi(fy);
                // row plan
                std::string rp = rpdir + "/" + qid + ".json";
                if (!fileExists(rp) || !parseRowPlanJSON(rp, rs.feats, rs.query_cost)) { ++skip; continue; }
                // column plan
                std::string cp = cpdir + "/" + qid + ".json";
                if (fileExists(cp)) {
                    std::ifstream cf(cp);
                    try { json cj; cf >> cj; rs.colGraph = parseColPlan(cj, cst, globalOp2id_); }
                    catch (...) { logWarn("JSON error: " + cp); }
                } else {
                    logWarn("Missing column plan " + cp);
                }
                data_.push_back(std::move(rs)); ++val;
            }
        }
        logInfo("Loaded " + std::to_string(val) + " samples, skipped " + std::to_string(skip));
    }
    size_t size()    const { return data_.size(); }
    const Sample& operator[](size_t i) const { return data_[i]; }
};




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









/* ───────────────────────  weighted random sampler  ───────────────────── */
class WeightedRandomSampler{
    std::discrete_distribution<> dist_;
    std::mt19937 rng_;
public:
    explicit WeightedRandomSampler(const std::vector<int>&labels){
        int c0=0,c1=0; for(int l:labels){ if(l==0)++c0; else if(l==1)++c1;}
        double w0 = c0?1.0/c0:1.0, w1=c1?1.0/c1:1.0;
        std::vector<double> w; w.reserve(labels.size());
        for(int l:labels) w.push_back(l==0?w0:(l==1?w1:0.0));
        dist_=std::discrete_distribution<>(w.begin(),w.end());
        rng_.seed(std::random_device{}());
    }
    size_t sample(){ return dist_(rng_); }
};

/* ───────────────────────── training sample struct ─────────────────────── */
struct TrainSample{ std::vector<float> x; float y; };

/* ───────────────────────  ROC / metrics helpers  ─────────────────────── */
double sigmoid(double z){ return 1.0/(1.0+std::exp(-z)); }
struct Metrics{
    double loss=0,acc=0,prec=0,rec=0,f1=0,auc=0,recN=0,recP=0,precN=0,precP=0;
    float  thr=0.5f;
};


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

    /* sigmoid helpers */
    static double sigmoid(double z)  { return 1.0 / (1.0 + std::exp(-z)); }
    static double dsigmoid(double a) { return a * (1.0 - a); }            // derivative

public:
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
            double          a2 = sigmoid(z2);

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
        double out = sigmoid(W2.dot(a1) + b2);
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



/* ───────────────────────  helper to split dataset  ───────────────────── */
std::pair<std::vector<size_t>,std::vector<size_t>>
splitDataset(size_t n,double train_ratio=0.8){
    std::vector<size_t> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::shuffle(idx.begin(),idx.end(),std::mt19937(std::random_device{}()));
    size_t k=size_t(train_ratio*n);
    return {{idx.begin(),idx.begin()+k},{idx.begin()+k,idx.end()}};
}
std::vector<TrainSample> prepareSamples(const ExecutionPlanDataset&ds,
                                        const std::vector<size_t>&idxs){
    std::vector<TrainSample> v;
    for(auto i:idxs){
        const auto& r=ds[i]; if(r.label==-1) continue;
        TrainSample ts; ts.x.assign(r.feats,r.feats+8); ts.y=float(r.label==1);
        v.push_back(std::move(ts));
    }
    return v;
}



int main(int argc, char* argv[]) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();

    //── CLI defaults ────────────────────────────────────────────────
    std::vector<std::string> dataDirs;
    int    epochs        = 10000;
    int    hiddenNeurons = 128;
    double lr            = 0.001;
    std::string bestModel= "checkpoints/best_mlp.net";
    bool   skipTrain     = false;
    bool   useCol        = false;

    //── Parse args ─────────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a.find("--data_dirs=")       == 0) dataDirs.push_back(a.substr(12));
        else if (a.find("--epochs=")          == 0) epochs        = std::stoi(a.substr(9));
        else if (a.find("--hidden_neurons=")  == 0) hiddenNeurons = std::stoi(a.substr(17));
        else if (a.find("--lr=")              == 0) lr            = std::stod(a.substr(5));
        else if (a.find("--best_model_path=") == 0) bestModel     = a.substr(18);
        else if (a == "--skip_train")                    skipTrain     = true;
        else if (a == "--use_col")                       useCol        = true;
    }

    if (useCol)
        bestModel = "checkpoints/best_mlp_row_column_gnn.net";

    if (dataDirs.empty()) {
        logWarn("No --data_dirs provided; using defaults.");
        dataDirs = {"tpch_sf1_templates","tpch_sf1_templates_index"};
    }

    //── Load dataset ────────────────────────────────────────────────
    ExecutionPlanDataset dataset(dataDirs, 60.0);
    if (dataset.size() < 30) {
        logError("Need at least 30 samples.");
        return 1;
    }

    //── Split 80/20 ─────────────────────────────────────────────────
    auto splits = splitDataset(dataset.size(), 0.8);
    auto &trainIdx = splits.first;
    auto &valIdx   = splits.second;

    //── Prepare train samples ───────────────────────────────────────
    std::vector<int> trainLabels;
    for (auto i : trainIdx) {
        int L = dataset[i].label;
        if (L != -1) trainLabels.push_back(L);
    }
    WeightedRandomSampler sampler(trainLabels);
    auto trainingSamples = prepareSamples(dataset, trainIdx);

    //── Build GIN + classifier ─────────────────────────────────────
    int maxOp = 0;
    for (size_t i = 0; i < dataset.size(); ++i)
        for (int op : dataset[i].colGraph.op)
            maxOp = std::max(maxOp, op);
    int nOps = maxOp + 1;
    GIN gin(nOps, hiddenNeurons);
    int inDim = useCol ? (8 + gin.hidden) : 8;
    EigenMLP model(inDim, hiddenNeurons, 1, lr);

    //── Load or train ────────────────────────────────────────────────
    if (skipTrain) {
        if (!fileExists(bestModel)) {
            logError("Checkpoint not found: " + bestModel);
            return 1;
        }
        model.load(bestModel);
        logInfo("Loaded model from " + bestModel);
    } else {
        logInfo("Training for up to " + std::to_string(epochs)
                + (useCol ? " epochs with GIN" : " epochs"));
        double bestValMSE = 1e30;
        int patience = 20, wait = 0;

        for (int ep = 1; ep <= epochs; ++ep) {
            std::shuffle(
                trainingSamples.begin(),
                trainingSamples.end(),
                std::mt19937((unsigned)std::random_device{}())
            );
            double trainMSE = 0.0;
            size_t total = trainingSamples.size();

            // show progress through this epoch
            for (size_t k = 0; k < total; ++k) {
                showProgressBar(k+1, total);

                auto &ts = trainingSamples[k];
                // base features
                std::vector<float> fv = ts.x;
                // append GIN embedding if requested
                if (useCol) {
                    size_t idx = trainIdx[k];
                    Eigen::VectorXd emb = gin.encode(dataset[idx].colGraph);
                    for (int d = 0; d < emb.size(); ++d)
                        fv.push_back((float)emb[d]);
                }
                // train head
                EigenMLP::Sample smp{fv, ts.y};
                model.train({smp});
                trainMSE += model.getMSE();

                // backprop into GIN (placeholder gradient)
                if (useCol) {
                    size_t idx = trainIdx[k];
                    Eigen::VectorXd d_out = Eigen::VectorXd::Zero(gin.hidden);
                    gin.backward_and_update(dataset[idx].colGraph, d_out, lr);
                }
            }
            trainMSE /= double(trainingSamples.size());

            // validation
            double valMSE = 0.0; int cnt = 0;
            for (auto ii : valIdx) {
                auto &rs = dataset[ii];
                if (rs.label < 0) continue;
                std::vector<float> fv(rs.feats, rs.feats+8);
                if (useCol) {
                    auto emb = gin.encode(rs.colGraph);
                    for (int d=0; d<emb.size(); ++d) fv.push_back((float)emb[d]);
                }
                float p = model.run(fv);
                double t = rs.label ? 1.0 : 0.0;
                valMSE += (p - t)*(p - t);
                ++cnt;
            }
            if (cnt) valMSE /= cnt;

            logInfo("Ep " + std::to_string(ep)
                    + " TrainMSE=" + std::to_string(trainMSE)
                    + " ValMSE="   + std::to_string(valMSE)
                    + " patience=" + std::to_string(wait));

            if (valMSE < bestValMSE) {
                bestValMSE = valMSE; wait = 0;
                std::string dir = bestModel.substr(0, bestModel.find_last_of("/\\"));
                system(("mkdir -p " + dir).c_str());
                model.save(bestModel);
                logInfo("→ saved new best model");
            } else if (++wait >= patience) {
                logInfo("Early stopping at ep " + std::to_string(ep));
                break;
            }
        }
        model.load(bestModel);
        logInfo("Reloaded best model.");
    }

    //── Final eval + confusion ───────────────────────────────────────
    int TP=0, FP=0, TN=0, FN=0;
    double rt_row=0, rt_col=0, rt_cost=0, rt_hyb=0, rt_fann=0, rt_opt=0, rt_ai=0;
    size_t nr=0, nc=0, ncst=0, nh=0, nf=0, no=0, nai=0;
    auto online_avg = [&](double &acc, double v, size_t &n) {
        acc = (acc*n + v) / double(n+1);
        ++n;
    };

    for (size_t i = 0; i < dataset.size(); ++i) {
        auto &rs = dataset[i];
        if (rs.label < 0) continue;
        online_avg(rt_row, rs.row_time, nr);
        online_avg(rt_col, rs.col_time, nc);
        online_avg(rt_cost,
                   (rs.query_cost > 50000.0 ? rs.col_time : rs.row_time),
                   ncst);
        online_avg(rt_hyb,
                   (rs.hybrid_use_imci ? rs.col_time : rs.row_time),
                   nh);
        online_avg(rt_fann,
                    (rs.fann_use_imci ? rs.col_time : rs.row_time),
                   nf);
        online_avg(rt_opt,
                   std::min(rs.row_time, rs.col_time),
                   no);

        std::vector<float> fv(rs.feats, rs.feats+8);
        if (useCol) {
            auto emb = gin.encode(rs.colGraph);
            for (int d=0; d<emb.size(); ++d)
                fv.push_back((float)emb[d]);
        }
        bool pred = model.run(fv) >= 0.5f;
        online_avg(rt_ai,
                   (pred ? rs.col_time : rs.row_time),
                   nai);

        if      (rs.label==1 &&  pred) ++TP;
        else if (rs.label==1 && !pred) ++FN;
        else if (rs.label==0 &&  pred) ++FP;
        else                            ++TN;
    }

    logInfo("Confusion  TN=" + std::to_string(TN)
          + " FP=" + std::to_string(FP)
          + " FN=" + std::to_string(FN)
          + " TP=" + std::to_string(TP));

    double tot = double(TP+TN+FP+FN);
    logInfo("Acc="  + std::to_string((TP+TN)/tot)
          + " Prec=" + std::to_string(TP?double(TP)/(TP+FP):0)
          + " Rec="  + std::to_string(TP?double(TP)/(TP+FN):0)
          + " F1="   + std::to_string(
                        (TP+FP && TP+FN)
                          ? 2.0*TP/(2.0*TP+FP+FN)
                          : 0.0));

    logInfo(std::string("Avg runtimes (s/query):")
          + " row="  + std::to_string(rt_row)
          + " col="  + std::to_string(rt_col)
          + " cost=" + std::to_string(rt_cost)
          + " hyb="  + std::to_string(rt_hyb)
          + " fann=" + std::to_string(rt_fann)
          + " opt="  + std::to_string(rt_opt)
          + " ai="   + std::to_string(rt_ai));

    double wall = std::chrono::duration_cast<std::chrono::milliseconds>(
                      Clock::now() - t_start
                  ).count()/1000.0;
    logInfo("Total wall-clock: " + std::to_string(wall) + " s");
    return 0;
}
