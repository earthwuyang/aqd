/************************************************************************************
 * row_column_gnn.cpp
 *
 * Dual‐regression Row/Column router:
 *
 *   – Row‐time regressor: 8 → hidden → 1 MLP (Eigen)
 *   – Column‐time regressor: GIN encoder + linear head
 *   – Online SGD, early‐stopping on validation MSE
 *   – Classify by comparing predicted times
 *   – "--skip_train" skips training and just evaluates (not fully implemented)
 *
 * Compile:
 *   g++ -std=c++11 -O2 -I/path/to/eigen3 -I/path/to/json/single_include \
 *       -o row_column_gnn row_column_gnn.cpp
 *
 * Example:
 *   ./row_column_gnn --data_dirs=tpch_sf1 --epochs=1000 --hidden_neurons=128 \
 *     --lr=0.001 --best_model_path=checkpoints/best_model.json --skip_train
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
#include <unordered_set>
#include <set>

#include <Eigen/Dense>
#include "json.hpp"

using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/* ────────────────────────────── logging ────────────────────────────── */
static void logInfo (const std::string &m){ std::cout  << "[INFO]  " << m << '\n'; }
static void logWarn (const std::string &m){ std::cerr  << "[WARN]  " << m << '\n'; }
static void logError(const std::string &m){ std::cerr  << "[ERROR] " << m << '\n'; }

/* ─────────────────────────── filesystem ──────────────────────────── */
bool isDirectory(const std::string &p){
    struct stat s; return (stat(p.c_str(),&s)==0)&&S_ISDIR(s.st_mode);
}
bool fileExists(const std::string &p){
    std::ifstream f(p); return f.good();
}

/* ─────────────────────────── progress ──────────────────────────── */
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

/* ────────────────────────── utilities ────────────────────────── */
double parsePossibleNumber(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string()) return std::stod(j[k].get<std::string>());
        if(j[k].is_number()) return j[k].get<double>();
    } catch(...){
        logWarn("Invalid number for key: "+k);
    }
    return 0.0;
}
double convert_data_size_to_numeric(std::string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    char suff=s.back(); double f=1.0;
    if(suff=='G'){f=1e9; s.pop_back();}
    else if(suff=='M'){f=1e6; s.pop_back();}
    else if(suff=='K'){f=1e3; s.pop_back();}
    try{ return std::stod(s)*f; } catch(...){ return 0.0;}
}
inline double safeLog1p(double v){ return std::log1p(std::max(0.0,v)); }

/* ──────────────── parseRowPlanJSON (8‐D) ───────────────── */
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
                if(re>=0&&rp>=0){
                    fc.re+=re; fc.rp+=rp; fc.f+=f;
                    fc.rc+=rc; fc.ec+=ec; fc.pc+=pc; fc.dr+=dr;
                    ++fc.n;
                }
            }
            for(const auto&kv:b.items()) rec(kv.value());
        } else if(b.is_array()){
            for(const auto&v:b) rec(v);
        }
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

/* ──────────────────── column planning stats ──────────────────── */
struct ColStats {
    double rows_c=0, rows_s=1, cost_c=0, cost_s=1;
    std::unordered_map<std::string,int> op2id;
};
static ColStats buildColStats(const std::string& dir) {
    ColStats st;
    std::string cache = dir+"/column_plan_statistics.json";
    if(fileExists(cache)){
        try{
            json j; std::ifstream(cache)>>j;
            if(j.contains("op2id") && j["op2id"].size()>0){
                st.rows_c=j["rows_c"];
                st.rows_s=j["rows_s"];
                st.cost_c=j["cost_c"];
                st.cost_s=j["cost_s"];
                for(auto&kv:j["op2id"].items())
                    st.op2id[kv.key()]=kv.value();
                return st;
            }
        }catch(...){
            logWarn("Bad stats cache, rebuilding: "+cache);
        }
    }
    logInfo("Scanning column plans in "+dir);
    std::vector<std::string> files;
    std::string cdir=dir+"/column_plans";
    DIR* dp = opendir(cdir.c_str());
    if(!dp){ logWarn("No directory "+cdir); return st; }
    struct dirent* de;
    while((de=readdir(dp))){
        std::string fn=de->d_name;
        if(fn.size()>5 && fn.substr(fn.size()-5)==".json")
            files.push_back(fn);
    }
    closedir(dp);
    size_t total=files.size();
    logInfo("Found "+std::to_string(total)+" column plan files");

    std::vector<double> rows,cost;
    std::set<std::string> ops;
    for(size_t i=0;i<total;++i){
        showProgressBar(i+1,total);
        std::ifstream f(cdir+"/"+files[i]);
        if(!f) continue;
        json j; try{ f>>j; }catch(...){ continue; }
        if(!j.contains("plan")||!j["plan"].is_array()) continue;
        std::vector<json> stack(j["plan"].begin(),j["plan"].end());
        while(!stack.empty()){
            json node=stack.back(); stack.pop_back();
            std::string op=node.value("operator","UNK");
            ops.insert(op);
            rows.push_back(safeLog1p(parsePossibleNumber(node,"esti_rows")));
            cost.push_back(safeLog1p(parsePossibleNumber(node,"esti_cost")));
            if(node.contains("children"))
                for(auto&c:node["children"]) stack.push_back(c);
        }
    }
    if(!rows.empty()){
        auto mid=[&](std::vector<double>&v){
            std::nth_element(v.begin(),v.begin()+v.size()/2,v.end());
            return v[v.size()/2];
        };
        auto iqr=[&](std::vector<double>&v){
            size_t n=v.size();
            std::nth_element(v.begin(),v.begin()+n/4,v.end());
            double q1=v[n/4];
            std::nth_element(v.begin()+n/4,v.begin()+3*n/4,v.end());
            double q3=v[3*n/4];
            return (q3-q1)/2.0;
        };
        st.rows_c=mid(rows);
        st.cost_c=mid(cost);
        st.rows_s=std::max(1e-6, iqr(rows));
        st.cost_s=std::max(1e-6, iqr(cost));
        int id=0;
        for(const auto&op:ops) st.op2id[op]=id++;
    }
    json dump{
        {"rows_c",st.rows_c},
        {"rows_s",st.rows_s},
        {"cost_c",st.cost_c},
        {"cost_s",st.cost_s},
        {"op2id",st.op2id}
    };
    std::ofstream(cache)<<dump.dump(2);
    return st;
}

/* ──────────────────── parseColPlan → Graph ──────────────────── */
struct Graph{
    int N=0;
    std::vector<Eigen::Vector2d> x;
    std::vector<std::vector<int>> adj;
    std::vector<int> op;
};
static Graph parseColPlan(const json&j,
                          const ColStats&st,
                          const std::unordered_map<std::string,int>&globalOp2id)
{
    Graph g;
    if(!j.contains("plan")||!j["plan"].is_array()) return g;
    std::vector<const json*> stack{&j["plan"][0]};
    std::vector<std::pair<int,int>> edges;
    while(!stack.empty()){
        auto cur=stack.back(); stack.pop_back();
        int idx=g.N++;
        double rawR=safeLog1p(parsePossibleNumber(*cur,"esti_rows"));
        double rawC=safeLog1p(parsePossibleNumber(*cur,"esti_cost"));
        double nr=(rawR-st.rows_c)/st.rows_s;
        double nc=(rawC-st.cost_c)/st.cost_s;
        g.x.emplace_back(nr,nc);
        std::string opName=cur->value("operator","UNK");
        int gid=globalOp2id.count(opName)?globalOp2id.at(opName):globalOp2id.at("UNK");
        g.op.push_back(gid);
        g.adj.emplace_back();
        if(cur->contains("children")&&cur->at("children").is_array()){
            for(auto&c:cur->at("children")){
                stack.push_back(&c);
                edges.emplace_back(idx,g.N);
                edges.emplace_back(g.N,idx);
            }
        }
    }
    for(auto&e:edges)
        if(e.first<g.N&&e.second<g.N)
            g.adj[e.first].push_back(e.second);
    return g;
}

/* ─────────────────── dataset loader ─────────────────── */
struct Sample{
    float feats[8];
    Graph colGraph;
    int   label;           // 0=row,1=col,-1=skip
    double row_time,col_time,query_cost;
    int hybrid_use_imci,fann_use_imci;
};

class ExecutionPlanDataset{
    std::vector<Sample> data_;
public:
    ExecutionPlanDataset(const std::vector<std::string>& dirs,double timeout){
        std::vector<ColStats> statsList;
        std::unordered_set<std::string> allOps;
        for(auto&d:dirs){
            std::string dir="/home/wuy/query_costs/"+d;
            auto cst=buildColStats(dir);
            statsList.push_back(cst);
            for(auto&kv:cst.op2id) allOps.insert(kv.first);
        }
        allOps.insert("UNK");
        std::unordered_map<std::string,int> globalOp2id;
        int oid=0; for(auto&op:allOps) globalOp2id[op]=oid++;
        size_t skip=0,val=0;
        for(size_t di=0;di<dirs.size();++di){
            auto&d=dirs[di];
            std::string dir="/home/wuy/query_costs/"+d;
            std::string csv=dir+"/query_costs.csv";
            if(!fileExists(csv)){ logWarn("Missing "+csv); continue; }
            size_t lines=0;
            { std::ifstream f(csv); std::string l; while(std::getline(f,l)) ++lines; }
            std::ifstream in(csv);
            std::string header; std::getline(in,header);
            size_t cur=0,tot=(lines?lines-1:0);
            const auto&cst=statsList[di];
            std::string rpdir=dir+"/row_plans", cpdir=dir+"/column_plans", line;
            while(std::getline(in,line)){
                ++cur; showProgressBar(cur,tot);
                if(line.empty()) continue;
                std::stringstream ss(line);
                std::string qid,lbl,rt,ct,hy,fy;
                std::getline(ss,qid,','); std::getline(ss,lbl,',');
                std::getline(ss,rt,',');  std::getline(ss,ct,',');
                std::getline(ss,hy,',');  std::getline(ss,fy,',');
                if(qid.empty()||lbl.empty()){ ++skip; continue;}
                Sample s{};
                s.label=std::stoi(lbl);
                s.row_time = rt.empty()?timeout:std::stod(rt);
                s.col_time = ct.empty()?timeout:std::stod(ct);
                if(std::fabs(s.row_time-s.col_time)<1e-3) s.label=-1;
                s.hybrid_use_imci = hy.empty()?0:std::stoi(hy);
                s.fann_use_imci   = fy.empty()?0:std::stoi(fy);
                if(!parseRowPlanJSON(rpdir+"/"+qid+".json",s.feats,s.query_cost)){
                    ++skip; continue;
                }
                if(fileExists(cpdir+"/"+qid+".json")){
                    std::ifstream cf(cpdir+"/"+qid+".json"); json cj; cf>>cj;
                    s.colGraph=parseColPlan(cj,cst,globalOp2id);
                }
                data_.push_back(s); ++val;
            }
        }
        logInfo("Loaded "+std::to_string(val)+" samples, skipped="+std::to_string(skip));
    }
    size_t size() const{ return data_.size(); }
    const Sample& operator[](size_t i) const{ return data_[i]; }
};

/* ──────────────────────── GIN encoder ──────────────────────── */
struct GIN {
    int in_dim, hidden, n_ops;
    MatrixXd W1, W2;
    MatrixXd b1, b2;
    MatrixXd opEmb;
    std::mt19937 rng;
    std::normal_distribution<> nd;

    // --- column‐time head ---
    RowVectorXd w_col;
    double       b_col;

    GIN(int n_ops_,int hidden_)
      : in_dim(2+8), hidden(hidden_), n_ops(n_ops_),
        W1(hidden,in_dim), b1(hidden,1),
        W2(hidden,hidden), b2(hidden,1),
        opEmb(8,n_ops_),
        rng(123), nd(0.1)
    {
        auto init=[&](MatrixXd&m){ for(int i=0;i<m.size();++i) m.data()[i]=nd(rng); };
        init(W1); init(b1);
        init(W2); init(b2);
        init(opEmb);
        // init col head
        w_col.resize(hidden);
        for(int i=0;i<hidden;++i) w_col(i)=nd(rng);
        b_col=0.0;
    }

    static VectorXd relu(const VectorXd&v){ return v.cwiseMax(0.0); }

    VectorXd encode(const Graph&g) const {
        if(g.N==0) return VectorXd::Zero(hidden);
        std::vector<VectorXd> z1(g.N), h(g.N);
        VectorXd out=VectorXd::Zero(hidden);
        for(int i=0;i<g.N;++i){
            VectorXd xin(in_dim);
            xin.head(2)=g.x[i];
            xin.tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*xin + b1.col(0));
        }
        for(int i=0;i<g.N;++i){
            VectorXd agg=VectorXd::Zero(hidden);
            for(int nb:g.adj[i]) agg+=z1[nb];
            h[i]=relu(W2*agg + b2.col(0));
            out+=h[i];
        }
        return out/double(g.N);
    }

    void backward_and_update(const Graph&g,
                             const VectorXd&d_out,
                             double lr)
    {
        if(g.N==0) return;
        // recompute
        std::vector<VectorXd> xin(g.N), z1(g.N), agg(g.N), pre2(g.N), h(g.N);
        for(int i=0;i<g.N;++i){
            xin[i].resize(in_dim);
            xin[i].head(2)=g.x[i];
            xin[i].tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*xin[i]+b1.col(0));
        }
        for(int i=0;i<g.N;++i){
            agg[i]=VectorXd::Zero(hidden);
            for(int nb:g.adj[i]) agg[i]+=z1[nb];
            pre2[i]=W2*agg[i]+b2.col(0);
            h[i]=relu(pre2[i]);
        }
        // grads
        MatrixXd gW2=MatrixXd::Zero(W2.rows(),W2.cols()),
                 gb2=MatrixXd::Zero(b2.rows(),b2.cols()),
                 gW1=MatrixXd::Zero(W1.rows(),W1.cols()),
                 gb1=MatrixXd::Zero(b1.rows(),b1.cols()),
                 gOp=MatrixXd::Zero(opEmb.rows(),opEmb.cols());
        VectorXd dh=d_out/double(g.N);
        for(int i=0;i<g.N;++i){
            VectorXd mask2=(pre2[i].array()>0.0).cast<double>().matrix();
            VectorXd d_pre2=dh.cwiseProduct(mask2);
            gW2+=d_pre2*agg[i].transpose();
            gb2.col(0)+=d_pre2;
            VectorXd d_agg=W2.transpose()*d_pre2;
            for(int nb:g.adj[i]){
                VectorXd mask1=((W1*xin[nb]+b1.col(0)).array()>0.0).cast<double>().matrix();
                VectorXd d_pre1=d_agg.cwiseProduct(mask1);
                gW1+=d_pre1*xin[nb].transpose();
                gb1.col(0)+=d_pre1;
                VectorXd dxin=W1.transpose()*d_pre1;
                gOp.col(g.op[nb])+=dxin.tail(8);
            }
        }
        // sgd
        W2   -=lr*gW2;  b2   -=lr*gb2;
        W1   -=lr*gW1;  b1   -=lr*gb1;
        opEmb-=lr*gOp;
    }

    // --- column‐time head forward/back/backprop ---
    double predict_col_time(const Graph&g) const {
        VectorXd h=encode(g);
        return (w_col*h)(0)+b_col;
    }
    void backward_col(double d_out,const Graph&g,double lr){
        VectorXd h=encode(g);
        RowVectorXd gw=d_out*h.transpose();
        double gb=d_out;
        w_col-=lr*gw;
        b_col-=lr*gb;
        // propagate into GIN body
        VectorXd grad_h = d_out*w_col.transpose();
        backward_and_update(g,grad_h,lr);
    }
};

/* ──────────────────── simple MLP regressor ──────────────────── */
class MLPReg {
public:
    // ctor: in = number of input features, hid = hidden layer size, lr = learning rate
    MLPReg(int in, int hid, double lr)
      : in_(in), hid_(hid), lr_(lr),
        W1(hid, in), b1(hid),
        W2(hid), b2(0.0)
    {
        // uniform random init in [-0.1, 0.1]
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        // W1
        for(int r = 0; r < W1.rows(); ++r)
            for(int c = 0; c < W1.cols(); ++c)
                W1(r,c) = dist(gen);
        // b1
        for(int i = 0; i < b1.size(); ++i)
            b1(i) = dist(gen);
        // W2
        for(int i = 0; i < W2.size(); ++i)
            W2(i) = dist(gen);
        // b2 remains 0
    }

    // train on a single example (online SGD)
    void train_sample(const std::vector<float>& xin, double y) {
        // forward
        Eigen::VectorXd x(in_);
        for(int i = 0; i < in_; ++i) x(i) = xin[i];

        Eigen::VectorXd z1 = W1 * x + b1;                   // (hid)
        Eigen::VectorXd a1 = z1.unaryExpr(&sigmoid);        // activation
        double       z2 = W2.dot(a1) + b2;                  // output
        double       err = z2 - y;                          // error

        // backward MSE loss: dL/dz2 = 2*(z2 - y)
        double delta2 = 2.0 * err;

        // gradients for W2, b2
        Eigen::RowVectorXd gradW2 = delta2 * a1.transpose();  // (1×hid)
        double             gradb2 = delta2;

        // backprop into hidden
        Eigen::VectorXd delta1 = (W2.transpose() * delta2)
            .cwiseProduct(a1.unaryExpr(&dsigmoid));          // (hid)

        // gradients for W1, b1
        Eigen::MatrixXd gradW1 = delta1 * x.transpose();     // (hid×in)
        Eigen::VectorXd gradb1 = delta1;

        // update
        W2 -= lr_ * gradW2;
        b2 -= lr_ * gradb2;
        W1 -= lr_ * gradW1;
        b1 -= lr_ * gradb1;
    }

    // inference: returns predicted scalar
    double run(const std::vector<float>& xin) const {
        Eigen::VectorXd x(in_);
        for(int i = 0; i < in_; ++i) x(i) = xin[i];
        Eigen::VectorXd a1 = (W1 * x + b1).unaryExpr(&sigmoid);
        return W2.dot(a1) + b2;
    }

    // serialize to JSON
    json to_json() const {
        json j;
        j["in"]  = in_;
        j["hid"] = hid_;
        j["W1"]  = std::vector<double>(W1.data(),  W1.data()  + W1.size());
        j["b1"]  = std::vector<double>(b1.data(),  b1.data()  + b1.size());
        j["W2"]  = std::vector<double>(W2.data(),  W2.data()  + W2.size());
        j["b2"]  = b2;
        return j;
    }

    // deserialize from JSON
    void from_json(const json& j) {
        int in2  = j.at("in").get<int>();
        int hid2 = j.at("hid").get<int>();
        if (in2 != in_ || hid2 != hid_)
            throw std::runtime_error("Shape mismatch in MLPReg::from_json");

        auto w1 = j.at("W1").get<std::vector<double>>();
        std::copy(w1.begin(), w1.end(), W1.data());

        auto bb1 = j.at("b1").get<std::vector<double>>();
        std::copy(bb1.begin(), bb1.end(), b1.data());

        auto w2 = j.at("W2").get<std::vector<double>>();
        std::copy(w2.begin(), w2.end(), W2.data());

        b2 = j.at("b2").get<double>();
    }

private:
    int              in_, hid_;
    double           lr_;
    Eigen::MatrixXd  W1;  // (hid × in)
    Eigen::VectorXd  b1;  // (hid)
    Eigen::RowVectorXd W2; // (1 × hid)
    double           b2;  // scalar

    // sigmoid and its derivative
    static double sigmoid(double z)  { return 1.0 / (1.0 + std::exp(-z)); }
    static double dsigmoid(double a) { return a * (1.0 - a); }
};

/* ──────────────────── dataset split ──────────────────── */
std::pair<std::vector<size_t>,std::vector<size_t>>
splitDataset(size_t n,double ratio=0.8){
    std::vector<size_t> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::shuffle(idx.begin(),idx.end(),std::mt19937(std::random_device{}()));
    size_t k= size_t(ratio*n);
    return {{idx.begin(),idx.begin()+k},{idx.begin()+k,idx.end()}};
}

int main(int argc, char* argv[]) {
    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();

    // ─────── CLI defaults ───────
    std::vector<std::string> dataDirs;
    int    epochs        = 1000;
    int    hiddenNeurons = 128;
    double lr            = 0.001;
    std::string bestModel = "checkpoints/row_column_regression.json";
    bool   skipTrain     = false;
    double costThreshold = 50000.0;

    // ─────── Parse args ───────
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a.rfind("--data_dirs=",0)==0)       dataDirs.push_back(a.substr(12));
        else if (a.rfind("--epochs=",0)==0)          epochs        = std::stoi(a.substr(9));
        else if (a.rfind("--hidden_neurons=",0)==0)  hiddenNeurons = std::stoi(a.substr(17));
        else if (a.rfind("--lr=",0)==0)              lr            = std::stod(a.substr(5));
        else if (a.rfind("--best_model_path=",0)==0) bestModel     = a.substr(18);
        else if (a == "--skip_train")                             skipTrain = true;
    }
    if (dataDirs.empty()) {
        logError("need --data_dirs=...");
        return 1;
    }

    // ─────── Load dataset ───────
    ExecutionPlanDataset dataset(dataDirs, /*timeout*/ 60.0);
    if (dataset.size() < 30) {
        logError("Too few samples");
        return 1;
    }

    // ─────── Split 80/20 ───────
    auto [trainIdx, valIdx] = splitDataset(dataset.size(), 0.8);

    // ─────── Build models ───────
    // figure out number of operators
    int maxOp = 0;
    for (size_t i = 0; i < dataset.size(); ++i)
        for (int op : dataset[i].colGraph.op)
            maxOp = std::max(maxOp, op);
    GIN gin(maxOp+1, hiddenNeurons);
    MLPReg rowReg(8, hiddenNeurons, lr);

    // ─────── Train or load ───────
    if (skipTrain) {
        std::ifstream f(bestModel);
        if (!f) { logError("Cannot open " + bestModel); return 1; }
        json j; f >> j;
        rowReg.from_json(j["rowReg"]);
        // TODO: load gin weights & head
        logInfo("Loaded rowReg from " + bestModel);
    } else {
        double bestValMSE = 1e30;
        int    patience    = 10, wait = 0;
        for (int ep = 1; ep <= epochs; ++ep) {
            printf("epoch: %d", epoch);
            std::shuffle(trainIdx.begin(), trainIdx.end(),
                         std::mt19937(std::random_device{}()));
            double trainMSE = 0.0;
            size_t Ntrain = trainIdx.size();
            for (size_t k = 0; k < Ntrain; ++k) {
                showProgressBar(k+1, Ntrain);
                auto &s = dataset[trainIdx[k]];

                // — train row‐time regressor
                std::vector<float> x8(s.feats, s.feats+8);
                rowReg.train_sample(x8, s.row_time);
                double pr = rowReg.run(x8);
                trainMSE += (pr - s.row_time)*(pr - s.row_time);

                // — train column‐time regressor
                double pc = gin.predict_col_time(s.colGraph);
                gin.backward_col(2*(pc - s.col_time), s.colGraph, lr);
            }
            trainMSE /= double(Ntrain);

            // validation MSE (row‐time only)
            double valMSE = 0.0; int cnt = 0;
            for (auto i : valIdx) {
                auto &s = dataset[i];
                std::vector<float> x8(s.feats, s.feats+8);
                double pr = rowReg.run(x8);
                valMSE += (pr - s.row_time)*(pr - s.row_time);
                ++cnt;
            }
            if (cnt) valMSE /= cnt;

            logInfo("Ep " + std::to_string(ep)
                  + " trainMSE=" + std::to_string(trainMSE)
                  + " valMSE="   + std::to_string(valMSE)
                  + " wait="     + std::to_string(wait));

            if (valMSE < bestValMSE) {
                bestValMSE = valMSE;
                wait = 0;
                // save both models
                json out;
                out["rowReg"] = rowReg.to_json();
                // TODO: serialize gin
                std::ofstream of(bestModel);
                of << out.dump(2);
                logInfo("→ saved new best model");
            } else if (++wait >= patience) {
                logInfo("Early stopping at ep " + std::to_string(ep));
                break;
            }
        }
    }

    // ─────── Evaluate ───────
    // average runtimes
    double rt_row_only = 0, rt_col_only = 0,
           rt_cost = 0, rt_hyb = 0, rt_fann = 0;
    size_t nr=0, nc=0, ncost=0, nh=0, nf=0;

    // Q-error collections
    std::vector<double> qerrs_row, qerrs_col;

    // helper for running average
    auto online_avg = [&](double &acc, double v, size_t &n){
        acc = (acc*double(n) + v) / double(n+1);
        ++n;
    };

    for (size_t i = 0; i < dataset.size(); ++i) {
        auto &s = dataset[i];
        if (s.label < 0) continue;

        // true times
        online_avg(rt_row_only, s.row_time, nr);
        online_avg(rt_col_only, s.col_time, nc);
        online_avg(rt_cost,
                   ((s.query_cost > costThreshold) ? s.col_time : s.row_time),
                   ncost);
        online_avg(rt_hyb,
                   (s.hybrid_use_imci ? s.col_time : s.row_time),
                   nh);
        online_avg(rt_fann,
                   (s.fann_use_imci ? s.col_time : s.row_time),
                   nf);

        // predictions
        std::vector<float> x8(s.feats, s.feats+8);
        double pred_r = rowReg.run(x8);
        double pred_c = gin.predict_col_time(s.colGraph);

        // Q-errors
        if (pred_r > 0 && s.row_time > 0) {
            double q = pred_r / s.row_time;
            if (q < 1.0) q = 1.0 / q;
            qerrs_row.push_back(q);
        }
        if (pred_c > 0 && s.col_time > 0) {
            double q = pred_c / s.col_time;
            if (q < 1.0) q = 1.0 / q;
            qerrs_col.push_back(q);
        }
    }

    // ─────── Q-error statistics ───────
    auto compute_stats = [&](std::vector<double>& v){
        std::sort(v.begin(), v.end());
        size_t N = v.size();
        double mn = v.front();
        double mx = v.back();
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / N;
        double med  = v[N/2];
        double p95  = v[ std::min(N-1, size_t(0.95*N)) ];
        return std::make_tuple(mn, mean, med, p95, mx);
    };

    auto [r_min, r_mean, r_med, r_p95, r_max] = compute_stats(qerrs_row);
    auto [c_min, c_mean, c_med, c_p95, c_max] = compute_stats(qerrs_col);

    std::cout << "\n=== Q-ERROR ROW-TIME REGRESSOR ===\n"
              << "min     : " << r_min  << "\n"
              << "mean    : " << r_mean << "\n"
              << "median  : " << r_med  << "\n"
              << "95th %  : " << r_p95  << "\n"
              << "max     : " << r_max  << "\n";

    std::cout << "\n=== Q-ERROR COL-TIME REGRESSOR ===\n"
              << "min     : " << c_min  << "\n"
              << "mean    : " << c_mean << "\n"
              << "median  : " << c_med  << "\n"
              << "95th %  : " << c_p95  << "\n"
              << "max     : " << c_max  << "\n";

    // ─────── Average runtimes ───────
    std::cout << "\n=== AVG RUNTIME (s/query) ===\n"
              << "Row only         : " << rt_row_only << "\n"
              << "Column only      : " << rt_col_only << "\n"
              << "Cost threshold   : " << rt_cost     << "\n"
              << "Hybrid optimizer : " << rt_hyb      << "\n"
              << "FANN-based       : " << rt_fann     << "\n";

    double wall = std::chrono::duration_cast<std::chrono::seconds>(
                      Clock::now() - t_start
                  ).count();
    logInfo("Total wall-clock: " + std::to_string(wall) + " s");

    return 0;
}

