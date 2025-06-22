/************************************************************************************
 *
 * Row/Column Time Dual-Regression Router  (v3 – Row Tree-CNN · Col GIN, C++11)
 *
 * ▸ Row regressor : Tree-CNN encoder → hidden → 1  (log1p(time) target)
 * ▸ Col regressor : GIN encoder     → hidden → 1  (log1p(time) target)
 * ▸ Early stopping: monitor (valRowMSE + valColMSE)
 * ▸ Thread safety : single-thread train loop (no OpenMP)
 * ▸ Runtime choose: pick engine with lower   expm1(predicted_log_time)
 *
 * Compile:
 *   g++ -std=c++11 -O2 -I/path/to/eigen3 -I/path/to/json/single_include \
 *       -o row_treecnn_col_gin row_treecnn_col_gin.cpp
 *
 ************************************************************************************/

#include <bits/stdc++.h>
#include <dirent.h>
#include <Eigen/Dense>
#include "json.hpp"

using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/*──────────────────── logging / helpers ───────────────────*/
static void logInfo (const std::string &m){ std::cout  << "[INFO]  " << m << '\n'; }
static void logWarn (const std::string &m){ std::cerr  << "[WARN]  " << m << '\n'; }
static void logError(const std::string &m){ std::cerr  << "[ERROR] " << m << '\n'; }
inline bool   exists(const std::string&p){ std::ifstream f(p); return f.good(); }
inline double safeLog1p(double v){ return std::log1p(std::max(0.0,v)); }
inline double dhuber(double e,double d=1.0){
    double a=std::abs(e); return (a<=d)?e:d*((e>0)?1.0:-1.0);}
inline double qerr(double p,double t){
    const double eps=1e-6; if(p<eps||t<eps) return 1e9;
    double q=p/t; return (q<1.0)?1.0/q:q;}

/*──────────────────── progress bar ───────────────────────*/
void bar(size_t cur,size_t tot,size_t w=50){
    double f=tot?double(cur)/tot:1.0; size_t k=f*w;
    std::cout<<"\r["; for(size_t i=0;i<k;++i)std::cout<<'=';
    for(size_t i=k;i<w;++i)std::cout<<' ';
    std::cout<<"] "<<int(f*100)<<"% ("<<cur<<"/"<<tot<<")";
    if(cur>=tot) std::cout<<'\n'; std::fflush(stdout);
}

/*──────────────────── operator dictionary ─────────────────*/
static std::unordered_map<std::string,int> g_op2id{{"UNK",0}};
static int op_id(const std::string&op){
    std::unordered_map<std::string,int>::iterator it=g_op2id.find(op);
    if(it!=g_op2id.end()) return it->second;
    int id=g_op2id.size(); g_op2id[op]=id; return id;
}

/*──────── safe numeric extraction (number OR numeric string) ────────*/
static double num(const json& o, const char* k)
{
    if(!o.contains(k)) return 0.0;
    const auto& v = o[k];
    if(v.is_number())  return v.get<double>();
    if(v.is_string()){ try{ return std::stod(v.get<std::string>()); }catch(...){ } }
    return 0.0;
}

/*──────────────────── graph container ───────────────────*/
struct Graph{
    int N;
    std::vector<VectorXd>          x;    // node features
    std::vector<std::vector<int> > adj;  // undirected
    std::vector<int>               op;   // operator id
    Graph():N(0){}
};

/*════════════════════ PARSERS ════════════════════*/
/* column stats for normalisation */
struct ColStats{ double rc,rs,cc,cs; ColStats():rc(0),rs(1),cc(0),cs(1){} };

/* --- parse column plan → Graph ------------------ */
static void colRec(const json& node,int parent,Graph& g,const ColStats& st)
{
    int idx = g.N++;
    double r = safeLog1p( num(node,"esti_rows") );
    double c = safeLog1p( num(node,"esti_cost") );

    VectorXd v(2);  v << (r - st.rc) / st.rs ,
                      (c - st.cc) / st.cs;

    g.x .push_back(v);
    g.op.push_back( op_id(node.value("operator","UNK")) );
    g.adj.push_back({});

    if(parent >= 0){
        g.adj[parent].push_back(idx);
        g.adj[idx]   .push_back(parent);
    }
    if(node.contains("children"))
        for(const auto& ch : node["children"])
            colRec(ch, idx, g, st);
}
static Graph parseColPlan(const json&j,const ColStats&st){
    Graph g; if(!j.contains("plan")||!j["plan"].is_array()) return g;
    colRec(j["plan"][0],-1,g,st); return g;
}

/* --- parse row plan → Graph (5 numeric + op-emb) --- */
static void rowRec(const json& node,int parent,Graph& g)
{
    int idx = g.N++;
    double re = 0, rp = 0, rd = 0, ev = 0, pc = 0;
    std::string op = "node";

    if(node.contains("table"))
    {
        const json& t  = node["table"];
        op = t.value("access_type","table");
        re = num(t,"rows_examined_per_scan");
        rp = num(t,"rows_produced_per_join");

        if(t.contains("cost_info"))
        {
            const json& ci = t["cost_info"];
            rd = num(ci,"read_cost");
            ev = num(ci,"eval_cost");
            pc = num(ci,"prefix_cost");
        }
    }
    else if(node.contains("op_type"))
        op = node.value("op_type","node");

    VectorXd v(5);
    v << safeLog1p(re), safeLog1p(rp), safeLog1p(rd),
         safeLog1p(ev), safeLog1p(pc);

    g.x .push_back(v);
    g.op.push_back( op_id(op) );
    g.adj.push_back({});

    if(parent >= 0){
        g.adj[parent].push_back(idx);
        g.adj[idx]   .push_back(parent);
    }
    const char* ks[] = {"grouping_operation","ordering_operation"};
    for(auto k : ks) if(node.contains(k)) rowRec(node[k], idx, g);
    if(node.contains("nested_loop"))
        for(const auto& ch : node["nested_loop"])
            rowRec(ch, idx, g);
}

static Graph parseRowPlan(const json&plan){
    Graph g; if(!plan.contains("query_block")) return g;
    rowRec(plan["query_block"],-1,g); return g;
}

/*════════════════════ MODELS ════════════════════*/
/* --- Tree-CNN encoder for row plans ----------- */
struct TreeCNN{
    int in_dim,hidden,n_ops,emb;
    MatrixXd W1,W2,opEmb; VectorXd b1,b2; RowVectorXd w; double b;
    std::mt19937 rng; std::uniform_real_distribution<double> ud;
    TreeCNN(int n_ops_,int hid)
      : in_dim(5+8),hidden(hid),n_ops(n_ops_),emb(8),
        W1(hid,in_dim),W2(hid,hid),opEmb(emb,n_ops_),
        b1(hid),b2(hid),w(hid),b(0),rng(123),ud(-0.08,0.08)
    {
        for(int i=0;i<W1.size();++i) W1.data()[i]=ud(rng);
        for(int i=0;i<W2.size();++i) W2.data()[i]=ud(rng);
        for(int i=0;i<opEmb.size();++i) opEmb.data()[i]=ud(rng);
        for(int i=0;i<hidden;++i){ b1(i)=ud(rng); b2(i)=ud(rng); w(i)=ud(rng); }
    }
    static VectorXd relu(const VectorXd&v){ return v.cwiseMax(0.0); }
    VectorXd encode(const Graph&g)const{
        if(g.N==0) return VectorXd::Zero(hidden);
        std::vector<VectorXd> z1(g.N);
        for(int i=0;i<g.N;++i){
            VectorXd x(in_dim); x.head(5)=g.x[i]; x.tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*x + b1);
        }
        std::vector<VectorXd> h(g.N);
        for(int i=0;i<g.N;++i){
            VectorXd agg=VectorXd::Zero(hidden);
            for(size_t k=0;k<g.adj[i].size();++k) agg+=z1[g.adj[i][k]];
            h[i]=relu(W2*agg + b2);
        }
        VectorXd out=VectorXd::Zero(hidden); for(int i=0;i<g.N;++i) out+=h[i];
        return out/double(g.N);
    }
    double predict(const Graph&g)const{ return w*encode(g) + b; }

    void backward(double d,const Graph&g,double lr){
        if(g.N==0) return;
        /* forward caches */
        std::vector<VectorXd> xin(g.N),z1(g.N),agg(g.N),pre2(g.N),h(g.N);
        for(int i=0;i<g.N;++i){
            xin[i].resize(in_dim); xin[i].head(5)=g.x[i]; xin[i].tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*xin[i]+b1);
        }
        for(int i=0;i<g.N;++i){
            agg[i]=VectorXd::Zero(hidden);
            for(size_t k=0;k<g.adj[i].size();++k) agg[i]+=z1[g.adj[i][k]];
            pre2[i]=W2*agg[i]+b2; h[i]=relu(pre2[i]);
        }
        RowVectorXd gw=d*encode(g).transpose(); double gb=d; w-=lr*gw; b-=lr*gb;
        VectorXd g_out=(d*w.transpose())/double(g.N);
        std::vector<VectorXd> dh(g.N,g_out);

        MatrixXd gW2=MatrixXd::Zero(W2.rows(),W2.cols()),
                 gW1=MatrixXd::Zero(W1.rows(),W1.cols()),
                 gOp=MatrixXd::Zero(opEmb.rows(),opEmb.cols());
        VectorXd gb2=VectorXd::Zero(b2.size()), gb1=VectorXd::Zero(b1.size());

        for(int i=0;i<g.N;++i){
            VectorXd mask2=((pre2[i].array()>0).cast<double>());
            VectorXd dp2=dh[i].cwiseProduct(mask2);
            gW2+=dp2*agg[i].transpose(); gb2+=dp2;
            VectorXd dagg=W2.transpose()*dp2;
            for(size_t k=0;k<g.adj[i].size();++k){
                int nb=g.adj[i][k];
                VectorXd mask1=((W1*xin[nb]+b1).array()>0).cast<double>();
                VectorXd dp1=dagg.cwiseProduct(mask1);
                gW1+=dp1*xin[nb].transpose(); gb1+=dp1;
                VectorXd dx=W1.transpose()*dp1;
                gOp.col(g.op[nb])+=dx.tail(8);
            }
        }
        W2-=lr*gW2; b2-=lr*gb2; W1-=lr*gW1; b1-=lr*gb1; opEmb-=lr*gOp;
    }

    json to_json()const{
        return json{
          {"hidden",hidden},{"n_ops",n_ops},
          {"W1",std::vector<double>(W1.data(),W1.data()+W1.size())},
          {"W2",std::vector<double>(W2.data(),W2.data()+W2.size())},
          {"b1",std::vector<double>(b1.data(),b1.data()+b1.size())},
          {"b2",std::vector<double>(b2.data(),b2.data()+b2.size())},
          {"opEmb",std::vector<double>(opEmb.data(),opEmb.data()+opEmb.size())},
          {"w",std::vector<double>(w.data(),w.data()+w.size())},
          {"b",b}
        };
    }
    void from_json(const json&j){
        hidden=j.at("hidden").get<int>();
        int newOps=j.at("n_ops").get<int>();
        if(newOps!=n_ops){ n_ops=newOps; opEmb.resize(8,n_ops);}
        auto copy=[&](const std::vector<double>&v,double*dst,int sz){
            std::copy(v.begin(),v.begin()+std::min(sz,(int)v.size()),dst); };
        copy(j.at("W1").get<std::vector<double>>(),W1.data(),W1.size());
        copy(j.at("W2").get<std::vector<double>>(),W2.data(),W2.size());
        copy(j.at("b1").get<std::vector<double>>(),b1.data(),b1.size());
        copy(j.at("b2").get<std::vector<double>>(),b2.data(),b2.size());
        copy(j.at("opEmb").get<std::vector<double>>(),opEmb.data(),opEmb.size());
        copy(j.at("w").get<std::vector<double>>(),w.data(),w.size());
        b=j.at("b").get<double>();
    }
};

/* --- GIN for column plans (unchanged except mask fix) -------------------- */
struct GIN{
    int in_dim,hidden,n_ops,emb;
    MatrixXd W1,W2,opEmb; VectorXd b1,b2; RowVectorXd w; double b;
    std::mt19937 rng; std::uniform_real_distribution<double> ud;
    GIN(int n_ops_,int hid)
      : in_dim(2+8),hidden(hid),n_ops(n_ops_),emb(8),
        W1(hid,in_dim),W2(hid,hid),opEmb(emb,n_ops_),
        b1(hid),b2(hid),w(hid),b(0),rng(321),ud(-0.08,0.08)
    {
        for(int i=0;i<W1.size();++i) W1.data()[i]=ud(rng);
        for(int i=0;i<W2.size();++i) W2.data()[i]=ud(rng);
        for(int i=0;i<opEmb.size();++i) opEmb.data()[i]=ud(rng);
        for(int i=0;i<hidden;++i){ b1(i)=ud(rng); b2(i)=ud(rng); w(i)=ud(rng); }
    }
    static VectorXd relu(const VectorXd&v){ return v.cwiseMax(0.0); }
    VectorXd encode(const Graph&g)const{
        if(g.N==0) return VectorXd::Zero(hidden);
        std::vector<VectorXd> z1(g.N);
        for(int i=0;i<g.N;++i){
            VectorXd x(in_dim); x.head(2)=g.x[i]; x.tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*x+b1);
        }
        std::vector<VectorXd> h(g.N);
        for(int i=0;i<g.N;++i){
            VectorXd agg=VectorXd::Zero(hidden);
            for(size_t k=0;k<g.adj[i].size();++k) agg+=z1[g.adj[i][k]];
            h[i]=relu(W2*agg+b2);
        }
        VectorXd out=VectorXd::Zero(hidden);
        for(int i=0;i<g.N;++i) out+=h[i];
        return out/double(g.N);
    }
    double predict(const Graph&g)const{ return w*encode(g)+b; }
    void backward(double d,const Graph&g,double lr){
        if(g.N==0) return;
        std::vector<VectorXd> xin(g.N),z1(g.N),agg(g.N),pre2(g.N),h(g.N);
        for(int i=0;i<g.N;++i){
            xin[i].resize(in_dim); xin[i].head(2)=g.x[i]; xin[i].tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*xin[i]+b1);
        }
        for(int i=0;i<g.N;++i){
            agg[i]=VectorXd::Zero(hidden);
            for(size_t k=0;k<g.adj[i].size();++k) agg[i]+=z1[g.adj[i][k]];
            pre2[i]=W2*agg[i]+b2; h[i]=relu(pre2[i]);
        }
        RowVectorXd gw=d*encode(g).transpose(); double gb=d; w-=lr*gw; b-=lr*gb;
        VectorXd g_out=(d*w.transpose())/double(g.N);
        std::vector<VectorXd> dh(g.N,g_out);

        MatrixXd gW2=MatrixXd::Zero(W2.rows(),W2.cols()),
                 gW1=MatrixXd::Zero(W1.rows(),W1.cols()),
                 gOp=MatrixXd::Zero(opEmb.rows(),opEmb.cols());
        VectorXd gb2=VectorXd::Zero(b2.size()), gb1=VectorXd::Zero(b1.size());
        for(int i=0;i<g.N;++i){
            VectorXd mask2=((pre2[i].array()>0).cast<double>());
            VectorXd dp2=dh[i].cwiseProduct(mask2);
            gW2+=dp2*agg[i].transpose(); gb2+=dp2;
            VectorXd dagg=W2.transpose()*dp2;
            for(size_t k=0;k<g.adj[i].size();++k){
                int nb=g.adj[i][k];
                VectorXd mask1=((W1*xin[nb]+b1).array()>0).cast<double>();
                VectorXd dp1=dagg.cwiseProduct(mask1);
                gW1+=dp1*xin[nb].transpose(); gb1+=dp1;
                VectorXd dx=W1.transpose()*dp1;
                gOp.col(g.op[nb])+=dx.tail(8);
            }
        }
        W2-=lr*gW2; b2-=lr*gb2; W1-=lr*gW1; b1-=lr*gb1; opEmb-=lr*gOp;
    }
    json to_json()const{
        return json{
          {"hidden",hidden},{"n_ops",n_ops},
          {"W1",std::vector<double>(W1.data(),W1.data()+W1.size())},
          {"W2",std::vector<double>(W2.data(),W2.data()+W2.size())},
          {"b1",std::vector<double>(b1.data(),b1.data()+b1.size())},
          {"b2",std::vector<double>(b2.data(),b2.data()+b2.size())},
          {"opEmb",std::vector<double>(opEmb.data(),opEmb.data()+opEmb.size())},
          {"w",std::vector<double>(w.data(),w.data()+w.size())},
          {"b",b}
        };
    }
    void from_json(const json&j){
        hidden=j.at("hidden").get<int>();
        int newOps=j.at("n_ops").get<int>();
        if(newOps!=n_ops){ n_ops=newOps; opEmb.resize(8,n_ops);}
        auto copy=[&](const std::vector<double>&v,double*dst,int sz){
            std::copy(v.begin(),v.begin()+std::min(sz,(int)v.size()),dst); };
        copy(j.at("W1").get<std::vector<double>>(),W1.data(),W1.size());
        copy(j.at("W2").get<std::vector<double>>(),W2.data(),W2.size());
        copy(j.at("b1").get<std::vector<double>>(),b1.data(),b1.size());
        copy(j.at("b2").get<std::vector<double>>(),b2.data(),b2.size());
        copy(j.at("opEmb").get<std::vector<double>>(),opEmb.data(),opEmb.size());
        copy(j.at("w").get<std::vector<double>>(),w.data(),w.size());
        b=j.at("b").get<double>();
    }
};



struct Sample
{
    Graph  rowG, colG;
    double rt = 0.0, ct = 0.0;          // true runtimes
    double query_cost = 0.0;            // optimiser cost (row plan)
    bool   hybrid_use_imci = false;     // trace flags
    bool   fann_use_imci   = false;
};

class Dataset
{
    std::vector<Sample> data_;
    ColStats            st_;            // column-plan normalisation

public:
    /* -------- constructor loads & parses everything -------- */
    explicit Dataset(const std::vector<std::string>& dirs)
    {
        /* 1) scan column plans to collect stats (progress bar inside) */
        std::vector<double> rowsRaw , costRaw;

        for(const auto& d : dirs)
        {
            std::string dir = "/home/wuy/query_costs/" + d + "/column_plans";
            DIR* dp = opendir(dir.c_str()); if(!dp) continue;

            size_t tot = 0; for(dirent* de; (de=readdir(dp)); )
                if(std::string(de->d_name).size() > 5) ++tot;
            rewinddir(dp);

            size_t cur = 0;
            for(dirent* de; (de=readdir(dp)); )
            {
                std::string fn(de->d_name);
                if(fn.size()<6 || fn.substr(fn.size()-5)!=".json") continue;

                json j; std::ifstream(dir+'/'+fn) >> j;
                if(!j.contains("plan") || !j["plan"].is_array()) continue;
                const json& n = j["plan"][0];

                rowsRaw .push_back( safeLog1p(num(n,"esti_rows")) );
                costRaw .push_back( safeLog1p(num(n,"esti_cost")) );
                bar(++cur, tot, 35);
            }
            if(tot) bar(tot, tot, 35);
            closedir(dp);
        }

        auto robustStat = [](std::vector<double>& v,double& c,double& s){
            if(v.empty()){ c=0; s=1; return; }
            size_t n=v.size(), mid=n/2, q1=n/4, q3=3*n/4;
            std::nth_element(v.begin(),v.begin()+mid,v.end()); c=v[mid];
            std::nth_element(v.begin(),v.begin()+q1 ,v.end()); double lo=v[q1];
            std::nth_element(v.begin(),v.begin()+q3 ,v.end()); double hi=v[q3];
            s = std::max(1e-6,(hi-lo)/2.0);
        };
        robustStat(rowsRaw, st_.rc, st_.rs);
        robustStat(costRaw, st_.cc, st_.cs);

        /* 2) read every CSV line + the two corresponding plans */
        size_t skipped = 0;
        for(const auto& d : dirs)
        {
            std::string base = "/home/wuy/query_costs/" + d;
            std::ifstream csv(base+"/query_costs.csv");
            if(!csv){ logWarn("missing CSV in "+d); continue; }

            std::vector<std::string> lines;
            { std::string tmp; std::getline(csv,tmp);          // header
              while(std::getline(csv,tmp)) lines.push_back(std::move(tmp)); }

            size_t tot = lines.size(), cur = 0;
            for(const auto& l : lines)
            {
                bar(++cur, tot, 35);

                std::stringstream ss(l);
                std::string qid,lab,rt,ct,hy,fy;
                std::getline(ss,qid,','); std::getline(ss,lab,',');
                std::getline(ss,rt,',');  std::getline(ss,ct,',');
                std::getline(ss,hy,',');  std::getline(ss,fy,',');

                double rtv = std::strtod(rt.c_str(),nullptr);
                double ctv = std::strtod(ct.c_str(),nullptr);
                if(!std::isfinite(rtv)||!std::isfinite(ctv)){ ++skipped; continue; }

                std::ifstream rp(base+"/row_plans/"+qid+".json"),
                              cp(base+"/column_plans/"+qid+".json");
                if(!rp||!cp){ ++skipped; continue; }

                json jr,jc; rp>>jr; cp>>jc;

                Sample s;
                s.rt = rtv;  s.ct = ctv;
                s.hybrid_use_imci = (hy=="1");
                s.fann_use_imci   = (fy=="1");
                s.rowG = parseRowPlan(jr);
                s.colG = parseColPlan(jc, st_);
                if(jr.contains("query_block") &&
                   jr["query_block"].contains("cost_info"))
                    s.query_cost = num(jr["query_block"]["cost_info"],"query_cost");

                data_.push_back(std::move(s));
            }
            if(tot) bar(tot, tot, 35);
        }
        std::ostringstream oss;
        oss<<"Dataset ready: "<<data_.size()<<" samples  (skipped "<<skipped<<')';
        logInfo(oss.str());
    }

    /* accessors + range-for helpers */
    size_t        size() const              { return data_.size(); }
    const Sample& operator[](size_t i)const { return data_[i];     }

    auto begin()       -> decltype(data_.begin()) { return data_.begin(); }
    auto end()         -> decltype(data_.end())   { return data_.end();   }
    auto begin() const -> decltype(data_.cbegin()){ return data_.cbegin();}
    auto end()   const -> decltype(data_.cend())  { return data_.cend();  }
};

/*════════════════════ SPLIT ════════════════════*/
static std::pair< std::vector<size_t>, std::vector<size_t> >
splitIds(size_t n,double r){
    std::vector<size_t> id(n); std::iota(id.begin(),id.end(),0);
    std::shuffle(id.begin(),id.end(),std::mt19937(20250621));
    size_t k=size_t(n*r);
    return std::make_pair(
        std::vector<size_t>(id.begin(),id.begin()+k),
        std::vector<size_t>(id.begin()+k,id.end()));
}

/*═══════════════════  MAIN  ═══════════════════*/
int main(int argc,char* argv[])
{
    /* ------------ CLI --------------- */
    std::vector<std::string> dirs;
    int    epochs   = 400;       // lower default → faster
    int    hidden   = 128;
    int    epoch_samples = 4000; // NEW: subsample per epoch
    bool   skip     = false;
    double costThr  = 5e4;

    for(int i=1;i<argc;++i){
        std::string a(argv[i]);
        if(a.rfind("--data_dirs=",0)==0){
            std::stringstream ss(a.substr(12)); std::string d;
            while(std::getline(ss,d,',')) dirs.push_back(d);
        }else if(a=="--skip_train")              skip   = true;
        else if(a.rfind("--epochs=",0)==0)       epochs = std::stoi(a.substr(9));
        else if(a.rfind("--hidden=",0)==0)       hidden = std::stoi(a.substr(9));
        else if(a.rfind("--epoch_samples=",0)==0)epoch_samples=std::stoi(a.substr(16));
        else if(a.rfind("--cost_threshold=",0)==0) costThr=std::stod(a.substr(17));
    }
    if(dirs.empty()){ logError("need --data_dirs=..."); return 1; }

    /* ------------ load data ------------ */
    Dataset ds(dirs);
    if(ds.size()<30){ logError("too few samples"); return 1; }

    /* ------------ models --------------- */
    int nOps = g_op2id.size();
    TreeCNN rowNet(nOps, hidden);
    GIN     colNet(nOps, hidden);

    if(!skip && exists("row_treecnn_model.json") && exists("col_gin_model.json")){
        std::ifstream fr("row_treecnn_model.json"); json jr; fr>>jr; rowNet.from_json(jr);
        std::ifstream fc("col_gin_model.json");    json jc; fc>>jc; colNet.from_json(jc);
        skip=true; logInfo("models loaded (skip-train)");
    }

    /* ------------ training ------------- */
    if(!skip)
    {
        auto split = splitIds(ds.size(), 0.8);
        std::vector<size_t>& tr = split.first;
        std::vector<size_t>& va = split.second;

        int patience=10, wait=0; double best=1e30;
        std::mt19937 rng(42);

        for(int ep=1; ep<=epochs; ++ep)
        {
            /* --- random tiny subset each epoch for speed --- */
            std::vector<size_t> batch(tr);
            std::shuffle(batch.begin(),batch.end(),rng);
            if(batch.size() > (size_t)epoch_samples) batch.resize(epoch_samples);

            double trR=0, trC=0; size_t cur=0, tot=batch.size();
            for(size_t idx : batch)
            {
                const Sample& s = ds[idx];

                /* row model */
                double pr = rowNet.predict(s.rowG);
                rowNet.backward(2.0*(pr-std::log1p(s.rt)), s.rowG, 1e-3);
                trR += (pr-std::log1p(s.rt))*(pr-std::log1p(s.rt));

                /* column model */
                double pc = colNet.predict(s.colG);
                colNet.backward(dhuber(pc-std::log1p(s.ct)), s.colG, 5e-4);
                trC += (pc-std::log1p(s.ct))*(pc-std::log1p(s.ct));

                if(++cur % 512 == 0 || cur == tot) bar(cur, tot, 35);
            }
            bar(tot, tot, 35);
            trR /= batch.size(); trC /= batch.size();

            /* --- validation (full) --- */
            double vR=0, vC=0;
            for(size_t idx: va){
                const Sample& s=ds[idx];
                double dr=rowNet.predict(s.rowG)-std::log1p(s.rt);
                double dc=colNet.predict(s.colG)-std::log1p(s.ct);
                vR+=dr*dr; vC+=dc*dc;
            }
            vR/=va.size(); vC/=va.size(); double vSum=vR+vC;

            std::ostringstream msg;
            msg<<"\n[ep "<<ep<<'/'<<epochs<<"] trR="<<trR<<" trC="<<trC
               <<"  vR="<<vR<<" vC="<<vC<<"  vSum="<<vSum;
            logInfo(msg.str());

            if(vSum<best){
                best=vSum; wait=0;
                std::ofstream("row_treecnn_model.json")
                       <<rowNet.to_json().dump(2);
                std::ofstream("col_gin_model.json")
                       <<colNet.to_json().dump(2);
            }else if(++wait>=patience){
                logInfo("early-stop"); break;
            }
        }
    }

    /* ------------ evaluation (unchanged) ------------- */
    size_t TP=0,FP=0,TN=0,FN=0; size_t n=0;
    double rt_row=0,rt_col=0,rt_rule=0,rt_router=0,rt_hyb=0,rt_fann=0,rt_opt=0;

    for(const Sample& s : ds)
    {
        ++n;
        double pr = std::expm1(rowNet.predict(s.rowG));
        double pc = std::expm1(colNet.predict(s.colG));

        rt_row  += (s.rt - rt_row )/n;
        rt_col  += (s.ct - rt_col )/n;
        double rule = (s.query_cost > costThr)? s.ct:s.rt;
        rt_rule += (rule - rt_rule)/n;
        bool dec = (pc<pr);
        rt_router+=((dec? s.ct:s.rt) - rt_router)/n;
        rt_hyb  += ((s.hybrid_use_imci? s.ct:s.rt) - rt_hyb )/n;
        rt_fann += ((s.fann_use_imci  ? s.ct:s.rt) - rt_fann)/n;
        rt_opt  += (std::min(s.rt,s.ct) - rt_opt)/n;

        bool truth=(s.ct<s.rt);
        if( truth &&  dec) ++TP;
        if(!truth &&  dec) ++FP;
        if(!truth && !dec) ++TN;
        if( truth && !dec) ++FN;
    }

    double prec=TP?double(TP)/(TP+FP):0, rec=TP?double(TP)/(TP+FN):0;
    double f1  =(prec+rec)?2*prec*rec/(prec+rec):0;

    std::cout<<"\n=== CONFUSION (Tree-CNN router) ===\n"
             <<"TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN
             <<"\nPrecision="<<prec<<" Recall="<<rec<<" F1="<<f1<<"\n\n";

    std::cout<<"=== AVG RUNTIME (s) ===\n"
             <<"Row-only      : "<<rt_row
             <<"\nColumn-only   : "<<rt_col
             <<"\nCost rule     : "<<rt_rule
             <<"\nTree-CNN route: "<<rt_router
             <<"\nHybrid flag   : "<<rt_hyb
             <<"\nFANN flag     : "<<rt_fann
             <<"\nOptimal       : "<<rt_opt<<'\n';
    return 0;
}