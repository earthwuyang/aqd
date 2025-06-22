/************************************************************************************
 * row_column_gnn.cpp
 *
 * Dual‐regression Row/Column router (updated):
 *
 *   – Row‐time regressor: **63** → hidden → 1 MLP (Eigen), trained in batches of 128 with OpenMP
 *   – Column‐time regressor: GIN encoder + linear head
 *   – Early‐stopping on validation MSE
 *   – Classify by comparing predicted times
 *   – Reports Q-error stats for both regressors
 *
 * Compile:
 *   g++ -std=c++11 -O2 -fopenmp -I/path/to/eigen3 -I/path/to/json/single_include \
 *       -o row_column_gnn row_column_gnn.cpp
 *
 ************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <omp.h>
#include <mysql/mysql.h>
#include <regex>
#include <Eigen/Dense>
#include "json.hpp"
#include <array>
#include <cstring>

using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using namespace std;

#ifndef DELTA
#define DELTA 1.0
#endif

// Huber loss gradient
inline double dhuber(double err) {
  double a = std::abs(err);
  if (a <= DELTA) return err;
  return DELTA * ((err > 0) ? 1.0 : -1.0);
}

inline double safe_q_error(double pred, double truth) {
  const double eps = 1e-6;
  if (pred < eps || truth < eps) return std::numeric_limits<double>::infinity();
  double q = pred / truth;
  if (q < 1.0) q = 1.0 / q;
  return q;
}

static void progress(const std::string& tag,
                     std::size_t cur, std::size_t tot,
                     std::size_t w = 40)
{
    double f = tot ? double(cur) / tot : 1.0;
    std::size_t filled = std::size_t(f * w);
    std::cout << '\r' << tag << " ["
              << std::string(filled, '=')
              << std::string(w - filled, ' ')
              << "] " << std::setw(3) << int(f * 100) << "% ("
              << cur << '/' << tot << ')' << std::flush;
    if (cur == tot) std::cout << '\n';
}

/*────────────────────────────── logging ──────────────────────────────*/
static void logInfo (const std::string &m){ std::cout  << "[INFO]  " << m << '\n'; }
static void logWarn (const std::string &m){ std::cerr  << "[WARN]  " << m << '\n'; }
static void logError(const std::string &m){ std::cerr  << "[ERROR] " << m << '\n'; }

static double safe_f(const json&v){
    if(v.is_number())  return v.get<double>();
    if(v.is_string())  { try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
static double safe_f(const json&obj,const char*key){
    if(!obj.contains(key)) return 0.0;
    return safe_f(obj[key]);
}

/*──────────────────────────── filesystem ────────────────────────────*/
bool fileExists(const std::string &p){
    std::ifstream f(p);
    return f.good();
}

/*───────────────────────────── progress ─────────────────────────────*/
void showProgressBar(size_t cur,size_t tot,size_t w=50){
    double frac = tot? double(cur)/tot : 1.0;
    size_t filled = size_t(frac*w);
    std::cout << "\r[";
    for(size_t i=0;i<filled;++i) std::cout << '=';
    for(size_t i=filled;i<w;++i) std::cout << ' ';
    std::cout << "] " << int(frac*100) << "% ("<<cur<<"/"<<tot<<")";
    if(cur>=tot) std::cout<<'\n';
    std::fflush(stdout);
}

/*──────────────────────── utilities ──────────────────────────*/
double parseNum(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string()) return std::stod(j[k].get<std::string>());
        if(j[k].is_number()) return j[k].get<double>();
    } catch(...) { /*ignore*/ }
    return 0.0;
}
double safeLog1p(double v){ return std::log1p(std::max(0.0,v)); }

/* ─────────── feature constants ─────────── */
// constexpr int NUM_FEATS  = 63;        // ← 43 old + 1 all-covered flag
constexpr double GAP_EMPH=2.0;

/*────────────────── random‐forest’s plan2feat ──────────────────*/
static const int NUM_FEATS = 63;
static std::unordered_map<std::string,std::unordered_set<std::string>> indexCols;

static double strSizeToNum(std::string s){
    if(s.empty()) return 0;
    char suf=s.back(); double m=1;
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return std::stod(s)*m; }catch(...){ return 0; }
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

static bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    auto &v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){
        std::string s=v.get<std::string>(); std::transform(s.begin(),s.end(),s.begin(),::tolower);
        return s=="true"||s=="yes"||s=="1";
    }
    return false;
}

struct Agg {
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0;
    int cnt=0, cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0;
    double selSum=0,selMin=1e30,selMax=0,fanoutMax=0;
    double ratioSum=0,ratioMax=0;
    bool grp=false, ord=false, tmp=false;
    int coverCount=0;
    int maxDepth=0;
};

static void walk(const json&n, Agg&a, int depth){
    if(n.is_object()){
        if(n.contains("table")){
            auto &t = n["table"];
            double re = parseNum(t,"rows_examined_per_scan");
            double rp = parseNum(t,"rows_produced_per_join");
            double fl = parseNum(t,"filtered");
            double rc=0,ec=0, pc=0, dr=0;
            if(t.contains("cost_info")){
                auto &ci = t["cost_info"];
                rc = parseNum(ci,"read_cost");
                ec = parseNum(ci,"eval_cost");
                pc = parseNum(ci,"prefix_cost");
                if(ci.contains("data_read_per_join")){
                    if(ci["data_read_per_join"].is_string())
                        dr = strSizeToNum(ci["data_read_per_join"].get<std::string>());
                    else dr = parseNum(ci,"data_read_per_join");
                }
            }
            a.re += re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr;
            a.cnt++;
            if(re>0){ double sel=rp/re; a.selSum+=sel; a.selMin=fmin(a.selMin,sel); a.selMax=fmax(a.selMax,sel); a.fanoutMax=fmax(a.fanoutMax,sel);}
            double ratio = ec>0?rc/ec:rc;
            a.ratioSum += ratio; a.ratioMax=fmax(a.ratioMax,ratio);
            std::string at = t.value("access_type","ALL");
            if      (at=="range")  a.cRange++;
            else if (at=="ref")    a.cRef++;
            else if (at=="eq_ref") a.cEq++;
            else if (at=="index")  a.cIdx++;
            else                   a.cFull++;
            if(getBool(t,"using_index")) a.idxUse++;
            if(t.contains("used_columns")&&t.contains("key")){
                std::string idx = t["key"].get<std::string>();
                auto it = indexCols.find(idx);
                if(it!=indexCols.end()){
                    bool full=true;
                    for(auto &u:t["used_columns"])
                        if(!it->second.count(u.get<std::string>())){ full=false; break;}
                    if(full) a.coverCount++;
                }
            }
        }
        if(n.contains("grouping_operation"))             a.grp=true;
        if(n.contains("ordering_operation")||getBool(n,"using_filesort")) a.ord=true;
        if(getBool(n,"using_temporary_table"))           a.tmp=true;
        for(auto&kv:n.items()) if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }
    else if(n.is_array()){
        for(auto &v:n) walk(v,a,depth);
    }
    a.maxDepth = std::max(a.maxDepth, depth);
}

static bool plan2feat(const json&plan, float f[NUM_FEATS]){
    if(!plan.contains("query_block")) return false;
    const json* qb = &plan["query_block"];
    if(qb->contains("union_result")){
        auto &specs = (*qb)["union_result"]["query_specifications"];
        if(specs.is_array()&&!specs.empty())
            qb = &specs[0]["query_block"];
    }
    Agg a{};
    walk(*qb,a,1);
    if(a.cnt==0) return false;
    double inv = 1.0 / a.cnt;
    int k=0;
    double qCost  = parseNum(qb->value("cost_info", json::object()),"query_cost");
    double rootRow= parseNum(*qb,"rows_produced_per_join");
    // 0–6
    f[k++] = safeLog1p(a.re*inv);
    f[k++] = safeLog1p(a.rp*inv);
    f[k++] = safeLog1p(a.f *inv);
    f[k++] = safeLog1p(a.rc*inv);
    f[k++] = safeLog1p(a.ec*inv);
    f[k++] = safeLog1p(a.pc*inv);
    f[k++] = safeLog1p(a.dr*inv);
    // 7–12
    f[k++] = a.cRange*inv; f[k++] = a.cRef*inv; f[k++] = a.cEq*inv;
    f[k++] = a.cIdx*inv;   f[k++] = a.cFull*inv;f[k++] = a.idxUse*inv;
    // 13–17
    f[k++] = a.selSum*inv; f[k++] = a.selMin;  f[k++] = a.selMax;
    f[k++] = a.maxDepth;   f[k++] = a.fanoutMax;
    // 18–20
    f[k++] = a.grp?1.f:0.f; f[k++] = a.ord?1.f:0.f; f[k++] = a.tmp?1.f:0.f;
    // 21–22
    f[k++] = a.ratioSum*inv; f[k++] = a.ratioMax;
    // 23–24
    f[k++] = safeLog1p(qCost);
    f[k++] = safeLog1p(rootRow);
    // 25–27
    f[k++] = safeLog1p((a.pc*inv)/std::max(1e-6, a.rc*inv));
    f[k++] = safeLog1p((a.rc*inv)/std::max(1e-6, a.re*inv));
    f[k++] = safeLog1p((a.ec*inv)/std::max(1e-6, a.re*inv));
    // 28–29
    f[k++] = (a.cnt==1)?1.f:0.f;
    f[k++] = (a.cnt>1)?1.f:0.f;
    // 30–31
    f[k++] = safeLog1p(a.maxDepth*(a.idxUse*inv));
    { double fullFrac = a.cFull? (a.cFull*inv) : 1e-3;
      f[k++] = safeLog1p((a.idxUse*inv)/fullFrac); }
    // 32–39
    f[k++] = float(a.cnt);
    f[k++] = a.cnt? float(a.selSum)/a.cnt : 0.f;  // reused slot
    f[k++] = safeLog1p(a.pc*inv);
    f[k++] = safeLog1p(a.rc*inv);
    f[k++] = (a.cnt>1)? float(a.cnt-1)/a.cnt : 0.f;
    f[k++] = rootRow>0? float(a.re)/rootRow : 0.f;
    f[k++] = a.selMax - a.selMin;
    { float denom = float(a.cRange+a.cRef+a.cEq+a.cIdx);
      f[k++] = denom? float(a.idxUse)/denom : 0.f; }
    // 40–41
    f[k++] = float(qCost);
    f[k++] = (qCost>5e4)?1.f:0.f;
    // 42–43
    f[k++] = a.cnt? float(a.coverCount)/a.cnt : 0.f;
    f[k++] = (a.coverCount==a.cnt)?1.f:0.f;
    // 44
    f[k++] = safeLog1p(a.re*inv) - safeLog1p(a.selSum*inv);
    // 45–46
    f[k++] = float(a.cnt);
    f[k++] = safeLog1p(a.cnt);
    // 47–48
    f[k++] = float(a.cnt);           // reused
    f[k++] = a.cnt? float(a.coverCount)/a.cnt : 0.f;
    // 49–50
    f[k++] = float(a.cnt);           // reused
    f[k++] = a.cnt? float(a.coverCount)/a.cnt : 0.f;
    // 51–56
    f[k++] = a.idxUse*inv;
    f[k++] = a.cRange*inv;
    f[k++] = a.cRef*inv;
    f[k++] = a.cEq*inv;
    f[k++] = a.cIdx*inv;
    f[k++] = a.cFull*inv;
    // 57–59
    f[k++] = safeLog1p(a.pc*inv);
    f[k++] = safeLog1p(a.rc*inv);
    f[k++] = a.selMax - a.selMin;
    // 60–62
    f[k++] = a.ratioMax;
    f[k++] = a.fanoutMax;
    f[k++] = a.selMin>0? (a.selMax/a.selMin) : 0.f;
    assert(k==NUM_FEATS);
    return true;
}

/* ─────────── CART decision tree ─────────── */
struct DTNode{ int feat=-1,left=-1,right=-1; float thr=0,prob=0; };
class DecisionTree {
  vector<DTNode> nodes_;
  double min_gain_;
  int max_depth_, min_samples_;
  /* Gini impurity helper */
  static inline double gini(double pos,double tot){
    if(tot<=0) return 1.0;
    double q=pos/tot; return 2.0*q*(1.0-q);
  }
  /* build recursively */
  int build(const std::vector<int>& idx,
              const std::vector<std::array<float,NUM_FEATS>>& X,
              const std::vector<int>&  y,
              const std::vector<float>& w,
              int depth,int max_depth,int min_samples)
    {
        /* ---- weighted positive / total ---- */
        double pos_w = 0.0, tot_w = 0.0;
        for(int i:idx){ tot_w += w[i]; pos_w += w[i]*y[i]; }

        DTNode node; node.prob = float( pos_w / std::max(1e-12, tot_w) );

        if(depth>=max_depth || idx.size()<=min_samples ||
           node.prob<=0.02  || node.prob>=0.98){
            nodes_.push_back(node);           // make this a leaf
            return int(nodes_.size())-1;
        }

        int   best_f = -1;
        float best_thr = 0.f, best_g = std::numeric_limits<float>::max();

        for(int f=0; f<NUM_FEATS; ++f){
            std::vector<float> vals; vals.reserve(idx.size());
            for(int i:idx) vals.push_back(X[i][f]);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(),vals.end()), vals.end());
            if(vals.size()<2) continue;

            std::vector<float> cand;
            int steps = std::min<int>(9, vals.size()-1);
            for(int s=1; s<=steps; ++s)
                cand.push_back(vals[ vals.size()*s/(steps+1) ]);

            for(float thr : cand){
                double l_tot=0,l_pos=0,r_tot=0,r_pos=0;
                int    l_cnt=0,r_cnt=0;
                for(int i:idx){
                    if(X[i][f] < thr){
                        l_tot+=w[i]; l_pos+=w[i]*y[i]; l_cnt++;
                    }else{
                        r_tot+=w[i]; r_pos+=w[i]*y[i]; r_cnt++;
                    }
                }
                if(l_cnt<min_samples || r_cnt<min_samples) continue;

                auto gini = [](double pos,double tot){
                    if(tot<=0) return 1.0;
                    double q = pos/tot;
                    return 2.0*q*(1.0-q);
                };
                float g = gini(l_pos,l_tot) + gini(r_pos,r_tot);
                if(g < best_g){ best_g=g; best_f=f; best_thr=thr; }
            }
        }

        if(best_f==-1){                       // could not split → leaf
            nodes_.push_back(node);
            return int(nodes_.size())-1;
        }

        std::vector<int> left, right;
        for(int i:idx){
            (X[i][best_f] < best_thr ? left : right).push_back(i);
        }
        node.feat = best_f; node.thr = best_thr;
        int self = int(nodes_.size());
        nodes_.push_back(node);               // placeholder

        int l = build(left ,X,y,w,depth+1,max_depth,min_samples);
        int r = build(right,X,y,w,depth+1,max_depth,min_samples);
        nodes_[self].left  = l;
        nodes_[self].right = r;
        return self;
    }
public:
  DecisionTree(int md,int ms,double mg)
    : min_gain_(mg), max_depth_(md), min_samples_(ms) {}

  void fit(const vector<array<float,NUM_FEATS>>&X,
           const vector<int>&y,const vector<float>&w){
    nodes_.clear();
    vector<int> idx(X.size());
    iota(idx.begin(),idx.end(),0);
    // build(idx,X,y,w,0);
    build(idx, X, y, w, 0, max_depth_, min_samples_);
  }

  float predict(const float*f) const {
    // if (f[43]==1.0f) { return 0.0f; }
    int id=0;
    while(nodes_[id].feat!=-1)
      id = (f[nodes_[id].feat] < nodes_[id].thr)
           ? nodes_[id].left
           : nodes_[id].right;
    return nodes_[id].prob;
  }

  json to_json() const {
    json arr = json::array();
    for(const auto& n: nodes_)
      arr.push_back({
        {"feat",  n.feat},
        {"thr",   n.thr},
        {"left",  n.left},
        {"right", n.right},
        {"prob",  n.prob}
      });
    return arr;
  }

  // ─────── NEW from_json ───────
  void from_json(const json& arr) {
    nodes_.clear();
    for(const auto& e: arr) {
      DTNode n;
      n.feat  = e.at("feat").get<short>();
      n.thr   = e.at("thr").get<float>();
      n.left  = e.at("left").get<short>();
      n.right = e.at("right").get<short>();
      n.prob  = e.at("prob").get<float>();
      nodes_.push_back(n);
    }
  }
};


/* ─────────── Random-Forest ─────────── */
class RandomForest {
  std::vector<DecisionTree> trees_;
  double sampleRatio_;
public:
  RandomForest(int n,int md,int ms,double mg,double sr)
    : sampleRatio_(sr)
  {
    trees_.reserve(n);
    for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
  }

  void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
           const std::vector<int>&  y,
           const std::vector<float>& w,
           std::mt19937& rng)
  {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    std::uniform_int_distribution<int> uni(0, int(X.size() - 1));
    std::size_t T = trees_.size();
    #pragma omp parallel for schedule(dynamic)
    for(int t = 0; t < int(T); ++t)
    {
      std::mt19937 rng_local(rng()+t);
      std::uniform_int_distribution<int> uni_local(0, int(X.size()-1));

      std::size_t m = std::size_t(sampleRatio_ * X.size());
      std::vector<std::array<float,NUM_FEATS>> BX; BX.reserve(m);
      std::vector<int>  By;  By.reserve(m);
      std::vector<float>Bw;  Bw.reserve(m);

      for(std::size_t i=0;i<m;++i){
        int j = uni_local(rng_local);
        BX.push_back(X[j]);
        By.push_back(y[j]);
        Bw.push_back(w[j]);
      }

      trees_[t].fit(BX,By,Bw);
      #pragma omp critical
      progress("RF-train", t+1, T);
    }

    auto t1 = clock::now();
    logInfo("Random-Forest training took " +
         std::to_string(std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()) +
         " s");
  }

  float predict(const float*f) const {
    double s=0;
    for(const auto& tr: trees_) s += tr.predict(f);
    return float(s/trees_.size());
  }

  json to_json() const {
    json arr = json::array();
    for(const auto& tr: trees_) arr.push_back(tr.to_json());
    return arr;
  }

  // ─────── NEW from_json ───────
  void from_json(const json& arr) {
    trees_.clear();
    for(const auto& jt : arr) {
      DecisionTree tr(0,0,0);
      tr.from_json(jt);
      trees_.push_back(std::move(tr));
    }
  }
};


/*─────────────────── column stats & parseColPlan ───────────────────*/

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

/*────────────────── dataset loader (with 63‐dim row feats) ──────────────────*/
struct Sample {
    float  feats[NUM_FEATS];
    Graph  colGraph;
    int    label;        // 0=row,1=col,-1=skip
    double row_time, col_time, query_cost;
    int    hybrid_use_imci, fann_use_imci;
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

/*─────────────────────── GIN & MLPReg as defined above ─────────────────────*/

/* ──────────────────────── GIN encoder ──────────────────────── */
struct GIN {
    int in_dim, hidden, n_ops;

    MatrixXd W1, W2;
    VectorXd b1, b2;
    MatrixXd opEmb;

    RowVectorXd w_col;
    double      b_col;

    std::mt19937                      rng;
    std::uniform_real_distribution<> dist;  // uniform [-0.1,0.1]

    GIN(int n_ops_, int hidden_)
    : in_dim(2 + 8),
        hidden(hidden_),
        n_ops(n_ops_),
        W1(hidden, in_dim),
        b1(hidden),
        W2(hidden, hidden),
        b2(hidden),
        opEmb(8, n_ops_),
        w_col(hidden),
        b_col(0.0),
        rng(123),
        dist(-0.1, 0.1)
    {
        // Initialize W1
        for (int i = 0; i < W1.rows(); ++i)
            for (int j = 0; j < W1.cols(); ++j)
                W1(i, j) = dist(rng);

        // Initialize b1
        for (int i = 0; i < b1.size(); ++i)
            b1(i) = dist(rng);

        // Initialize W2
        for (int i = 0; i < W2.rows(); ++i)
            for (int j = 0; j < W2.cols(); ++j)
                W2(i, j) = dist(rng);

        // Initialize b2
        for (int i = 0; i < b2.size(); ++i)
            b2(i) = dist(rng);

        // Initialize opEmb
        for (int i = 0; i < opEmb.rows(); ++i)
            for (int j = 0; j < opEmb.cols(); ++j)
                opEmb(i, j) = dist(rng);

        // Initialize w_col
        for (int i = 0; i < w_col.size(); ++i)
            w_col(i) = dist(rng);

        // b_col remains zero
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

    json to_json() const {
    return json{
      {"in_dim",      in_dim},
      {"hidden",      hidden},
      {"n_ops",       n_ops},
      {"W1",          std::vector<double>(W1.data(),   W1.data()   + W1.size())},
      {"b1",          std::vector<double>(b1.data(),   b1.data()   + b1.size())},
      {"W2",          std::vector<double>(W2.data(),   W2.data()   + W2.size())},
      {"b2",          std::vector<double>(b2.data(),   b2.data()   + b2.size())},
      {"opEmb",       std::vector<double>(opEmb.data(),opEmb.data()+opEmb.size())},
      {"w_col",       std::vector<double>(w_col.data(),w_col.data()+w_col.size())},
      {"b_col",       b_col}
    };
  }

  // deserialize
  void from_json(const json& j) {
    // 1) resize opEmb to the trained size
    int new_n_ops = j.at("n_ops").get<int>();
    if (new_n_ops != n_ops) {
      n_ops = new_n_ops;
      opEmb.resize(8, n_ops);
    }

    // 2) copy all the weight arrays
    auto vecW1 = j.at("W1").get<std::vector<double>>();
    std::copy(vecW1.begin(), vecW1.end(), W1.data());

    auto vecb1 = j.at("b1").get<std::vector<double>>();
    std::copy(vecb1.begin(), vecb1.end(), b1.data());

    auto vecW2 = j.at("W2").get<std::vector<double>>();
    std::copy(vecW2.begin(), vecW2.end(), W2.data());

    auto vecb2 = j.at("b2").get<std::vector<double>>();
    std::copy(vecb2.begin(), vecb2.end(), b2.data());

    auto vecOp = j.at("opEmb").get<std::vector<double>>();
    std::copy(vecOp.begin(), vecOp.end(), opEmb.data());

    auto vecwc = j.at("w_col").get<std::vector<double>>();
    std::copy(vecwc.begin(), vecwc.end(), w_col.data());

    b_col = j.at("b_col").get<double>();
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


/*──────────────── dataset split ─────────────────*/
std::pair<std::vector<size_t>,std::vector<size_t>>
splitDataset(size_t n,double ratio=0.8){
    std::vector<size_t> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::shuffle(idx.begin(),idx.end(),std::mt19937(std::random_device{}()));
    size_t k = size_t(ratio * n);
    return {{idx.begin(),idx.begin()+k},{idx.begin()+k,idx.end()}};
}


/* ─────────── CSV / plan loader ─────────── */
static bool load_plan(const string& path, float f[NUM_FEATS], double &qcost) {
    ifstream in(path); if(!in) return false;
    json j; try{ in>>j; }catch(...){ 
      printf("in to j fails\n"); 
      return false; 
    }
    if(!plan2feat(j,f)) { 
      // printf("plan2feat return false\n"); 
      return false; 
    }
    qcost=0; if(j.contains("query_block"))
        qcost=safe_f(j["query_block"].value("cost_info",json::object()),"query_cost");
    return true;
}



//------------------------------------------------------------------------------
// Load all INDEX definitions (name → set of columns) for a single table
//------------------------------------------------------------------------------
static void load_index_defs_from_db(const std::string &host,
                                    int                 port,
                                    const std::string  &user,
                                    const std::string  &pass,
                                    const std::string  &db,
                                    const std::string  &table_name)
{
    // connect to MySQL
    MYSQL *conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn,
                            host.c_str(),
                            user.c_str(),
                            pass.c_str(),
                            db.c_str(),
                            port,
                            nullptr,
                            0))
    {
        logWarn("MySQL connect failed: " + std::string(mysql_error(conn)));
        return;
    }

    // SHOW CREATE TABLE
    std::string q = "SHOW CREATE TABLE `" + table_name + "`";
    if (mysql_query(conn, q.c_str()))
    {
        logWarn("SHOW CREATE TABLE failed: " + std::string(mysql_error(conn)));
        mysql_close(conn);
        return;
    }

    MYSQL_RES *res = mysql_store_result(conn);
    if (!res) { mysql_close(conn); return; }
    MYSQL_ROW row = mysql_fetch_row(res);
    if (!row || !row[1]) { mysql_free_result(res); mysql_close(conn); return; }

    // grab the DDL
    std::string ddl = row[1];
    mysql_free_result(res);
    mysql_close(conn);

    // regex to find: KEY `idx_name` (`colA`,`colB`,…)
    static const std::regex re(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))", std::regex::icase);
    std::smatch m;
    auto it  = ddl.cbegin();
    auto end = ddl.cend();

    // clear any previous definitions
    indexCols.clear();

    while (std::regex_search(it, end, m, re))
    {
        // 1) index name
        std::string idx  = m[1].str();
        // 2) raw column list inside parentheses
        std::string cols = m[2].str();

        // split cols on commas
        std::unordered_set<std::string> S;
        std::istringstream colss(cols);
        std::string tok;
        while (std::getline(colss, tok, ','))
        {
            // strip backticks
            tok.erase(std::remove(tok.begin(), tok.end(), '`'), tok.end());
            // trim whitespace
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t") + 1);

            if (!tok.empty())
                S.insert(tok);
        }

        // insert into global map
        indexCols.emplace(std::move(idx), std::move(S));

        // advance past this match
        it = m.suffix().first;
    }

    logInfo("Loaded " + std::to_string(indexCols.size())
         + " indexes from `" + table_name + "`");
}


// ─────────── loads all index→column maps for every table in each database ───────────
static void load_all_index_defs(const string &host,
                                int          port,
                                const string &user,
                                const string &pass,
                                const vector<string> &dbs)
{
    for (const auto &db : dbs) {
        MYSQL *conn = mysql_init(nullptr);
        if (!mysql_real_connect(conn,
                                host.c_str(), user.c_str(), pass.c_str(),
                                db.c_str(), port, nullptr, 0))
        {
            logWarn("MySQL connect to `" + db + "` failed: " + mysql_error(conn));
            mysql_close(conn);
            continue;
        }
        // switch into that database
        if (mysql_query(conn, ("USE `" + db + "`").c_str())) {
            logWarn("USE `" + db + "` failed: " + mysql_error(conn));
            mysql_close(conn);
            continue;
        }
        // list all tables
        if (mysql_query(conn, "SHOW TABLES")) {
            logWarn("SHOW TABLES in `" + db + "` failed: " + mysql_error(conn));
            mysql_close(conn);
            continue;
        }
        MYSQL_RES *tables = mysql_store_result(conn);
        if (!tables) { mysql_close(conn); continue; }

        MYSQL_ROW row;
        while ((row = mysql_fetch_row(tables))) {
            string tbl = row[0];
            // for each table, pull in its SHOW CREATE → indexCols
            load_index_defs_from_db(host, port, user, pass, db, tbl);
        }
        mysql_free_result(tables);
        mysql_close(conn);
    }
}


// int main(int argc, char* argv[]) {
//     using Clock = std::chrono::steady_clock;
//     auto t_start = Clock::now();

//     // ─── CLI defaults ───
//     std::vector<std::string> dataDirs;
//     int    hiddenNeurons = 128;
//     int    nTrees        = 32;
//     int    maxDepth      = 15;
//     int    minSamples    = 40;
//     double minGain       = 0.0005;
//     double sampleRatio   = 0.6;
//     bool   skipLoad      = false;
//     std::string rfModelPath  = "rf_model_more_features.json";
//     std::string rowModelPath = "row_mlp_model.json";
//     std::string colModelPath = "col_gin_model.json";
//     double delta        = 1.0;
//     const int    timeout = 60;

//     // ─── Parse args ───
//     for (int i = 1; i < argc; ++i) {
//         std::string a(argv[i]);
//         if      (a.rfind("--data_dirs=",0)==0)     dataDirs.push_back(a.substr(12));
//         else if (a.rfind("--hidden_neurons=",0)==0) hiddenNeurons = std::stoi(a.substr(17));
//         else if (a.rfind("--rf_model=",0)==0)      rfModelPath   = a.substr(11);
//         else if (a.rfind("--row_model=",0)==0)     rowModelPath  = a.substr(12);
//         else if (a.rfind("--col_model=",0)==0)     colModelPath  = a.substr(12);
//         else if (a.rfind("--delta=",0)==0)         delta         = std::stod(a.substr(8));
//         else if (a == "--skip_load")               skipLoad      = true;
//     }
//     if (dataDirs.empty()) {
//         logError("need --data_dirs=...");
//         return 1;
//     }

//     // ─── Load dataset ───
//     ExecutionPlanDataset ds(dataDirs, double(timeout));
//     if (ds.size() < size_t(minSamples*2)) {
//         logError("Too few samples");
//         return 1;
//     }

//     // ─── Compute per-feature mean/std for 63 dims ───
//     std::vector<double> feat_mean(NUM_FEATS, 0.0), feat_std(NUM_FEATS, 0.0);
//     for (size_t i = 0; i < ds.size(); ++i)
//       for (int j = 0; j < NUM_FEATS; ++j)
//         feat_mean[j] += ds[i].feats[j];
//     for (int j = 0; j < NUM_FEATS; ++j)
//       feat_mean[j] /= double(ds.size());
//     for (size_t i = 0; i < ds.size(); ++i)
//       for (int j = 0; j < NUM_FEATS; ++j) {
//         double d = ds[i].feats[j] - feat_mean[j];
//         feat_std[j] += d*d;
//       }
//     for (int j = 0; j < NUM_FEATS; ++j) {
//       feat_std[j] = std::sqrt(feat_std[j]/double(ds.size()));
//       if (feat_std[j] < 1e-6) feat_std[j] = 1.0;
//     }

//     // ─── Load RandomForest ───
//     RandomForest rf(nTrees, maxDepth, minSamples, minGain, sampleRatio);
//     {
//       json j;
//       std::ifstream f(rfModelPath);
//       f >> j;
//       rf.from_json(j["forest"]);
//       logInfo("Loaded RF → " + rfModelPath);
//     }

//     // ─── Load MLPReg (row‐time) ───
//     MLPReg rowReg(NUM_FEATS, hiddenNeurons, 0.0);
//     {
//       json j; std::ifstream f(rowModelPath); f >> j;
//       rowReg.from_json(j);
//       logInfo("Loaded row‐MLP → " + rowModelPath);
//     }

//     // ─── Load GIN (col‐time) ───
//     json j; std::ifstream f(colModelPath); f >> j;
//     int trained_n_ops = j.at("n_ops").get<int>();
//     GIN colReg(trained_n_ops, hiddenNeurons);
//     colReg.from_json(j);
//     logInfo("Loaded col-GIN → " + colModelPath);
//     // int maxOp = 0;
//     // for (size_t i = 0; i < ds.size(); ++i)
//     //   for (int op : ds[i].colGraph.op)
//     //     maxOp = std::max(maxOp, op);
//     // GIN colReg(maxOp+1, hiddenNeurons);
//     // {
//     //   json j; std::ifstream f(colModelPath); f >> j;
//     //   colReg.from_json(j);
//     //   logInfo("Loaded col‐GIN → " + colModelPath);
//     // }

//     // ─── Evaluate router ───
//     size_t TP=0, FP=0, TN=0, FN=0;
//     double rt_row=0, rt_col=0, rt_cost=0, rt_hyb=0, rt_fann=0;
//     double rt_pred=0, rt_rf_all=0, rt_rf_close=0, rt_router=0, rt_opt=0;
//     size_t n=0, n_cost=0, n_hyb=0, n_fann=0, n_pred=0, n_rf_all=0, n_rf_close=0;

//     for (size_t i = 0; i < ds.size(); ++i) {
//       auto &s = ds[i];
//       progress("eval", i+1, ds.size());
//       if (s.label < 0) continue;
//       ++n;

//       // ground-truth averages
//       rt_row += (s.row_time - rt_row)/n;
//       rt_col += (s.col_time - rt_col)/n;
//       rt_opt += (std::min(s.row_time, s.col_time) - rt_opt)/n;

//       // cost-threshold method
//       ++n_cost;
//       {
//         double act = (s.query_cost > /*costThreshold*/ 5e4)
//                      ? s.col_time : s.row_time;
//         rt_cost += (act - rt_cost)/n_cost;
//       }

//       // original hybrid optimizer decision
//       ++n_hyb;
//       {
//         double act = s.hybrid_use_imci ? s.col_time : s.row_time;
//         rt_hyb += (act - rt_hyb)/n_hyb;
//       }

//       // FANN-based decision
//       ++n_fann;
//       {
//         double act = s.fann_use_imci ? s.col_time : s.row_time;
//         rt_fann += (act - rt_fann)/n_fann;
//       }

//       // normalize features
//       std::vector<float> x63(NUM_FEATS);
//       for (int j = 0; j < NUM_FEATS; ++j)
//         x63[j] = float((s.feats[j] - feat_mean[j]) / feat_std[j]);

//       // predicted-time decision
//       ++n_pred;
//       {
//         double pr = rowReg.run(x63);
//         double pc = colReg.predict_col_time(s.colGraph);
//         double act = (pc < pr) ? s.col_time : s.row_time;
//         rt_pred += (act - rt_pred)/n_pred;
//       }

//       // RF-only on all datapoints
//       ++n_rf_all;
//       {
//         float rf_feat[NUM_FEATS];
//         for (int j = 0; j < NUM_FEATS; ++j) rf_feat[j] = s.feats[j];
//         bool v_rf_all = rf.predict(rf_feat) >= 0.5f;
//         double act = v_rf_all ? s.col_time : s.row_time;
//         rt_rf_all += (act - rt_rf_all)/n_rf_all;
//       }

//       // ensemble router: if close use RF, else predicted-time
//       {
//         double pr = rowReg.run(x63);
//         double pc = colReg.predict_col_time(s.colGraph);
//         bool choose_col;
//         // RF-close
//         if (std::fabs(pr - pc) <= delta) {
//           ++n_rf_close;
//           float rf_feat[NUM_FEATS];
//           for (int j = 0; j < NUM_FEATS; ++j) rf_feat[j] = s.feats[j];
//           choose_col = (rf.predict(rf_feat) >= 0.5f);
//           double act = choose_col ? s.col_time : s.row_time;
//           rt_rf_close += (act - rt_rf_close)/n_rf_close;
//         } else {
//           choose_col = (pc < pr);
//         }
//         // router overall
//         double act = choose_col ? s.col_time : s.row_time;
//         rt_router += (act - rt_router)/n;

//         // confusion
//         if (s.label &&  choose_col) ++TP;
//         if (!s.label &&  choose_col) ++FP;
//         if (!s.label && !choose_col) ++TN;
//         if ( s.label && !choose_col) ++FN;
//       }
//     }

//     double prec = TP ? double(TP)/(TP+FP) : 0.0;
//     double rec  = TP ? double(TP)/(TP+FN) : 0.0;
//     double f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : 0.0;

//     std::cout << "\n=== FINAL CONFUSION ===\n"
//               << "TP=" << TP << " FP=" << FP
//               << " TN=" << TN << " FN=" << FN << "\n"
//               << "Precision=" << prec
//               << " Recall=" << rec
//               << " F1=" << f1 << "\n\n";

//     std::cout << "=== AVG RUNTIMES (s/query) ===\n"
//               << "Row only          : " << rt_row    << "\n"
//               << "Column only       : " << rt_col    << "\n"
//               << "Cost threshold    : " << rt_cost   << "\n"
//               << "Hybrid optimizer  : " << rt_hyb    << "\n"
//               << "FANN-based        : " << rt_fann   << "\n"
//               << "Predicted-time    : " << rt_pred   << "\n"
//               << "RF-only (all)     : " << rt_rf_all << "\n"
//               << "RF-only (close)   : " << rt_rf_close << "\n"
//               << "Ensemble router   : " << rt_router << "\n"
//               << "Optimal oracle    : " << rt_opt    << "\n";

//     double wall = std::chrono::duration_cast<std::chrono::seconds>(
//                       Clock::now() - t_start
//                   ).count();
//     logInfo("Total wall-clock: " + std::to_string(wall) + " s");

//     return 0;
// }


int main(int argc, char* argv[]) {
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    // ─── CLI defaults ───
    std::vector<std::string> dataDirs;
    int    hiddenNeurons = 128;
    int    nTrees        = 32;
    int    maxDepth      = 15;
    int    minSamples    = 40;
    double minGain       = 0.0005;
    double sampleRatio   = 0.6;
    bool   skipLoad      = false;
    std::string rfModelPath  = "rf_model_more_features.json";
    std::string rowModelPath = "row_mlp_model.json";
    std::string colModelPath = "col_gin_model.json";
    double delta        = 1.0;
    const int    timeout = 60;

    // ─── Parse args ───
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a.rfind("--data_dirs=",0)==0)     dataDirs.push_back(a.substr(12));
        else if (a.rfind("--hidden_neurons=",0)==0) hiddenNeurons = std::stoi(a.substr(17));
        else if (a.rfind("--rf_model=",0)==0)      rfModelPath   = a.substr(11);
        else if (a.rfind("--row_model=",0)==0)     rowModelPath  = a.substr(12);
        else if (a.rfind("--col_model=",0)==0)     colModelPath  = a.substr(12);
        else if (a.rfind("--delta=",0)==0)         delta         = std::stod(a.substr(8));
        else if (a == "--skip_load")               skipLoad      = true;
    }
    if (dataDirs.empty()) {
        logError("need --data_dirs=...");
        return 1;
    }

    // ─── Load dataset ───
    ExecutionPlanDataset ds(dataDirs, double(timeout));
    if (ds.size() < size_t(minSamples*2)) {
        logError("Too few samples");
        return 1;
    }

    // ─── Compute per-feature mean/std for 63 dims ───
    std::vector<double> feat_mean(NUM_FEATS, 0.0), feat_std(NUM_FEATS, 0.0);
    for (size_t i = 0; i < ds.size(); ++i)
      for (int j = 0; j < NUM_FEATS; ++j)
        feat_mean[j] += ds[i].feats[j];
    for (int j = 0; j < NUM_FEATS; ++j)
      feat_mean[j] /= double(ds.size());
    for (size_t i = 0; i < ds.size(); ++i)
      for (int j = 0; j < NUM_FEATS; ++j) {
        double d = ds[i].feats[j] - feat_mean[j];
        feat_std[j] += d*d;
      }
    for (int j = 0; j < NUM_FEATS; ++j) {
      feat_std[j] = std::sqrt(feat_std[j]/double(ds.size()));
      if (feat_std[j] < 1e-6) feat_std[j] = 1.0;
    }

    // ─── Load RandomForest ───
    RandomForest rf(nTrees, maxDepth, minSamples, minGain, sampleRatio);
    {
      json j;
      std::ifstream f(rfModelPath);
      f >> j;
      rf.from_json(j["forest"]);
      logInfo("Loaded RF → " + rfModelPath);
    }

    // ─── Load MLPReg (row‐time) ───
    MLPReg rowReg(NUM_FEATS, hiddenNeurons, 0.0);
    {
      json j; std::ifstream f(rowModelPath); f >> j;
      rowReg.from_json(j);
      logInfo("Loaded row‐MLP → " + rowModelPath);
    }

    // ─── Load GIN (col‐time) ───
    json jcol; std::ifstream fcol(colModelPath); fcol >> jcol;
    int trained_n_ops = jcol.at("n_ops").get<int>();
    GIN colReg(trained_n_ops, hiddenNeurons);
    colReg.from_json(jcol);
    logInfo("Loaded col-GIN → " + colModelPath);

    // ─── Evaluate router ───
    size_t TP=0, FP=0, TN=0, FN=0;
    double rt_row=0, rt_col=0, rt_cost=0, rt_hyb=0, rt_fann=0;
    double rt_pred=0, rt_rf_all=0, rt_rf_close=0, rt_router=0, rt_opt=0;
    size_t n=0, n_cost=0, n_hyb=0, n_fann=0, n_pred=0, n_rf_all=0, n_rf_close=0;

    for (size_t i = 0; i < ds.size(); ++i) {
      auto &s = ds[i];
      progress("eval", i+1, ds.size());
      if (s.label < 0) continue;
      ++n;

      // ground-truth averages
      rt_row += (s.row_time - rt_row)/n;
      rt_col += (s.col_time - rt_col)/n;
      rt_opt += (std::min(s.row_time, s.col_time) - rt_opt)/n;

      // cost-threshold method
      ++n_cost;
      {
        double act = (s.query_cost > 5e4) ? s.col_time : s.row_time;
        rt_cost += (act - rt_cost)/n_cost;
      }

      // original hybrid optimizer
      ++n_hyb;
      {
        double act = s.hybrid_use_imci ? s.col_time : s.row_time;
        rt_hyb += (act - rt_hyb)/n_hyb;
      }

      // FANN-based decision
      ++n_fann;
      {
        double act = s.fann_use_imci ? s.col_time : s.row_time;
        rt_fann += (act - rt_fann)/n_fann;
      }

      // normalize features
      std::vector<float> x63(NUM_FEATS);
      for (int j = 0; j < NUM_FEATS; ++j)
        x63[j] = float((s.feats[j] - feat_mean[j]) / feat_std[j]);

      // predicted-time decision
      ++n_pred;
      {
        double pr = rowReg.run(x63);
        double pc = colReg.predict_col_time(s.colGraph);
        double act = (pc < pr) ? s.col_time : s.row_time;
        rt_pred += (act - rt_pred)/n_pred;
      }

      // RF-only on all datapoints
      ++n_rf_all;
      {
        float rf_feat[NUM_FEATS];
        for (int j = 0; j < NUM_FEATS; ++j) rf_feat[j] = s.feats[j];
        bool v_rf_all = rf.predict(rf_feat) >= 0.5f;
        double act = v_rf_all ? s.col_time : s.row_time;
        rt_rf_all += (act - rt_rf_all)/n_rf_all;
      }

      // ── UPDATED ENSEMBLE LOGIC ──
      {
        double pr = rowReg.run(x63);
        double pc = colReg.predict_col_time(s.colGraph);

        // get RF score
        float rf_feat[NUM_FEATS];
        for (int j = 0; j < NUM_FEATS; ++j) rf_feat[j] = s.feats[j];
        float rf_score = rf.predict(rf_feat);

        bool choose_col;
        // if RF is uncertain (0.3–0.7), fall back to time comparison
        if (rf_score >= 0.3f && rf_score <= 0.7f) {
          choose_col = (pc < pr);
          ++n_rf_close;  // reuse this counter for “uncertain” zone
          double act = choose_col ? s.col_time : s.row_time;
          rt_rf_close += (act - rt_rf_close)/n_rf_close;
        } else {
          // otherwise trust RF
          choose_col = (rf_score >= 0.5f);
        }

        // overall router
        double act = choose_col ? s.col_time : s.row_time;
        rt_router += (act - rt_router)/n;

        // confusion
        if (s.label &&  choose_col) ++TP;
        if (!s.label &&  choose_col) ++FP;
        if (!s.label && !choose_col) ++TN;
        if ( s.label && !choose_col) ++FN;
      }
    }

    double prec = TP ? double(TP)/(TP+FP) : 0.0;
    double rec  = TP ? double(TP)/(TP+FN) : 0.0;
    double f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : 0.0;

    std::cout << "\n=== FINAL CONFUSION ===\n"
              << "TP=" << TP << " FP=" << FP
              << " TN=" << TN << " FN=" << FN << "\n"
              << "Precision=" << prec
              << " Recall=" << rec
              << " F1=" << f1 << "\n\n";

    std::cout << "=== AVG RUNTIMES (s/query) ===\n"
              << "Row only          : " << rt_row    << "\n"
              << "Column only       : " << rt_col    << "\n"
              << "Cost threshold    : " << rt_cost   << "\n"
              << "Hybrid optimizer  : " << rt_hyb    << "\n"
              << "FANN-based        : " << rt_fann   << "\n"
              << "Predicted-time    : " << rt_pred   << "\n"
              << "RF-only (all)     : " << rt_rf_all << "\n"
              << "RF-only (uncertain zone) : " << rt_rf_close << "\n"
              << "Ensemble router   : " << rt_router << "\n"
              << "Optimal oracle    : " << rt_opt    << "\n";

    double wall = std::chrono::duration_cast<std::chrono::seconds>(
                      Clock::now() - t_start
                  ).count();
    logInfo("Total wall-clock: " + std::to_string(wall) + " s");

    return 0;
}
