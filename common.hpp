/*───────────────────────────────────────────────────────────
  Portable <filesystem> pickup
  • GCC 9+ / Clang –> <filesystem>
  • GCC 5-8       –> <experimental/filesystem>  (+-lstdc++fs)
  • GCC 4.8       –> tiny fallback shim (no extra deps)
───────────────────────────────────────────────────────────*/
#pragma once
#include <bits/stdc++.h>
#include <dirent.h>     // opendir / readdir / closedir
#include <sys/stat.h>   // stat
#include <unistd.h>     // access
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>

#include "json.hpp"
#include <Eigen/Dense>
#include <regex>
#include <mysql/mysql.h>
using namespace std;

using json = nlohmann::json;

constexpr int    NUM_FEATS   = 102;          // ← unified!
constexpr double EPS_RUNTIME = 1e-6;
constexpr double COST_THR    = 5e4;

extern bool g_need_col_plans;   // 缺省为 true，兼容旧逻辑


struct Graph {
    int N = 0;
    std::vector<Eigen::Vector2d> x;          // (rows , cost )  – Z-score-ed
    std::vector<std::vector<int>> adj;       // adjacency list
    std::vector<int> op;                     // global op-id per node
};

struct Sample {
    std::array<float,NUM_FEATS> feat{{}};
    Graph  colGraph;            // <-- NEW (empty if not parsed)
    int    label   = 0;
    double row_t   = 0, col_t = 0, qcost = 0;
    int    fann_pred = -1, hybrid_pred = -1;
};


static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }


/* ———— 简易文件/目录工具 ———— */
static inline bool file_exists(const std::string& p){
    return ::access(p.c_str(), F_OK) == 0;
}
static inline bool is_directory(const std::string& p){
    struct stat sb{};
    return ::stat(p.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}
static inline bool has_ext(const std::string& f, const std::string& ext){
    return f.size() > ext.size() &&
           f.compare(f.size() - ext.size(), ext.size(), ext) == 0;
}
static inline std::string strip_ext(const std::string& f){
    auto pos = f.rfind('.');
    return (pos == std::string::npos) ? f : f.substr(0, pos);
}

static void progress(const string&tag,size_t cur,size_t tot,size_t W=40){
    double f=tot?double(cur)/tot:1.0; size_t filled=size_t(f*W);
    cerr<<"\r"<<tag<<" ["<<string(filled,'=')<<string(W-filled,' ')
        <<"] "<<setw(3)<<int(f*100)<<"% ("<<cur<<'/'<<tot<<')'<<flush;
    if(cur==tot) cerr<<'\n';
}


/* ─────────────────────────────  utilities  ────────────────────────────── */
inline double parsePossibleNumber(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string())               return std::stod(j[k].get<std::string>());
        if(j[k].is_number_float()||
           j[k].is_number_integer())       return j[k].get<double>();
    }catch(...){ logW("Invalid number for key: "+k);}
    return 0.0;
}

inline double convert_data_size_to_numeric(std::string s){
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

static double safe_f(const json&v){
    if(v.is_number()) return v.get<double>();
    if(v.is_string()){ try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
static double safe_f(const json&o,const char*k){ return o.contains(k)?safe_f(o[k]):0; }
static double log1p_clip(double v){ return log1p(max(0.0,v)); }
static double str_size_to_num(string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    double m=1; char suf=s.back();
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return stod(s)*m;}catch(...){ return 0.0;}
}
static bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    const auto&v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){ string s=v.get<string>(); transform(s.begin(),s.end(),s.begin(),::tolower);
                       return s=="yes"||s=="true"||s=="1";}
    return false;
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

               //  auto gini = [](double pos,double tot){
               //      if(tot<=0) return 1.0;
               //      double q = pos/tot;
               //      return 2.0*q*(1.0-q);
               //  };
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
   //  #pragma omp parallel for schedule(dynamic)
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
      // #pragma omp critical
      progress("RF-train", t+1, T);
    }

    auto t1 = clock::now();
    logI("Random-Forest training took " +
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

/* ─────────── tiny logistic regression (3 inputs + bias) ─────────── */
struct Logistic{
    double w[4]={0,0,0,0};          // w0=bias, w1 costFlag, w2 coverFlag, w3 RFprob
    static inline double sig(double z){ return 1.0/(1.0+exp(-z)); }
    void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
         const std::vector<int>&  y,
         const RandomForest& rf,
         int epochs = 400, double lr = 0.1)
    {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();

        for (int e = 0; e < epochs; ++e)
        {
            double g[4] = {0, 0, 0, 0};
            for (std::size_t i = 0; i < X.size(); ++i)
            {
                double x1 = X[i][41];                 // high-cost flag
                double x2 = X[i][43];                 // all-covered flag
                double x3 = rf.predict(X[i].data());  // forest prob

                double z = w[0] + w[1]*x1 + w[2]*x2 + w[3]*x3;
                double p = sig(z);
                double d = p - y[i];
                g[0] += d;
                g[1] += d * x1;
                g[2] += d * x2;
                g[3] += d * x3;
            }
            for (double &gi : g) gi /= X.size();
            for (int j = 0; j < 4; ++j) w[j] -= lr * g[j];

            if ((e + 1) % 10 == 0 || e + 1 == epochs)
                progress("LR-train", e + 1, epochs);
        }
        auto t1 = clock::now();
        logI("Logistic fine-tune took " +
            std::to_string(std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count())
            + " s");
    }

    double predict(const float*f,double rfProb)const{
        double z=w[0]+w[1]*f[41]+w[2]*f[43]+w[3]*rfProb;
        return sig(z);
    }
    json to_json()const{ return json::array({w[0],w[1],w[2],w[3]}); }
    void from_json(const json& j)
    {
        if (!j.is_array() || j.size() != 4)
            throw std::runtime_error("Logistic::from_json – malformed JSON");
        for (int i = 0; i < 4; ++i) w[i] = j[i].get<double>();
    }
};



/* ─────────── MySQL index-map (unchanged logic) ─────────── */
static unordered_map<string, unordered_set<string>> indexCols;

static void load_index_defs_from_db(const string &host,int port,
                                    const string &user,const string &pass,
                                    const string &db,const string &tbl)
{
    MYSQL *conn = mysql_init(nullptr);
    if(!mysql_real_connect(conn,host.c_str(),user.c_str(),pass.c_str(),
                           db.c_str(),port,nullptr,0))
    { logW("MySQL connect failed: "+string(mysql_error(conn))); return; }

    string q="SHOW CREATE TABLE `"+tbl+"`";
    if(mysql_query(conn,q.c_str()))
    { logW("SHOW CREATE TABLE failed: "+string(mysql_error(conn))); mysql_close(conn); return; }

    MYSQL_RES *res=mysql_store_result(conn);
    if(!res){ mysql_close(conn); return; }
    MYSQL_ROW row=mysql_fetch_row(res);
    if(!row||!row[1]){ mysql_free_result(res); mysql_close(conn); return; }
    string ddl=row[1]; mysql_free_result(res); mysql_close(conn);

    static const regex re(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))",regex::icase);
    smatch m; auto it=ddl.cbegin(), ed=ddl.cend();
    while(regex_search(it,ed,m,re)){
        string idx=m[1], colsRaw=m[2];
        unordered_set<string> S;
        string col; stringstream ss(colsRaw);
        while(getline(ss,col,',')){
            col.erase(remove(col.begin(),col.end(),'`'),col.end());
            col.erase(0,col.find_first_not_of(" \t"));
            col.erase(col.find_last_not_of(" \t")+1);
            if(!col.empty()) S.insert(col);
        }
        indexCols[idx]=std::move(S);
        it=m.suffix().first;
    }
}

static void load_all_index_defs(const string &host,int port,
                                const string &user,const string &pass,
                                const vector<string>& dbs)
{
    indexCols.clear();
    for(const auto&db:dbs){
        MYSQL *c=mysql_init(nullptr);
        if(!mysql_real_connect(c,host.c_str(),user.c_str(),pass.c_str(),
                               db.c_str(),port,nullptr,0))
        { logW("connect "+db+" failed"); mysql_close(c); continue; }

        if(mysql_query(c,"SHOW TABLES"))
        { logW("SHOW TABLES failed in "+db); mysql_close(c); continue; }
        MYSQL_RES *t=mysql_store_result(c);
        MYSQL_ROW r;
        while((r=mysql_fetch_row(t))) load_index_defs_from_db(host,port,user,pass,db,r[0]);
        mysql_free_result(t); mysql_close(c);
    }
}
/* -------------------------------------------------------------------- */


/* ─────────── 63-feature extraction ─────────── */
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0,selSum=0,selMin=1e30,selMax=0,
           ratioSum=0,ratioMax=0,maxPrefix=0,minRead=1e30,fanoutMax=0;
    int    cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,sumPK=0,
           coverCount=0,maxDepth=0;
    bool   grp=false,ord=false,tmp=false;
    double outerRows      = 0;   // first non-ALL driver cardinality
    int    eqChainDepth   = 0;   // longest consecutive eq_ref chain
    int    _curEqChain    = 0;   // internal: running counter
    double lateFanMax  = 0;   // ❶ NEW  – max fan-out seen at depth ≥ 4
    double pcDepth3 = 0;        // Σ prefix_cost of tables at depth == 3
};


/* -------------------------------------------------------------------------- */
/* Recursive traversal over the JSON plan tree                                */
/* Collects per-table statistics into an Agg struct                           */
/* -------------------------------------------------------------------------- */
static void walk(const json& n, Agg& a, int depth = 1)
{
    /* ---------- OBJECT node ------------------------------------------------ */
    if (n.is_object()) {

        /* ---- TABLE node ---- */
        if (n.contains("table") && n["table"].is_object()) {

            const auto& t  = n["table"];
            const auto& ci = t.value("cost_info", json::object());

            /* basic numerical fields */
            double re = safe_f(t , "rows_examined_per_scan");
            double rp = safe_f(t , "rows_produced_per_join");
            double fl = safe_f(t , "filtered");
            double rc = safe_f(ci, "read_cost");
            double ec = safe_f(ci, "eval_cost");
            double pc = safe_f(ci, "prefix_cost");
            double dr = ci.contains("data_read_per_join") && ci["data_read_per_join"].is_string()
                        ? str_size_to_num(ci["data_read_per_join"].get<string>())
                        : safe_f(ci, "data_read_per_join");

            /* aggregate into Agg */
            a.re += re;  a.rp += rp;  a.f  += fl;
            a.rc += rc;  a.ec += ec;  a.pc += pc;  a.dr += dr;  a.cnt++;
            if (depth == 3) a.pcDepth3 += pc;         // <── NEW

            a.maxPrefix = std::max(a.maxPrefix, pc);
            a.minRead   = std::min(a.minRead , rc);

            if (re > 0) {
                double sel = rp / re;
                a.selSum   += sel;
                a.selMin    = std::min(a.selMin, sel);
                a.selMax    = std::max(a.selMax, sel);
                a.fanoutMax = std::max(a.fanoutMax, sel);

                /* ❷ NEW: late fan-out (depth ≥ 4) */
                if (depth >= 4)
                    a.lateFanMax = std::max(a.lateFanMax, sel);
            }

            double ratio = ec > 0 ? rc / ec : rc;
            a.ratioSum  += ratio;
            a.ratioMax   = std::max(a.ratioMax, ratio);

            /* access-type counters ----------------------------------------- */
            const std::string at = t.value("access_type", "ALL");
            if      (at == "range")   a.cRange++;
            else if (at == "ref")     a.cRef++;
            else if (at == "eq_ref")  a.cEq++;
            else if (at == "index")   a.cIdx++;
            else                      a.cFull++;

            if (getBool(t, "using_index")) a.idxUse++;

            if (t.contains("possible_keys") && t["possible_keys"].is_array())
                a.sumPK += int(t["possible_keys"].size());

            /* covering-index check ----------------------------------------- */
            if (t.contains("used_columns") && t["used_columns"].is_array() &&
                t.contains("key") && t["key"].is_string())
            {
                const std::string idx = t["key"];
                auto it = indexCols.find(idx);
                if (it != indexCols.end()) {
                    bool cover = true;
                    for (const auto& u : t["used_columns"])
                        if (!u.is_string() ||
                            !it->second.count(u.get<string>()))
                        { cover = false; break; }
                    if (cover) a.coverCount++;
                }
            }

            /* ---------- NEW “row-wins” signals ---------------------------- */

            /* 1️⃣ first non-ALL access  -> outerRows */
            if (a.outerRows == 0 && at != "ALL")
                a.outerRows = re;

            /* 2️⃣ consecutive eq_ref chain depth */
            if (at == "eq_ref") {
                a._curEqChain++;
                a.eqChainDepth = std::max(a.eqChainDepth, a._curEqChain);
            } else {
                a._curEqChain = 0;          // break the chain
            }
        } /* end TABLE node */

        /* plan-level flags --------------------------------------------------- */
        if (n.contains("grouping_operation"))                                  a.grp = true;
        if (n.contains("ordering_operation") || getBool(n, "using_filesort"))  a.ord = true;
        if (getBool(n, "using_temporary_table"))                               a.tmp = true;

        /* recurse into children (skip “table” key we already handled) */
        for (const auto& kv : n.items())
            if (kv.key() != "table")
                walk(kv.value(), a, depth + 1);
    }

    /* ---------- ARRAY node ------------------------------------------------- */
    else if (n.is_array()) {
        for (const auto& v : n) walk(v, a, depth);
    }

    /* update maximum nesting depth */
    a.maxDepth = std::max(a.maxDepth, depth);
}



/* ---------- 取消 clip 的对数函数 ---------- */
static inline double lp(double v) { return std::log1p(std::max(0.0, v)); }
/* ----------------------------------------------------------- */
/*  Col-histogram helpers – very light metadata cache          */
/*  (stub implementation: fill `colStats` from your catalog)   */
/* ----------------------------------------------------------- */
/* ===================================================================== */
/*  Column statistics loader (stand-alone, no extra header needed)       */
/* ===================================================================== */
enum ColDType : uint8_t { COL_INT=0, COL_FLOAT, COL_STRING,
                          COL_DATETIME, COL_BOOL, COL_DTYPE_N };

struct ColStats {
    double   avg_width = 8;        // bytes
    double   ndv       = 1000;     // distinct values
    ColDType dtype     = COL_INT;
    double rows_c = 0, rows_s = 1, cost_c = 0, cost_s = 1;
    std::unordered_map<std::string,int> op2id;  // local operator IDs
};

static std::unordered_map<std::string, ColStats> colStats;



/* ---------- helpers -------------------------------------------------- */
static ColDType map_dtype(std::string t) {
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    if (t.find("int")      != std::string::npos) return COL_INT;
    if (t == "float" || t == "double" ||
        t.find("decimal")  != std::string::npos) return COL_FLOAT;
    if (t.find("char")     != std::string::npos ||
        t.find("text")     != std::string::npos ||
        t == "json")                                return COL_STRING;
    if (t == "date" || t == "datetime" ||
        t == "timestamp" || t == "time")            return COL_DATETIME;
    if (t == "bool" || t == "boolean")              return COL_BOOL;
    return COL_STRING;
}

static double ndv_from_histogram(const std::string& hist_json)
{
    try {
        auto j = nlohmann::json::parse(hist_json);
        double ndv = 0.0;
        for (const auto& b : j.at("buckets"))
            ndv += b.value("distinct-range", 0.0);
        return ndv > 0 ? ndv : -1;
    } catch (...) { return -1; }
}

static double ndv_from_statistics(MYSQL* conn,
                                  const std::string& db,
                                  const std::string& tbl,
                                  const std::string& col)
{
    std::string q =
        "SELECT CARDINALITY "
        "FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA='" + db  + "' AND "
              "TABLE_NAME='"   + tbl + "' AND "
              "COLUMN_NAME='"  + col + "' "
        "ORDER BY SEQ_IN_INDEX LIMIT 1";

    if (mysql_query(conn, q.c_str()) != 0) return -1;
    MYSQL_RES* r = mysql_store_result(conn);
    if (!r) return -1;
    MYSQL_ROW row = mysql_fetch_row(r);
    double v = (row && row[0]) ? atof(row[0]) : -1;
    mysql_free_result(r);
    return v;
}

/* --------------------------------------------------------------------- */
/*  Call once at startup: populate_col_stats(host,port,user,pass,dbs)    */
/* --------------------------------------------------------------------- */
static bool populate_col_stats(const std::string& host, int port,
                               const std::string& user, const std::string& pass,
                               const std::vector<std::string>& dbs)
{
    colStats.clear();

    MYSQL* conn = mysql_init(nullptr);
    mysql_options(conn, MYSQL_OPT_RECONNECT, "1");
    if (!mysql_real_connect(conn, host.c_str(), user.c_str(), pass.c_str(),
                            nullptr, port, nullptr, 0))
    {
        std::cerr << "[ERR]   MySQL connect failed: "
                  << mysql_error(conn) << '\n';
        return false;
    }

    /* Build IN (…) list safely */
    std::string in;
    for (size_t i = 0; i < dbs.size(); ++i) {
        in += "'" + dbs[i] + "'";
        if (i + 1 < dbs.size()) in += ",";
    }

    // std::string q =
    //     "SELECT C.TABLE_SCHEMA, C.TABLE_NAME, C.COLUMN_NAME, "
    //     "       C.DATA_TYPE, "
    //     "       COALESCE(C.CHARACTER_OCTET_LENGTH, "
    //     "                CEILING(C.NUMERIC_PRECISION/8), 8) AS AVG_LEN, "
    //     "       CS.HISTOGRAM_JSON "
    //     "FROM information_schema.COLUMNS AS C "
    //     "LEFT JOIN information_schema.COLUMN_STATISTICS AS CS "
    //     "  ON CS.SCHEMA_NAME = C.TABLE_SCHEMA "
    //     " AND CS.TABLE_NAME  = C.TABLE_NAME "
    //     " AND CS.COLUMN_NAME = C.COLUMN_NAME "
    //     "WHERE C.TABLE_SCHEMA IN (" + in + ")";
    std::string q =
        "SELECT C.TABLE_SCHEMA, C.TABLE_NAME, C.COLUMN_NAME, "
        "       C.DATA_TYPE, "
        "       COALESCE(C.CHARACTER_OCTET_LENGTH, "
        "                CEILING(C.NUMERIC_PRECISION/8), 8) AS AVG_LEN, "
        "       CS.HISTOGRAM "
        "FROM information_schema.COLUMNS AS C "
        "LEFT JOIN information_schema.COLUMN_STATISTICS AS CS "
        "  ON CS.SCHEMA_NAME = C.TABLE_SCHEMA "
        " AND CS.TABLE_NAME  = C.TABLE_NAME "
        " AND CS.COLUMN_NAME = C.COLUMN_NAME "
        "WHERE C.TABLE_SCHEMA IN (" + in + ")";

    if (mysql_query(conn, q.c_str()) != 0) {
        std::cerr << "[ERR]   Query failed: " << mysql_error(conn) << '\n';
        mysql_close(conn); return false;
    }

    MYSQL_RES* res = mysql_store_result(conn);
    if (!res) { mysql_close(conn); return false; }

    size_t rows = 0;
    while (MYSQL_ROW row = mysql_fetch_row(res)) {
        std::string db    = row[0];
        std::string tbl   = row[1];
        std::string col   = row[2];
        std::string dtype = row[3];
        double      avg   = row[4] ? atof(row[4]) : 8;
        std::string hist  = row[5] ? row[5] : "";

        ColStats cs;
        cs.avg_width = std::max(1.0, avg);
        cs.dtype     = map_dtype(dtype);

        double ndv = hist.empty() ? -1 : ndv_from_histogram(hist);
        if (ndv < 0) {
            ndv = ndv_from_statistics(conn, db, tbl, col);
            if (ndv < 0) ndv = 1000;                 // sensible default
        }
        cs.ndv = ndv;

        colStats[db + "." + tbl + "." + col] = cs;
        ++rows;
    }
    mysql_free_result(res);
    mysql_close(conn);

    std::cerr << "[INFO]  Loaded column stats for " << rows << " columns\n";
    return true;
}
/* ===================================================================== */
/*  End of column statistics loader                                      */
/* ===================================================================== */

/* ------------------------------------------------------------------ */
/*  Fast lookup with default – returns dummy ColStats if not present  */
/* ------------------------------------------------------------------ */
/* ---------- fast column-stat lookup ---------- */
static inline ColStats lookup_col_stats(const std::string& id)
{
    auto it = colStats.find(id);
    return (it == colStats.end()) ? ColStats{} : it->second;
}

/* ---------- C++11 normalise() for std::array ---------- */
template <size_t N>
static inline void normalise(std::array<float, N>& hist)
{
    float sum = 0.f;
    for (float v : hist) sum += v;
    if (sum < 1.f) sum = 1.f;
    for (float& v : hist) v /= sum;
}


/* ────────────────────────────────────────────────────────────── */
/*  Light table / index meta cached at start-up                  */
/* ────────────────────────────────────────────────────────────── */
struct TblStats {
    double rows        = 0;    // INFORMATION_SCHEMA.TABLES.TABLE_ROWS
    double data_bytes  = 0;    // DATA_LENGTH
    double idx_bytes   = 0;    // INDEX_LENGTH
    double frag_ratio  = 0;    // DATA_FREE / (DATA_LENGTH+1)
    double upd_pct     = 0;    // rows_updated / (rows_read+1)   (PS)
    int    partitions  = 1;    // # partitions (≥1)
    int    idx_cnt     = 0;    // total indexes
    int    uniq_cnt    = 0;    // UNIQUE indexes
    int    pk_len      = 0;    // ⅀ PK column byte width
    int    cover_cols  = 0;    // # columns covered by any index
    int    total_cols  = 0;    // total columns
    bool   compressed  = false;
};
static unordered_map<string,TblStats> tblStats;     // key = db.tbl


/* --------------------------------------------------------------------- */
/*  Call once at start-up: fill `tblStats` from INFORMATION_SCHEMA + PS  */
/*  – now robust against empty result-sets and NULL fields               */
/* --------------------------------------------------------------------- */
static bool populate_tbl_stats(const std::string &host, int port,
                               const std::string &user,
                               const std::string &pass,
                               const std::vector<std::string> &dbs)
{
    tblStats.clear();

    MYSQL *conn = mysql_init(nullptr);
    mysql_options(conn, MYSQL_OPT_RECONNECT, "1");

    if (!mysql_real_connect(conn, host.c_str(), user.c_str(), pass.c_str(),
                            /* db = */ nullptr, port, nullptr, 0))
    {
        logE(std::string("MySQL connect failed: ") + mysql_error(conn));
        return false;
    }

    /* ------------------------------------------------------------------ */
    /* Build a safe “… IN ('db1','db2',…)” list                           */
    /* ------------------------------------------------------------------ */
    std::string in;
    for (size_t i = 0; i < dbs.size(); ++i) {
        in += '\'' + dbs[i] + '\'';
        if (i + 1 < dbs.size()) in += ',';
    }

    MYSQL_RES *res = nullptr;
    MYSQL_ROW  row;

    /* ========================== 1. TABLES ============================ */
    std::string q =
        "SELECT TABLE_SCHEMA,TABLE_NAME,TABLE_ROWS,DATA_LENGTH,INDEX_LENGTH,"
        "DATA_FREE,ROW_FORMAT "
        "FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA IN(" + in + ")";
    if (mysql_query(conn, q.c_str()) == 0 && (res = mysql_store_result(conn)))
    {
        while ((row = mysql_fetch_row(res)))
        {
            if (!row[0] || !row[1]) continue;          /* paranoia */

            std::string key = std::string(row[0]) + '.' + row[1];
            TblStats &s     = tblStats[key];

            s.rows       = row[2] ? atof(row[2]) : 0;
            s.data_bytes = row[3] ? atof(row[3]) : 0;
            s.idx_bytes  = row[4] ? atof(row[4]) : 0;

            double free_b = row[5] ? atof(row[5]) : 0;
            s.frag_ratio  = (s.data_bytes > 0 ? free_b / (s.data_bytes + 1) : 0);

            std::string fmt = row[6] ? row[6] : "";
            std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
            s.compressed = fmt.find("compress") != std::string::npos;
        }
        mysql_free_result(res);
    }

    /* ========================= 2. PARTITIONS ========================= */
    q = "SELECT TABLE_SCHEMA,TABLE_NAME,COUNT(*) "
        "FROM information_schema.PARTITIONS "
        "WHERE TABLE_SCHEMA IN(" + in + ") GROUP BY 1,2";
    if (mysql_query(conn, q.c_str()) == 0 && (res = mysql_store_result(conn)))
    {
        while ((row = mysql_fetch_row(res)))
        {
            if (!row[0] || !row[1] || !row[2]) continue;
            tblStats[std::string(row[0]) + '.' + row[1]].partitions =
                std::max(1, atoi(row[2]));
        }
        mysql_free_result(res);
    }

    /* ========================= 3. STATISTICS ========================= */
    q = "SELECT TABLE_SCHEMA,TABLE_NAME,INDEX_NAME,NON_UNIQUE,"
        "COLUMN_NAME,SUB_PART "
        "FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA IN(" + in + ")";
    if (mysql_query(conn, q.c_str()) == 0 && (res = mysql_store_result(conn)))
    {
        while ((row = mysql_fetch_row(res)))
        {
            if (!row[0] || !row[1] || !row[2]) continue;

            TblStats &s = tblStats[std::string(row[0]) + '.' + row[1]];
            ++s.idx_cnt;
            if (strcmp(row[3], "0") == 0) ++s.uniq_cnt;

            /* total / covered column counts -------------------------- */
            ++s.total_cols;
            if (row[5] && strcmp(row[5], "0") == 0) ++s.cover_cols;

            /* crude PK length estimate (first column of PRIMARY) ----- */
            if (row[2] && strcmp(row[2], "PRIMARY") == 0 &&
                row[5] && strcmp(row[5], "0") == 0)
                s.pk_len += 8;                       // assume 8-byte key part
        }
        mysql_free_result(res);
    }

    /* ===== 4. UPDATE ratio from Performance Schema (may not exist) == */
    q = "SELECT OBJECT_SCHEMA,OBJECT_NAME,ROWS_UPDATED,ROWS_READ "
        "FROM performance_schema.table_io_waits_summary_by_table "
        "WHERE OBJECT_SCHEMA IN(" + in + ")";
    if (mysql_query(conn, q.c_str()) == 0 && (res = mysql_store_result(conn)))
    {
        while ((row = mysql_fetch_row(res)))
        {
            if (!row[0] || !row[1]) continue;
            TblStats &s = tblStats[std::string(row[0]) + '.' + row[1]];
            double upd  = row[2] ? atof(row[2]) : 0;
            double rd   = row[3] ? atof(row[3]) : 0;
            s.upd_pct   = upd / (rd + 1);            // avoid ÷0
        }
        mysql_free_result(res);
    }

    mysql_close(conn);

    logI("Loaded table stats for " + std::to_string(tblStats.size()) + " tables");
    return true;
}


static inline const TblStats& lookup_tbl(const string& id){
    static const TblStats dflt{};
    auto it=tblStats.find(id);
    return (it==tblStats.end())?dflt:it->second;
}

/* ----------------------------------------------------------- */
/*  plan → 95-dim feature (65 legacy + 18 IS/PS + 12 histos)   */
/* ----------------------------------------------------------- */
static bool plan2feat(const json &plan, float f[NUM_FEATS])
{
    if (!plan.contains("query_block")) return false;
    const json *qb = &plan["query_block"];

    /* UNION  → use first branch */
    if (qb->contains("union_result")) {
        const auto &specs = (*qb)["union_result"]["query_specifications"];
        if (specs.is_array() && !specs.empty())
            qb = &specs[0]["query_block"];
    }

    /* ---------- ❶ OLD 65-dim aggregation ---------- */
    Agg a; walk(*qb, a);
    if (!a.cnt) return false;

    const double inv = 1.0 / a.cnt;
    int   k      = 0;
    double qCost = safe_f(qb->value("cost_info", json::object()), "query_cost");
    double rootRow = safe_f(*qb, "rows_produced_per_join");

#define PUSH(x) f[k++] = static_cast<float>(x)

    /* --- 0‒6 basic costs / rows ------------------------------------ */
    PUSH(lp(a.re * inv)); PUSH(lp(a.rp * inv)); PUSH(lp(a.f * inv));
    PUSH(lp(a.rc * inv)); PUSH(lp(a.ec * inv)); PUSH(lp(a.pc * inv));
    PUSH(lp(a.dr * inv));

    /* --- 7‒12 access-type counters (ratio) ------------------------- */
    PUSH(a.cRange * inv); PUSH(a.cRef * inv); PUSH(a.cEq * inv);
    PUSH(a.cIdx   * inv); PUSH(a.cFull* inv); PUSH(a.idxUse* inv);

    /* --- 13‒17 selectivity / shape -------------------------------- */
    PUSH(a.selSum * inv); PUSH(a.selMin); PUSH(a.selMax);
    PUSH(a.maxDepth);     PUSH(a.fanoutMax);

    /* --- 18‒20 plan-level flags ----------------------------------- */
    PUSH(a.grp); PUSH(a.ord); PUSH(a.tmp);

    /* --- 21‒22 rc/ec ratios --------------------------------------- */
    PUSH(a.ratioSum * inv); PUSH(a.ratioMax);

    /* --- 23‒24 root cost & rows ----------------------------------- */
    PUSH(lp(qCost)); PUSH(lp(rootRow));

    /* --- 25‒27 derived cost ratios -------------------------------- */
    PUSH(lp((a.pc * inv) / std::max(1e-6, a.rc * inv)));
    PUSH(lp((a.rc * inv) / std::max(1e-6, a.re * inv)));
    PUSH(lp((a.ec * inv) / std::max(1e-6, a.re * inv)));

    /* --- 28‒31 misc counters -------------------------------------- */
    PUSH(a.cnt == 1); PUSH(a.cnt > 1);
    PUSH(lp(a.maxDepth * (a.idxUse * inv)));
    PUSH(lp((a.idxUse * inv) / std::max(a.cFull * inv, 1e-3)));

    /* --- 32‒39 other stats ---------------------------------------- */
    PUSH(a.cnt);
    PUSH(a.cnt ? double(a.sumPK) / a.cnt : 0);
    PUSH(lp(a.maxPrefix));
    PUSH(lp(a.minRead < 1e30 ? a.minRead : 0));
    PUSH(a.cnt > 1 ? double(a.cnt - 1) / a.cnt : 0);
    PUSH(rootRow > 0 ? double(a.re) / rootRow : 0);
    PUSH(a.selMax - a.selMin);
    PUSH(a.idxUse / double(std::max(1, a.cRange + a.cRef +
                                       a.cEq   + a.cIdx)));

    /* --- 40‒43 covering-index & big-cost flags -------------------- */
    PUSH(qCost);
    PUSH(qCost > 5e4);
    PUSH(a.cnt ? double(a.coverCount) / a.cnt : 0);
    PUSH(a.coverCount == a.cnt);

    /* --- 44‒46 log-diffs & counts -------------------------------- */
    PUSH(lp(a.re * inv) - lp(a.selSum * inv));
    PUSH(a.cnt);
    PUSH(lp(a.cnt));

    /* --- 47‒50 PK / cover counters ------------------------------- */
    PUSH(a.sumPK);
    PUSH(a.cnt ? double(a.sumPK) / a.cnt : 0);
    PUSH(a.coverCount);
    PUSH(a.cnt ? double(a.coverCount) / a.cnt : 0);

    /* --- 51‒56 repeated access shares ----------------------------- */
    PUSH(a.idxUse * inv); PUSH(a.cRange * inv); PUSH(a.cRef * inv);
    PUSH(a.cEq   * inv);  PUSH(a.cIdx  * inv);  PUSH(a.cFull * inv);

    /* --- 57‒59 prefix / read extremes ----------------------------- */
    PUSH(lp(a.maxPrefix * inv));
    PUSH(lp(a.minRead < 1e30 ? a.minRead : 0));
    PUSH(a.selMax - a.selMin);

    /* --- 60‒62 extremes & ratios ---------------------------------- */
    PUSH(a.ratioMax);
    PUSH(a.fanoutMax);
    PUSH(a.selMin > 0 ? double(a.selMax / a.selMin) : 0);

    /* --- 63‒64 row-wins signals ----------------------------------- */
    PUSH(lp(a.outerRows));
    PUSH(a.eqChainDepth);

    /* ❸  NEW 65 : late-fan-out  -------------------------- */
    double late_f = 0.0;
    if (a.lateFanMax > a.selMin + 1e-9) {           // make sure we have a signal
        double ratio = a.lateFanMax / std::max(1e-6, a.selMin);
        /* soft-cap :  ratio ≈ e⁴  =>  late_f = 1.0  */
        late_f = std::min(1.0, std::log(ratio) / 4.0);
    }
    PUSH(late_f);     // k == old_k + 1   (remember for asserts)

    /* ------------------------------------------------------------------ */
    /* ❷ NEW 18-dim Information-Schema / P-Schema table-level meta        */
    /* ------------------------------------------------------------------ */

    /* Gather touched tables ---------------------------------------- */
    unordered_set<string> touched_tbls;  // "db.tbl"
    function<void(const json&)> collect_tbl = [&](const json &n){
        if (n.is_object()) {
            if (n.contains("table") && n["table"].is_object()) {
                string tbl = n["table"].value("table_name", "");
                string db  = "";              // add if db present in plan
                touched_tbls.insert(db + "." + tbl);
            }
            for (const auto &kv : n.items()) collect_tbl(kv.value());
        } else if (n.is_array())
            for (const auto &v : n) collect_tbl(v);
    };
    collect_tbl(*qb);

    /* Aggregate ---------------------------------------------------- */
    double rows_avg=0,rows_max=0,data_mb_avg=0,idx_mb_avg=0,
           frag_max=0,part_avg=0,upd_max=0,
           idx_cnt_avg=0,uniq_ratio_avg=0,cover_ratio_avg=0,
           pk_len_log_avg=0;
    bool   compressed_any=false;
    const int tbl_n = touched_tbls.size();

    for (const auto &id : touched_tbls) {
        const TblStats &s = lookup_tbl(id);
        rows_avg      += s.rows;
        rows_max       = max(rows_max, s.rows);
        data_mb_avg   += s.data_bytes/1e6;
        idx_mb_avg    += s.idx_bytes /1e6;
        frag_max       = max(frag_max, s.frag_ratio);
        part_avg      += s.partitions;
        upd_max        = max(upd_max , s.upd_pct);
        idx_cnt_avg   += s.idx_cnt;
        uniq_ratio_avg += s.idx_cnt ? double(s.uniq_cnt)/s.idx_cnt : 0;
        cover_ratio_avg+= s.total_cols? double(s.cover_cols)/s.total_cols : 0;
        pk_len_log_avg += log1p_clip(s.pk_len);
        compressed_any|= s.compressed;
    }
    if (tbl_n) {
        rows_avg      /= tbl_n;
        data_mb_avg   /= tbl_n;
        idx_mb_avg    /= tbl_n;
        part_avg      /= tbl_n;
        idx_cnt_avg   /= tbl_n;
        uniq_ratio_avg/= tbl_n;
        cover_ratio_avg/=tbl_n;
        pk_len_log_avg/= tbl_n;
    }

    /* 65-82 push --------------------------------------------------- */
    PUSH(lp(rows_avg));           // 65
    PUSH(lp(rows_max));           // 66
    PUSH(lp(data_mb_avg));        // 67
    PUSH(lp(idx_mb_avg));         // 68
    PUSH(frag_max);               // 69
    PUSH(part_avg);               // 70
    PUSH(upd_max);                // 71
    PUSH(idx_cnt_avg / 16.0);     // 72   (light normalisation)
    PUSH(uniq_ratio_avg);         // 73
    PUSH(cover_ratio_avg);        // 74
    PUSH(pk_len_log_avg);         // 75
    PUSH(0); PUSH(0); PUSH(0);    // 76-78 reserved / stub
    PUSH(compressed_any);         // 79
    PUSH(std::thread::hardware_concurrency()/64.0); // 80
    PUSH(0);                      // 81  buffer-pool hit% (optionally filled)
    PUSH(0);                      // 82  IMCI hit% (optionally filled)

    /* ------------------------------------------------------------------ */
    /* ❸ Column-histogram features (width / NDV / dtype)  ─ 12 dims        */
    /* ------------------------------------------------------------------ */
    unordered_set<string> touched_cols;     // "db.tbl.col"
    function<void(const json&)> collect_col = [&](const json &n){
        if (n.is_object()) {
            if (n.contains("table") && n["table"].is_object()) {
                const auto &t = n["table"];
                if (t.contains("used_columns") && t["used_columns"].is_array()){
                    string tbl = t.value("table_name", "");
                    string db  = "";                // add DB if available
                    for (const auto &c : t["used_columns"])
                        if (c.is_string())
                            touched_cols.insert(db+"."+tbl+"."+c.get<string>());
                }
            }
            for (const auto &kv : n.items()) collect_col(kv.value());
        } else if (n.is_array())
            for (const auto &v : n) collect_col(v);
    };
    collect_col(*qb);

    array<float,4> width_hist{0,0,0,0};
    array<float,3> ndv_hist  {0,0,0};
    array<float,5> type_hist {0,0,0,0,0};

    for (const auto &id : touched_cols) {
        ColStats s = lookup_col_stats(id);

        if      (s.avg_width<=4)   width_hist[0]+=1;
        else if (s.avg_width<=16)  width_hist[1]+=1;
        else if (s.avg_width<=64)  width_hist[2]+=1;
        else                       width_hist[3]+=1;

        if      (s.ndv<=1e3)       ndv_hist[0]+=1;
        else if (s.ndv<=1e5)       ndv_hist[1]+=1;
        else                       ndv_hist[2]+=1;

        if (s.dtype<COL_DTYPE_N)   type_hist[s.dtype]+=1;
    }
    normalise(width_hist); normalise(ndv_hist); normalise(type_hist);

    for (float v:width_hist) PUSH(v);
    for (float v:ndv_hist)   PUSH(v);
    for (float v:type_hist)  PUSH(v);

    /* ❹ NEW 96 : depth-3 prefix-cost ratio ---------------------- */
    double pc_d3 = std::max(1e-6, a.pcDepth3);
    double pc_ratio_d3 = lp( (a.pc > 0 ? a.pc : 1e-6) / pc_d3 );   // ln(pc_total/pc_depth3)
    PUSH(pc_ratio_d3);            // k += 1   → 96-th index


    double cum_fan = 0, amp_step = 0;
    if (a.outerRows > 0 && rootRow > a.outerRows * 1.0001) {
        cum_fan = std::log(rootRow / a.outerRows);
        amp_step = cum_fan / std::max(1, a.maxDepth - 1);
    }
    PUSH(cum_fan);   // 97
    PUSH(amp_step);  // 98
    double rows_max_raw   = rows_max;                 // 不取 log
    double data_mb_total  = data_mb_avg * tbl_n;      // 粗略总数据量
    double qcost_per_row  = qCost / std::max(1.0, rows_max_raw);

    PUSH(rows_max_raw);   // 99
    PUSH(data_mb_total);  // 100
    PUSH(qcost_per_row);  // 101

#undef PUSH
    return k == NUM_FEATS;        // k should be 95 now
}

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



static ColStats buildColStats(const std::string& dir) {
    ColStats st;
    std::string cache = dir + "/column_plan_statistics.json";
    if (file_exists(cache)) {
        try {
            json j; std::ifstream(cache) >> j;
            if (j.contains("op2id") && j["op2id"].size() > 0) {
                st.rows_c = j["rows_c"]; st.rows_s = j["rows_s"];
                st.cost_c = j["cost_c"]; st.cost_s = j["cost_s"];
                for (auto& kv : j["op2id"].items())
                    st.op2id[kv.key()] = kv.value();
                return st;
            } else {
                logW("Empty op2id in cache, rebuilding stats: " + cache);
            }
        } catch (...) {
            logW("Bad stats cache, rebuilding: " + cache);
        }
    }
    // scan column plans to compute medians and IQRs + collect op names
    logI("Scanning column plans in " + dir);
    // first collect files
    std::vector<std::string> files;
    std::string cdir = dir + "/column_plans";
    DIR* dp = opendir(cdir.c_str());
    if (!dp) { logW("No directory " + cdir); return st; }
    struct dirent* de;
    while ((de = readdir(dp))) {
        std::string fn = de->d_name;
        if (fn.size()>5 && fn.substr(fn.size()-5)==".json")
            files.push_back(fn);
    }
    closedir(dp);
    size_t total = files.size();
    logI("Found " + std::to_string(total) + " column plan files");

    std::vector<double> rows, cost;
    std::set<std::string> ops;
    // show progress bar while scanning
    logI("|- scanning column-plan dir " + dir);
    for (size_t i = 0; i < total; ++i) {
        progress("col-scan", i+1, total);
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


static std::unordered_map<std::string,ColStats> STATS_CACHE;

/* 1. 目录级 col-stats 取函数  ─────────────────────────────── */
inline ColStats& get_col_stats_for_dir(const std::string& dir)
{
    auto it = STATS_CACHE.find(dir);
    if (it == STATS_CACHE.end()) {
        /* 第一次遇到就扫描并缓存 */
        ColStats cs = buildColStats(dir.substr(0, dir.find_last_of('/'))); // 传上层数据集目录
        it = STATS_CACHE.emplace(dir, std::move(cs)).first;
    }
    return it->second;
}

/* 2. 全局 operator → id  map  ─────────────────────────────── */
static std::unordered_map<std::string,int> GLOBAL_OP2ID    = { {"UNK",0} };

inline std::unordered_map<std::string,int>& global_op2id()
{
    return GLOBAL_OP2ID;
}

/* 额外加一个缺省参数 bool need_col = g_need_col_plans */
inline bool
load_plan_file(const std::string& fp_row,
               const std::string& qid,
               const std::string& rowDir,
               const std::string& colDir,
               float             feat[NUM_FEATS],
               double&           qcost,
               Graph&            colGraph,
               bool              need_col = g_need_col_plans)     // ← 新增
{
    /* ----------- row plan：保持不动 ----------- */
    if (!file_exists(fp_row)) return false;
    std::ifstream in(fp_row);
    nlohmann::json j;  try { in >> j; } catch (...) { return false; }

    if (!plan2feat(j, feat)) return false;
    qcost = j.contains("query_block")
              ? safe_f(j["query_block"].value("cost_info", json::object()),
                       "query_cost")
              : 0.0;

    /* ----------- column plan：按需解析 -------- */
    if (need_col) {
        std::string fp_col = colDir + "/" + qid + ".json";
        if (file_exists(fp_col)) {
            try {
                std::ifstream ic(fp_col);  nlohmann::json cj;  ic >> cj;
                const ColStats& st = get_col_stats_for_dir(colDir);
                const auto&     id = global_op2id();
                colGraph = parseColPlan(cj, st, id);
            } catch (...) { /* bad column plan ⇒ 忽略 */ }
        }
    }
    return true;
}


struct Split {
    vector<string> train_dirs;   // 用于 5-fold CV
    vector<string> test_dirs;    // 独立 hold-out
};

/* 给定全部数据集目录 & 测试集比例（默认 0.2） */
static Split split_dirs(vector<string> dirs,
                        double test_frac = 0.2,
                        uint32_t seed    = 42)
{
    std::mt19937 gen(seed);
    std::shuffle(dirs.begin(), dirs.end(), gen);

    size_t n_test = size_t(dirs.size() * test_frac + 0.5);
    Split sp;
    sp.test_dirs  .assign(dirs.begin(),                dirs.begin()+n_test);
    sp.train_dirs .assign(dirs.begin()+n_test,         dirs.end());
    return sp;
}

/* ───── cv5.h ───── */
struct Fold {
    vector<string> tr_dirs;   // train 子集
    vector<string> val_dirs;  // val  子集
};

static vector<Fold> make_cv5(const vector<string>& dirs,
                             uint32_t seed = 42)
{
    const int K = 5;
    vector<string> shuffled = dirs;
    std::mt19937 gen(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), gen);

    vector<Fold> folds(K);
    for (size_t i = 0; i < shuffled.size(); ++i) {
        folds[i % K].val_dirs.push_back(shuffled[i]);
    }
    /* train = 全部 - val */
    for (auto& f : folds)
        for (auto& d : shuffled)
            if (!count(f.val_dirs.begin(), f.val_dirs.end(), d))
                f.tr_dirs.push_back(d);
    return folds;
}



/* ───── new: 全量一次性加载 ───── */
using DirSamples = unordered_map<string, vector<Sample>>;


static DirSamples
load_all_datasets(const std::string& base,
                  const std::vector<std::string>& dirs)
{
    DirSamples M;

    for (const auto& d : dirs)
    {
        /* ---------- per-dataset paths ---------- */
        std::string csv        = base + "/" + d + "/query_costs.csv";
        std::string rowPlanDir = base + "/" + d + "/row_plans";
        std::string colPlanDir = base + "/" + d + "/column_plans";

        if (!file_exists(csv) || !is_directory(rowPlanDir)) {
            logW("skip " + d + "  (missing csv or row_plans dir)");
            continue;
        }

        /* ========== ★★ 1. 预先构建/读取本目录的列计划统计 ★★ ========== */
        const ColStats& cst = get_col_stats_for_dir(colPlanDir);

        /* ========== ★★ 2. 合并本目录的 op2id → 全球 map ★★ ========== */
        for (const auto& kv : cst.op2id)        // kv = {opName,id}
            GLOBAL_OP2ID.emplace(kv);           // 已存在的不覆盖
        

        /* ---------- read meta (csv) ------------ */
        struct Meta { int lab; double rt, ct; int fann, hybrid; };
        std::unordered_map<std::string, Meta> meta;

        {
            std::ifstream fin(csv);
            std::string line; std::getline(fin, line);          // header
            while (std::getline(fin, line)) {
                std::stringstream ss(line);
                std::string qid, lab, rt, ct, fann, hyb;
                std::getline(ss, qid , ',');
                std::getline(ss, lab , ',');
                std::getline(ss, rt  , ',');
                std::getline(ss, ct  , ',');
                std::getline(ss, hyb , ',');
                std::getline(ss, fann, ',');

                meta[qid] = {
                    lab  == "1",
                    rt.empty()   ? 0 : std::stod(rt),
                    ct.empty()   ? 0 : std::stod(ct),
                    fann.empty() ? -1 : std::stoi(fann),
                    hyb .empty() ? -1 : std::stoi(hyb)
                };
            }
        }

        /* ---------- collect *.json in row_plans ---------- */
        std::vector<std::string> rowFiles;
        if (DIR* dirp = ::opendir(rowPlanDir.c_str())) {
            while (dirent* dp = ::readdir(dirp)) {
                if (dp->d_type == DT_REG && has_ext(dp->d_name, ".json"))
                    rowFiles.emplace_back(dp->d_name);      // keep basename only
            }
            ::closedir(dirp);
        }

        /* ---------- parse plans -------------------------- */
        std::vector<Sample> samples;
        samples.reserve(rowFiles.size());

        size_t cur = 0, tot = rowFiles.size();
        logI("|- collecting row-plan list ...");
        for (const auto& baseName : rowFiles) {
            if (++cur % 500 == 0) progress("row-scan "+d, cur, tot);

            std::string qid = strip_ext(baseName);
            auto itm = meta.find(qid);
            if (itm == meta.end()) continue;                // no meta row

            Sample s;
            /* NB: fp_row is full path now */
            std::string fp_row = rowPlanDir + "/" + baseName;
            if (!load_plan_file(fp_row, qid, rowPlanDir, colPlanDir,
                                s.feat.data(), s.qcost, s.colGraph))
                continue;

            /* fill meta columns */
            s.label        = itm->second.lab;
            s.label  = int(itm->second.ct <= itm->second.rt);
            s.row_t        = itm->second.rt;
            s.col_t        = itm->second.ct;
            s.fann_pred    = itm->second.fann;
            s.hybrid_pred  = itm->second.hybrid;

            samples.emplace_back(std::move(s));
        }

        cerr << '\n';
        logI("dir " + d + " → " + std::to_string(samples.size()) + " samples");
        M.emplace(d, std::move(samples));
    }
    return M;
}


/* —— helper: 把若干目录样本合并 —— */
static vector<Sample> build_subset(const vector<string>& dirs,
                                   const DirSamples& all)
{
    vector<Sample> out;
    for (const auto& d : dirs) {
        auto it = all.find(d);
        if (it != all.end())
            out.insert(out.end(), it->second.begin(), it->second.end());
    }
    return out;
}

static vector<string> pick_test3(vector<string> dirs, uint32_t seed=42)
{
    std::mt19937 gen(seed);
    std::shuffle(dirs.begin(), dirs.end(), gen);
    if (dirs.size() < 3) { logE("need ≥3 dirs"); exit(1); }
    return {dirs.begin(), dirs.begin()+3};
}


/* ─────────────────────────────────────────────────────────────
 *  report_and_evaluate
 *  — 统计 7 种策略 (含 Oracle) 的混淆矩阵与平均运行时
 *    并写入 CSV + 打印控制台表格
 * ───────────────────────────────────────────────────────────── */
inline void report_metrics(const std::vector<int>& pred_lgb,
                         const std::vector<Sample>& DS_test)
{
    struct Stat {
        long TP=0,FP=0,TN=0,FN=0; double rt_sum=0;
        void add(bool p,bool g,double rt,double ct){
            (p? (g?++TP:++FP):(g?++FN:++TN));
            rt_sum += (p?ct:rt);
        }
        double acc(long n) const { return double(TP+TN)/n; }
        double rec()       const { return (TP+FN)? double(TP)/(TP+FN):0; }
        double f1()        const { return TP? 2.0*TP/(2*TP+FP+FN):0; }
        double avg(long n) const { return rt_sum/n; }
    };

    Stat S_row,S_col,S_cost,S_hopt,S_fann,S_lgb,S_opt;
    double oracle_rt_sum = 0.0;

    for (size_t i=0;i<DS_test.size();++i){
        const Sample &s = DS_test[i];
        // bool gt  = s.label;                  // 1 ⇒ column faster
        bool gt = (s.col_t < s.row_t);
        bool lgb = pred_lgb[i];

        S_row .add(false            ,gt,s.row_t,s.col_t);
        S_col .add(true             ,gt,s.row_t,s.col_t);
        S_cost.add(s.qcost>COST_THR ,gt,s.row_t,s.col_t);
        S_hopt.add(s.hybrid_pred    ,gt,s.row_t,s.col_t);
        S_fann.add(s.fann_pred      ,gt,s.row_t,s.col_t);
        S_lgb .add(lgb              ,gt,s.row_t,s.col_t);

        /* Oracle: always pick the faster of the two paths */
        // bool oracle_pred = s.label;          // same definition as gt
        bool oracle_pred = (s.col_t < s.row_t);   // true ⇒ choose column
        S_opt.add(oracle_pred   ,gt ,s.row_t, s.col_t);

        oracle_rt_sum += std::min(s.row_t, s.col_t);
    }
    const long N = DS_test.size();
    printf("oracle avg time: %f", oracle_rt_sum/N);

    /* ------------ CSV -------------- */
    std::ofstream fout("test_metrics.csv");
    fout<<"method,TP,FP,TN,FN,accuracy,recall,f1,avg_runtime\n";
    auto dump=[&](const std::string &n,Stat&s){
        fout<<n<<','<<s.TP<<','<<s.FP<<','<<s.TN<<','<<s.FN<<','
            <<s.acc(N)<<','<<s.rec()<<','<<s.f1()<<','<<s.avg(N)<<'\n';
    };
    dump("row_only"   ,S_row );
    dump("column_only",S_col );
    dump("cost_rule"  ,S_cost);
    dump("hybrid_opt" ,S_hopt);
    dump("Kernel Model"       ,S_fann);
    dump("AI Model"   ,S_lgb );
    dump("Oracle"     ,S_opt );
    fout.close();

    /* ------------ console ----------- */
    std::cout<<"\n*** Evaluation on "<<N<<" samples ***\n"
             <<"LightGBM ensemble  Acc="<<S_lgb.acc(N)
             <<"  Rec="<<S_lgb.rec()<<"  F1="<<S_lgb.f1()<<"\n"
             <<"TP="<<S_lgb.TP<<" FP="<<S_lgb.FP
             <<" TN="<<S_lgb.TN<<" FN="<<S_lgb.FN<<"\n";

    std::cout<<"\n| Method    | TP | FP | TN | FN |  Acc |  Rec |  F1 | Avg-RT |\n"
             <<"|-----------|----|----|----|----|------|------|-----|--------|\n";
    auto pr=[&](const std::string &n,Stat&s){
        std::cout<<'|'<<std::setw(10)<<std::left<<n<<'|'
                 <<std::setw(4)<<s.TP<<'|'<<std::setw(4)<<s.FP<<'|'
                 <<std::setw(4)<<s.TN<<'|'<<std::setw(4)<<s.FN<<'|'
                 <<std::setw(6)<<s.acc(N)<<'|'<<std::setw(6)<<s.rec()<<'|'
                 <<std::setw(5)<<s.f1()<<'|'<<std::setw(8)<<std::setprecision(6)
                 <<s.avg(N)<<"|\n";
        std::cout.unsetf(std::ios::floatfield);
    };
    pr("Row"     ,S_row );
    pr("Column"  ,S_col );
    pr("Cost"    ,S_cost);
    pr("Hybrid"  ,S_hopt);
    pr("Kernel Model"    ,S_fann);
    pr("AI Model"    ,S_lgb );
    pr("Oracle"  ,S_opt );

    std::cout<<"\n[CSV] test_metrics.csv written (includes Oracle)\n";
}

