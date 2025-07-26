/*───────────────────────────────────────────────────────────
  Portable <filesystem> pickup
  • GCC 9+ / Clang –> <filesystem>
  • GCC 5-8       –> <experimental/filesystem>  (+-lstdc++fs)
  • GCC 4.8       –> tiny fallback shim (no extra deps)
───────────────────────────────────────────────────────────*/
#include <cerrno>
#include <dirent.h>     // opendir / readdir / closedir
#include <sys/stat.h>   // stat
#include <unistd.h>     // access
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <regex>
#include <mysql/mysql.h>
#include <iostream>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <thread>
#include <stdexcept>
#include <cassert>
#include <iomanip>

#include "json.hpp"

#include "global_stats.hpp"
#include "common.hpp"

using namespace std;

using json = nlohmann::json;

/* ────────── NEW: feature 缓存 ────────── */
using FeatVec = std::array<float, NUM_FEATS>;

/* key = <dataset_dir>/<query_id>   例如  tpch_sf1/42 */
static std::unordered_map<std::string, FeatVec> FEAT_CACHE;


/* -----------------------------------------------------------
 *  load_features_for_dir()
 *    读取 <dir>/features_24d.csv ，填充 FEAT_CACHE
 * ----------------------------------------------------------- */
static void
load_features_for_dir(const std::string& dir)
{
    std::string csv = dir + "/features_24d.csv";
    if (!file_exists(csv)) {
        logW("features_24d.csv not found in " + dir);
        return;
    }
    std::ifstream fin(csv);
    std::string line; std::getline(fin, line);          // skip header

    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        std::string tok;

        /* —— 读 query_id —— */
        std::getline(ss, tok, ',');
        std::string qid = tok;

        /* —— 读 24 维特征 —— */
        FeatVec fv{};
        for (int i = 0; i < NUM_FEATS; ++i) {
            std::getline(ss, tok, ',');
            fv[i] = tok.empty() ? 0.f : std::stof(tok);
        }
        /* —— 剩余 SQL 列忽略 —— */

        FEAT_CACHE.emplace(dir + '/' + qid, std::move(fv));
    }
    logI("Loaded feature cache for " + dir);
}

bool
load_plan_file(const std::string& fp_row,
               const std::string& qid,
               const std::string& rowDir,
               const std::string& colDir,
               float             feat[NUM_FEATS],
               double&           qcost,
               Graph&            colGraph,
               bool              need_col)        // ← 原参数保留
{
    /* =========== 1. 第一次遇到目录时，加载 CSV =========== */
    if (FEAT_CACHE.empty() ||
        FEAT_CACHE.find(rowDir + '/' + qid) == FEAT_CACHE.end())
        load_features_for_dir(rowDir.substr(0, rowDir.find_last_of('/')));

    /* =========== 2. 直接取 24 维特征 =========== */
    auto it = FEAT_CACHE.find(rowDir + '/' + qid);
    if (it == FEAT_CACHE.end()) {
        logW("feature vector missing for " + qid);
        return false;
    }
    std::copy(it->second.begin(), it->second.end(), feat);

    /* query_cost 等可置 0（或从 CSV 追加一列再读取） */
    qcost = 0.0;

    /* 列计划不再需要，可根据 need_col 决定是否仍解析 */
    if (need_col) {
        /* 如仍想保留旧 GNN 功能，按原逻辑解析 col-plan；
           否则返回空 Graph 即可 */
        colGraph = Graph{};
    }

    return true;
}


/*───────────────────────────────────────────────────────────
 *  Abridged Agg / walk / plan2feat
 *  –   keeps only the 32 features:
 *      0 1 2 3 5 6 8 9 12 14 21 22 23 25 26 27 30 31 32 33
 *      34 35 38 39 44 47 57 62 63 64 96 123
 *───────────────────────────────────────────────────────────*/



std::unordered_map<std::string,double> g_db_size_gb;


extern bool g_need_col_plans;   // 缺省为 true，兼容旧逻辑


std::unordered_map<std::string,int> GLOBAL_OP2ID{ {"UNK",0} };


void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }


/* ———— 简易文件/目录工具 ———— */
bool file_exists(const std::string& p){
    return ::access(p.c_str(), F_OK) == 0;
}
bool is_directory(const std::string& p){
    struct stat sb{};
    return ::stat(p.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}
bool has_ext(const std::string& f, const std::string& ext){
    return f.size() > ext.size() &&
           f.compare(f.size() - ext.size(), ext.size(), ext) == 0;
}
std::string strip_ext(const std::string& f){
    auto pos = f.rfind('.');
    return (pos == std::string::npos) ? f : f.substr(0, pos);
}

void progress(const string&tag,size_t cur,size_t tot,size_t W){
    double f=tot?double(cur)/tot:1.0; size_t filled=size_t(f*W);
    cerr<<"\r"<<tag<<" ["<<string(filled,'=')<<string(W-filled,' ')
        <<"] "<<setw(3)<<int(f*100)<<"% ("<<cur<<'/'<<tot<<')'<<flush;
    if(cur==tot) cerr<<'\n';
}


/* ─────────────────────────────  utilities  ────────────────────────────── */
double parsePossibleNumber(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string())               return std::stod(j[k].get<std::string>());
        if(j[k].is_number_float()||
           j[k].is_number_integer())       return j[k].get<double>();
    }catch(...){ logW("Invalid number for key: "+k);}
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

double safe_f(const json&v){
    if(v.is_number()) return v.get<double>();
    if(v.is_string()){ try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
double safe_f(const json&o,const char*k){ return o.contains(k)?safe_f(o[k]):0; }
double log1p_clip(double v){ return log1p(max(0.0,v)); }
double str_size_to_num(string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    double m=1; char suf=s.back();
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return stod(s)*m;}catch(...){ return 0.0;}
}
bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    const auto&v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){ string s=v.get<string>(); transform(s.begin(),s.end(),s.begin(),::tolower);
                       return s=="yes"||s=="true"||s=="1";}
    return false;
}

/* ▶ 单库查询：返回 GB；失败时给 1.0 作稳妥默认 */
static double fetch_db_size_gb(const std::string& host,int port,
                               const std::string& user,const std::string& pass,
                               const std::string& schema)
{
    MYSQL* c = mysql_init(nullptr);
    mysql_options(c, MYSQL_OPT_RECONNECT, "1");
    if (!mysql_real_connect(c, host.c_str(), user.c_str(), pass.c_str(),
                            "information_schema", port, nullptr, 0))
    {
        logW("MySQL connect fail for "+schema+", use 1.0 GB default");
        return 1.0;
    }
    std::string sql =
        "SELECT COALESCE(SUM(DATA_LENGTH+INDEX_LENGTH),0) "
        "FROM TABLES WHERE TABLE_SCHEMA='" + schema + '\'';
    unsigned long long bytes = 0;
    if (mysql_query(c, sql.c_str()) == 0)
    {
        if (MYSQL_RES* r = mysql_store_result(c))
        {
            if (MYSQL_ROW row = mysql_fetch_row(r))
                bytes = row && row[0] ? strtoull(row[0],nullptr,10) : 0ULL;
            mysql_free_result(r);
        }
    }
    mysql_close(c);
    return bytes / (1024.0*1024*1024);          // → GB
}

/* ▶ 对外唯一接口：一次把所有 schema 的大小放进 g_db_size_gb */
void collect_db_sizes(const std::string& host,
                      int               port,
                      const std::string& user,
                      const std::string& pass,
                      const std::vector<std::string>& schemas)
{
    g_db_size_gb.clear();
    for (const auto& s : schemas)
    {
        double gb = fetch_db_size_gb(host,port,user,pass,s);
        // if (gb < 1e-6) gb = 1.0;                // 最低 1 MB ⇒ 1 GB 说明值异常
        g_db_size_gb[s] = gb;
        std::cerr << "[DB_SIZE] " << s << " = " << gb << " GB\n";
    }
}

double DecisionTree::gini(double pos,double tot){
    if(tot<=0) return 1.0;
    double q=pos/tot; return 2.0*q*(1.0-q);
}

int DecisionTree::build(const std::vector<int>& idx,
            const std::vector<std::array<float,NUM_FEATS>>& X,
            const std::vector<int>&  y,
            const std::vector<float>& w,
            int depth,int max_depth,int min_samples)
{
    /* ---- weighted positive / total ---- */
    double pos_w = 0.0, tot_w = 0.0;
    for(int i:idx){ tot_w += w[i]; pos_w += w[i]*y[i]; }

    DTNode node; node.prob = float( pos_w / std::max(1e-12, tot_w) );
    double parent_g = gini(pos_w, tot_w);  /* 父节点基尼 */

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

    if (best_f == -1 || parent_g - best_g < min_gain_) {
        nodes_.push_back(node);           // make this a leaf
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

DecisionTree::DecisionTree(int md,int ms,double mg)
    : min_gain_(mg), max_depth_(md), min_samples_(std::max(80,ms)) {}


void DecisionTree::fit(const vector<array<float,NUM_FEATS>>&X,
        const vector<int>&y,const vector<float>&w){
    nodes_.clear();
    vector<int> idx(X.size());
    iota(idx.begin(),idx.end(),0);
    // build(idx,X,y,w,0);
    build(idx, X, y, w, 0, max_depth_, min_samples_);
}

float DecisionTree::predict(const float*f) const {
    int id=0;
    while(nodes_[id].feat!=-1)
        id = (f[nodes_[id].feat] < nodes_[id].thr)
            ? nodes_[id].left
            : nodes_[id].right;
    return nodes_[id].prob;
}

json DecisionTree::to_json() const {
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
void DecisionTree::from_json(const json& arr) {
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


// RandomForest::RandomForest(int n,int md,int ms,double mg,double sr)
//     : sampleRatio_(sr)
//     {
//     trees_.reserve(n);
//     for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
// }

// void RandomForest::fit(const std::vector<std::array<float,NUM_FEATS>>& X,
//         const std::vector<int>&  y,
//         const std::vector<float>& w,
//         std::mt19937& rng)
// {
//     using clock = std::chrono::steady_clock;
//     auto t0 = clock::now();

//     std::uniform_int_distribution<int> uni(0, int(X.size() - 1));
//     std::size_t T = trees_.size();
//     //  #pragma omp parallel for schedule(dynamic)
//     for(int t = 0; t < int(T); ++t)
//     {
//         std::mt19937 rng_local(rng()+t);
//         std::uniform_int_distribution<int> uni_local(0, int(X.size()-1));

//         std::size_t m = std::size_t(sampleRatio_ * X.size());
//         std::vector<std::array<float,NUM_FEATS>> BX; BX.reserve(m);
//         std::vector<int>  By;  By.reserve(m);
//         std::vector<float>Bw;  Bw.reserve(m);

//         for(std::size_t i=0;i<m;++i){
//         int j = uni_local(rng_local);
//         BX.push_back(X[j]);
//         By.push_back(y[j]);
//         Bw.push_back(w[j]);
//         }

//         trees_[t].fit(BX,By,Bw);
//         // #pragma omp critical
//         progress("RF-train", t+1, T);
//     }

//     auto t1 = clock::now();
//     logI("Random-Forest training took " +
//             std::to_string(std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()) +
//             " s");
// }

// float RandomForest::predict(const float*f) const {
//     double s=0;
//     for(const auto& tr: trees_) s += tr.predict(f);
//     return float(s/trees_.size());
// }

// json RandomForest::to_json() const {
//     json arr = json::array();
//     for(const auto& tr: trees_) arr.push_back(tr.to_json());
//     return arr;
// }

// // ─────── NEW from_json ───────
// void RandomForest::from_json(const json& arr) {
//     trees_.clear();
//     for(const auto& jt : arr) {
//         DecisionTree tr(0,0,0);
//         tr.from_json(jt);
//         trees_.push_back(std::move(tr));
//     }
// }


/* ─────────── MySQL index-map (unchanged logic) ─────────── */
unordered_map<string, unordered_set<string>> indexCols;

void load_index_defs_from_db(const string &host,int port,
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

void load_all_index_defs(const string &host,int port,
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


/* 旧: /10.0 → 取值 10⁵ 以上就饱和 */
inline double log_tanh(double v, double c = 20.0) {
    return std::tanh(std::log1p(std::max(0.0, v)) / c);   // c=20
}


inline double log_scale(double v, double k = 1e6){
    return std::log1p(v) / std::log1p(k);   // v=k 时输出 1
}

/* ---------- 计表数：旧版 JSON 同样含 "table" ----------*/
static int count_tables(const json& node){
    int cnt = 0;
    std::function<void(const json&)> rec = [&](const json& n){
        if(n.is_object()){
            if(n.contains("table")) ++cnt;
            for(const auto& kv : n.items()) rec(kv.value());
        }else if(n.is_array()){
            for(const auto& v : n) rec(v);
        }
    };
    rec(node);
    return cnt;
}

/* ---------- 全树找 MIN / MAX 函数 ----------*/
static bool tree_has_minmax(const json& node){
    if(node.is_object()){
        if(node.contains("function") && node["function"].is_string()){
            std::string fn = node["function"];
            std::transform(fn.begin(),fn.end(),fn.begin(),::tolower);
            if(fn=="min" || fn=="max") return true;
        }
        for(const auto& kv:node.items())
            if(tree_has_minmax(kv.value())) return true;
    }else if(node.is_array()){
        for(const auto& v:node)
            if(tree_has_minmax(v)) return true;
    }
    return false;
}

static bool has_grouping(const json& qb){
    /* 各版本 key 略不同，这里罗列常见几种 */
    return qb.contains("grouping_operation") ||
           qb.contains("grouping")           ||
           qb.contains("grouping_sets");
}

static bool min_or_max_no_group(const json& qb){
    return tree_has_minmax(qb) && !has_grouping(qb);
}


/* ===================================================================== *
 *  walk()  – recursive aggregation over a MySQL JSON plan
 *           (now fully robust to string-encoded numbers)
 * ===================================================================== */

void walk(const nlohmann::json& n, Agg& a, int depth = 1)
{
    using json = nlohmann::json;

    if (n.is_object())
    {
        /* ── TABLE node ─────────────────────────────────────────────── */
        if (n.contains("table") && n["table"].is_object())
        {
            const auto& t  = n["table"];
            const auto& ci = t.value("cost_info", json::object());

            double re = safe_f(t , "rows_examined_per_scan");
            double rp = safe_f(t , "rows_produced_per_join");
            double filtered = safe_f(t , "filtered");
            double fl = std::max(0.01, filtered) / 100.0;

            double rc = safe_f(ci, "read_cost");
            double ec = safe_f(ci, "eval_cost");
            double pc = safe_f(ci, "prefix_cost");

            a.re += re;  a.rp += rp;  a.rc += rc;  a.ec += ec;  a.pc += pc;  ++a.cnt;
            a.minRead   = std::min(a.minRead , rc);
            a.maxPrefix = std::max(a.maxPrefix, pc);

            a.selSum += fl;
            a.selMin  = std::min(a.selMin, fl);
            a.selMax  = std::max(a.selMax, fl);

            if (ec > 1e-12) a.ratioMax = std::max(a.ratioMax, rc / ec);

            const std::string at = t.value("access_type", "ALL");
            if      (at == "range")   ++a.cRange;
            else if (at == "ref")     ++a.cRef;
            else if (at == "eq_ref")  ++a.cEq;
            else if (at == "index")   ++a.cIdx;

            if (t.value("using_index", false)) ++a.idxUse;

            if (t.contains("possible_keys") && t["possible_keys"].is_array())
                a.sumPK += static_cast<int>(t["possible_keys"].size());

            if (a.outerRows == 0 && at != "ALL") a.outerRows = re;

            if (at == "eq_ref") {
                ++a._curEqChain;
                a.eqChainDepth = std::max(a.eqChainDepth, a._curEqChain);
            } else {
                a._curEqChain = 0;
            }

            if (depth == 3) a.pcDepth3 += pc;

            a.maxDepth = std::max(a.maxDepth, depth);
        }

        /* Handle nested_loop to increment depth progressively */
        if (n.contains("nested_loop") && n["nested_loop"].is_array()) {
            int sub_depth = depth;
            for (const auto& v : n["nested_loop"]) {
                walk(v, a, sub_depth);
                sub_depth += 1;
            }
        } else {
            /* recurse (skip the “table” and "nested_loop" keys already handled) */
            for (const auto& kv : n.items())
                if (kv.key() != "table" && kv.key() != "nested_loop") walk(kv.value(), a, depth + 1);
        }
    }
    else if (n.is_array())
        for (const auto& v : n) walk(v, a, depth);
}

bool plan2feat(const nlohmann::json& plan, float f[NUM_FEATS])
{
  using json = nlohmann::json;

  if (!plan.contains("query_block")) return false;
  const json *qb = &plan["query_block"];

  if (qb->contains("union_result")) {
    const auto &specs = (*qb)["union_result"]["query_specifications"];
    if (specs.is_array() && !specs.empty())
      qb = &specs[0]["query_block"];
  }

  Agg a; walk(*qb, a);
  if (!a.cnt) return false;

  const double inv   = 1.0 / a.cnt;
  const double qCost = safe_f(qb->value("cost_info", json::object()),
                              "query_cost");

  auto lt = [](double v) { return std::tanh(std::log1p(std::max(0.0, v)) / 20.0); };

  int k = 0; auto PUSH = [&](double v){ f[k++] = float(v); };

  PUSH( lt(a.re*inv) );                              // 0
  PUSH( lt(a.rp*inv) );                              // 1
  PUSH( lt(a.rc*inv) );                              // 2
  PUSH( lt(a.pc*inv) );                              // 3

  PUSH( a.cRef   *inv );                             // 4
  PUSH( a.cEq    *inv );                             // 5
  PUSH( a.idxUse *inv );                             // 6

  PUSH( a.selMin );                                  // 7
  PUSH( a.ratioMax );                                // 8
  PUSH( std::log1p(qCost)/15.0 );                    // 9

  PUSH( lt((a.pc*inv)/std::max(1e-6,a.rc*inv)) );    // 10
  PUSH( lt((a.rc*inv)/std::max(1e-6,a.re*inv)) );    // 11

  PUSH( a.cnt );                                     // 12
  PUSH( a.cnt ? double(a.sumPK)/a.cnt : 0 );         // 13
  PUSH( lt(a.maxPrefix) );                           // 14
  PUSH( lt(a.minRead < 1e30 ? a.minRead : 0) );      // 15

  PUSH( a.selMax - a.selMin );                       // 16
  PUSH( a.idxUse / double(std::max(1, a.cRange + a.cRef + a.cEq + a.cIdx)) ); // 17

  PUSH( lt(a.re*inv) - lt(a.selSum*inv) );           // 18
  PUSH( lt(a.maxPrefix*inv) );                       // 19
  PUSH( a.selMin > 0 ? a.selMax / a.selMin : 0 );    // 20
  PUSH( lt(a.outerRows) );                           // 21
  PUSH( double(a.eqChainDepth) );                    // 22
  PUSH( lt( (a.pc > 0 ? a.pc : 1e-6) / std::max(1e-6, a.pcDepth3) ) ); // 23

  return k == NUM_FEATS;
}
std::unordered_map<std::string, ColStats> colStats;



/* ---------- helpers -------------------------------------------------- */
ColDType map_dtype(std::string t) {
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

double ndv_from_histogram(const std::string& hist_json)
{
    try {
        auto j = nlohmann::json::parse(hist_json);
        double ndv = 0.0;
        for (const auto& b : j.at("buckets"))
            ndv += b.value("distinct-range", 0.0);
        return ndv > 0 ? ndv : -1;
    } catch (...) { return -1; }
}

double ndv_from_statistics(MYSQL* conn,
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
bool populate_col_stats(const std::string& host, int port,
                               const std::string& user, const std::string& pass,
                               const std::vector<std::string>& dbs)
{
    colStats.clear();

    MYSQL* conn = mysql_init(nullptr);
    bool reconnect = 1;
    mysql_options(conn, MYSQL_OPT_RECONNECT, &reconnect);
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
ColStats lookup_col_stats(const std::string& id)
{
    auto it = colStats.find(id);
    return (it == colStats.end()) ? ColStats{} : it->second;
}

/* ---------- C++11 normalise() for std::array ---------- */
template <size_t N>
void normalise(std::array<float, N>& hist)
{
    float sum = 0.f;
    for (float v : hist) sum += v;
    if (sum < 1.f) sum = 1.f;
    for (float& v : hist) v /= sum;
}


/* ────────────────────────────────────────────────────────────── */
/*  Light table / index meta cached at start-up                  */
/* ────────────────────────────────────────────────────────────── */

unordered_map<string,TblStats> tblStats;     // key = db.tbl


/* --------------------------------------------------------------------- */
/*  Call once at start-up: fill `tblStats` from INFORMATION_SCHEMA + PS  */
/*  – now robust against empty result-sets and NULL fields               */
/* --------------------------------------------------------------------- */
bool populate_tbl_stats(const std::string &host, int port,
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


const TblStats& lookup_tbl(const string& id){
    static const TblStats dflt{};
    auto it=tblStats.find(id);
    return (it==tblStats.end())?dflt:it->second;
}




// parseColPlan now uses per-dir stats for numeric norms and a global op2id map
Graph parseColPlan(const json& j,
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
        // double nr = (rawR - st.rows_c) / st.rows_s;
        // double nc = (rawC - st.cost_c) / st.cost_s;
        auto gs = global_stats().get();
        double nr = (rawR - gs.rows_c) / gs.rows_s;
        double nc = (rawC - gs.cost_c) / gs.cost_s;
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



ColStats buildColStats(const std::string& dir) {
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
        // st.rows_c = mid(rows);  st.cost_c = mid(cost);
        // st.rows_s = std::max(1e-6, iqr(rows));
        // st.cost_s = std::max(1e-6, iqr(cost));
        for (size_t i = 0; i < rows.size(); ++i)
            global_stats().update(rows[i], cost[i]);
        int id = 0;
        for (const auto& op : ops) st.op2id[op] = id++;
    }
    // cache to disk
    json dump{{"rows_c",st.rows_c},{"rows_s",st.rows_s},{"cost_c",st.cost_c},{"cost_s",st.cost_s},{"op2id",st.op2id}};
    std::ofstream(cache) << dump.dump(2);
    return st;
}


std::unordered_map<std::string,ColStats> STATS_CACHE;

/* 1. 目录级 col-stats 取函数  ─────────────────────────────── */
ColStats& get_col_stats_for_dir(const std::string& dir)
{
    auto it = STATS_CACHE.find(dir);
    if (it == STATS_CACHE.end()) {
        /* 第一次遇到就扫描并缓存 */
        ColStats cs = buildColStats(dir.substr(0, dir.find_last_of('/'))); // 传上层数据集目录
        it = STATS_CACHE.emplace(dir, std::move(cs)).first;
    }
    return it->second;
}



// /* 额外加一个缺省参数 bool need_col = g_need_col_plans */
// bool
// load_plan_file(const std::string& fp_row,
//                const std::string& qid,
//                const std::string& rowDir,
//                const std::string& colDir,
//                float             feat[NUM_FEATS],
//                double&           qcost,
//                Graph&            colGraph,
//                bool              need_col)     // ← 新增
// {
//     /* ----------- row plan：保持不动 ----------- */
//     if (!file_exists(fp_row)) return false;
//     std::ifstream in(fp_row);
//     nlohmann::json j;  try { in >> j; } catch (...) { return false; }

//     if (!plan2feat(j, feat)) return false;
//     qcost = j.contains("query_block")
//               ? safe_f(j["query_block"].value("cost_info", json::object()),
//                        "query_cost")
//               : 0.0;

//     /* ----------- column plan：按需解析 -------- */
//     if (need_col) {
//         std::string fp_col = colDir + "/" + qid + ".json";
//         if (file_exists(fp_col)) {
//             try {
//                 std::ifstream ic(fp_col);  nlohmann::json cj;  ic >> cj;
//                 const ColStats& st = get_col_stats_for_dir(colDir);
//                 const auto&     id = global_op2id();
//                 colGraph = parseColPlan(cj, st, id);
//             } catch (...) { /* bad column plan ⇒ 忽略 */ }
//         }
//     }
//     return true;
// }



/* 给定全部数据集目录 & 测试集比例（默认 0.2） */
Split split_dirs(vector<string> dirs,
                        double test_frac,
                        uint32_t seed)
{
    std::mt19937 gen(seed);
    std::shuffle(dirs.begin(), dirs.end(), gen);

    size_t n_test = size_t(dirs.size() * test_frac + 0.5);
    Split sp;
    sp.test_dirs  .assign(dirs.begin(),                dirs.begin()+n_test);
    sp.train_dirs .assign(dirs.begin()+n_test,         dirs.end());
    return sp;
}

std::vector<Fold> make_lodo(const std::vector<std::string>& dirs){
    std::vector<Fold> out;
    for (const auto& d_val : dirs){
        Fold f;
        f.val_dirs = {d_val};
        for (const auto& d_tr : dirs)
            if (d_tr != d_val) f.tr_dirs.push_back(d_tr);
        out.push_back(std::move(f));
    }
    return out;                 // K = |dirs|
}

std::vector<Fold> make_cv3(const std::vector<std::string>& dirs)
{
    constexpr std::size_t K = 3;        // number of folds
    std::vector<Fold> folds(K);         // folds[0], folds[1], folds[2]

    /* ---------- assign validation dirs ---------- */
    for (std::size_t i = 0; i < dirs.size(); ++i)
        folds[i % K].val_dirs.push_back(dirs[i]);      // round-robin

    /* ---------- assign training dirs ---------- */
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t i = 0; i < dirs.size(); ++i)
            if (i % K != k) folds[k].tr_dirs.push_back(dirs[i]);

    return folds;                       // always returns 3 folds
}



/* ───── new: 全量一次性加载 ───── */
using DirSamples = unordered_map<string, vector<Sample>>;


/* ──────────────────────────────────────────────────────────
 *  load_all_datasets() ― 新版
 *    · 不再读取 row_plans / column_plans
 *    · 仅依赖
 *        <dir>/query_costs.csv   （9 列）
 *        <dir>/features_24d.csv  （25 列：query_id + 24-dim）
 * ──────────────────────────────────────────────────────────*/
DirSamples
load_all_datasets(const std::string& base,
                  const std::vector<std::string>& dirs)
{
    DirSamples M;                                         // {dir ⇒ vector<Sample>}

    for (const auto& d : dirs)
    {
        /* 1) 路径拼接 ---------------------------------------------------- */
        const std::string dir_path = base + '/' + d;
        const std::string csv_meta = dir_path + "/query_costs.csv";
        const std::string csv_feat = dir_path + "/features_140d.csv";

        if (!file_exists(csv_meta) || !file_exists(csv_feat)) {
            logW("skip " + d + "  (missing csv)");
            continue;
        }

        /* 2) 一次性把 24-d 特征读进 FEAT_CACHE -------------------------- *
         *    load_features_for_dir() 已在前文实现；
         *    key = "<dir_path>/<query_id>"
         * --------------------------------------------------------------- */
        load_features_for_dir(dir_path);

        /* 3) 读取 meta（延迟 / 预测标签等） ----------------------------- */
        struct Meta {
            double rt = 60, ct = 60;     // row_time / column_time（秒）
            double qcost = 0;            // optimizer cost
            int    cost = -1;            // cost_rule 预测 (0/1) – 可选
            int    hyb  = -1;            // hybrid_use_imci
            int    fann = -1;            // fann_use_imci
            int    label = -1;           // use_imci (gt)
        };
        std::unordered_map<std::string, Meta> meta;

        {
            std::ifstream fin(csv_meta);
            std::string line;
            std::getline(fin, line);                     // ← 跳过 header

            while (std::getline(fin, line))
            {
                /* CSV 最后一列是完整 SQL，里面可能含逗号。
                 * 因此只解析前 8 列，其余全部丢给 query 字段。 */
                std::stringstream ss(line);

                std::string qid, use_imci, rt, ct, qcost,
                            cost_flag, hyb_flag, fann_flag;

                std::getline(ss, qid      , ',');   // 0
                std::getline(ss, use_imci , ',');   // 1
                std::getline(ss, rt       , ',');   // 2
                std::getline(ss, ct       , ',');   // 3
                std::getline(ss, qcost    , ',');   // 4
                std::getline(ss, cost_flag, ',');   // 5
                std::getline(ss, hyb_flag , ',');   // 6
                std::getline(ss, fann_flag, ',');   // 7
                /* 第 8 列开始是原始 SQL，忽略即可 */

                Meta m;
                m.label = use_imci.empty() ? -1 : std::stoi(use_imci);
                m.rt    = rt.empty()      ? 60.0  : std::stod(rt);
                m.ct    = ct.empty()      ? 60.0  : std::stod(ct);
                m.qcost = qcost.empty()   ? 0.0   : std::stod(qcost);
                m.cost  = cost_flag.empty()? -1   : std::stoi(cost_flag);
                m.hyb   = hyb_flag.empty() ? -1   : std::stoi(hyb_flag);
                m.fann  = fann_flag.empty()? -1   : std::stoi(fann_flag);

                meta.emplace(qid, std::move(m));
            }
        }

        /* 4) 构造 Sample 向量 ------------------------------------------ */
        std::vector<Sample> samples;
        samples.reserve(meta.size());

        for (const auto& kv : meta)
        {
            const std::string& qid = kv.first;
            const Meta&        m   = kv.second;

            /* 4-a) 找到预缓存的 24-d 特征 ------------------------------ */
            auto it_feat = FEAT_CACHE.find(dir_path + '/' + qid);
            if (it_feat == FEAT_CACHE.end()) {
                logW("feature vector missing for " + d + '/' + qid);
                continue;
            }

            /* 4-b) 组装 Sample ---------------------------------------- */
            Sample s;
            std::copy(it_feat->second.begin(),
                      it_feat->second.end(),
                      s.feat.begin());

            s.qcost       = m.qcost;          // 供 cost-rule 动态阈值使用
            s.colGraph    = Graph{};          // 新管线无列计划
            s.row_t       = m.rt;
            s.col_t       = m.ct;
            s.label       = int(m.ct <= m.rt);      // 只认真实 runtime
            s.fann_pred   = m.fann;
            s.hybrid_pred = m.hyb;
            s.dir_tag     = d;
            /* 若想保留 cost_rule 预测，可把 m.cost 写进 Sample 新字段 */

            samples.emplace_back(std::move(s));
        }

        logI("dir " + d + " → " + std::to_string(samples.size()) + " samples");
        M.emplace(d, std::move(samples));
    }
    return M;
}



/* —— helper: 把若干目录样本合并 —— */
vector<Sample> build_subset(const vector<string>& dirs,
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

vector<string> pick_test3(vector<string> dirs, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::shuffle(dirs.begin(), dirs.end(), gen);
    if (dirs.size() < 3) { logE("need ≥3 dirs"); exit(1); }
    return {dirs.begin(), dirs.begin()+3};
}


/* ─────────────────────────────────────────────────────────────
 *  report_and_evaluate   — Macro-P / R / F1 版本
 *  仍输出 AI_cost_OK / AI_cost_BAD
 * ───────────────────────────────────────────────────────────── */
void report_metrics(const std::vector<int>& pred_lgb,
                    const std::vector<Sample>& DS_test)
{
    struct Stat {
        long TP = 0, FP = 0, TN = 0, FN = 0;
        double rt_sum = 0;

        void add(bool pred, bool gt, double rt, double ct) {
            (pred ? (gt ? ++TP : ++FP) : (gt ? ++FN : ++TN));
            rt_sum += (pred ? ct : rt);
        }

        /* —— 每类指标 —— */
        double prec_pos() const { return (TP + FP) ? double(TP) / (TP + FP) : 0; }
        double rec_pos()  const { return (TP + FN) ? double(TP) / (TP + FN) : 0; }
        double f1_pos()   const { return TP ? 2.0 * TP / (2 * TP + FP + FN) : 0; }

        double prec_neg() const { return (TN + FN) ? double(TN) / (TN + FN) : 0; }
        double rec_neg()  const { return (TN + FP) ? double(TN) / (TN + FP) : 0; }
        double f1_neg()   const { long TPn = TN, FPn = FN, FNn = FP;
                                   return TPn ? 2.0 * TPn / (2 * TPn + FPn + FNn) : 0; }

        /* —— Macro 指标 —— */
        double precM() const { return 0.5 * (prec_pos() + prec_neg()); }
        double recM () const { return 0.5 * (rec_pos () + rec_neg ()); }
        double f1M  () const { return 0.5 * (f1_pos  () + f1_neg  ()); }

        double acc(long n) const { return n ? double(TP + TN) / n : 0; }
        double avg(long n) const { return n ? rt_sum / n : 0; }
    };

    /* 统计结构体 */
    Stat S_row, S_col, S_cost, S_hopt, S_fann, S_lgb, S_opt;
    Stat S_ai_cost_ok, S_ai_cost_bad;
    long N_cost_ok = 0, N_cost_bad = 0;

    double oracle_rt_sum = 0;

    for (size_t i = 0; i < DS_test.size(); ++i) {
        const Sample& s = DS_test[i];
        bool gt        = (s.col_t < s.row_t);   // ground-truth
        bool lgb       = pred_lgb[i];
        bool cost_pred = (s.qcost > COST_THR);

        S_row .add(false , gt, s.row_t, s.col_t);
        S_col .add(true  , gt, s.row_t, s.col_t);
        S_cost.add(cost_pred        , gt, s.row_t, s.col_t);
        S_hopt.add(s.hybrid_pred==1 , gt, s.row_t, s.col_t);
        S_fann.add(s.fann_pred  ==1 , gt, s.row_t, s.col_t);
        S_lgb .add(lgb            , gt, s.row_t, s.col_t);

        /* Oracle 总是选择更快的 */
        S_opt.add(gt, gt, s.row_t, s.col_t);

        oracle_rt_sum += std::min(s.row_t, s.col_t);

        /* cost-rule 正 / 误 子集 */
        if (cost_pred == gt) {           // 命中
            S_ai_cost_ok .add(lgb, gt, s.row_t, s.col_t);
            ++N_cost_ok;
        } else {
            S_ai_cost_bad.add(lgb, gt, s.row_t, s.col_t);
            ++N_cost_bad;
        }
    }

    const long N = DS_test.size();
    printf("oracle avg time: %.6f\n", oracle_rt_sum / N);

    /* ============  CSV ============ */
    std::ofstream fout("test_metrics.csv");
    fout << "method,TP,FP,TN,FN,accuracy,macro_precision,macro_recall,macro_f1,avg_runtime\n";

    auto dump = [&](const std::string& name, Stat& s, long n_samples) {
        fout << name << ','
             << s.TP << ',' << s.FP << ',' << s.TN << ',' << s.FN << ','
             << s.acc(n_samples) << ','
             << s.precM()        << ','
             << s.recM()         << ','
             << s.f1M()          << ','
             << s.avg(n_samples) << '\n';
    };

    dump("row_only"      , S_row , N);
    dump("column_only"   , S_col , N);
    dump("cost_rule"     , S_cost, N);
    dump("hybrid_opt"    , S_hopt, N);
    dump("Kernel Model"  , S_fann, N);
    dump("AI Model"      , S_lgb , N);
    dump("Oracle"        , S_opt , N);
    dump("AI_on_cost_OK" , S_ai_cost_ok , N_cost_ok );
    dump("AI_on_cost_BAD", S_ai_cost_bad, N_cost_bad);
    fout.close();

    /* ============  终端输出 ============ */
    std::cout << "\n*** Evaluation on " << N << " samples ***\n"
              << "LightGBM ensemble  Acc="   << S_lgb.acc(N)
              << "  Macro-Prec=" << S_lgb.precM()
              << "  Macro-Rec="  << S_lgb.recM()
              << "  Macro-F1="   << S_lgb.f1M() << "\n"
              << "TP=" << S_lgb.TP << " FP=" << S_lgb.FP
              << " TN=" << S_lgb.TN << " FN=" << S_lgb.FN << "\n";

    std::cout << "\n| Method    | TP | FP | TN | FN |  Acc | "
              << "PrecM | RecM | F1M | Avg-RT |\n"
              << "|-----------|----|----|----|----|------|"
              << "-------|-------|------|--------|\n";

    auto pr = [&](const std::string& name, Stat& s, long n_samples) {
        std::cout << '|' << std::setw(10) << std::left << name << '|'
                  << std::setw(4) << s.TP << '|'
                  << std::setw(4) << s.FP << '|'
                  << std::setw(4) << s.TN << '|'
                  << std::setw(4) << s.FN << '|'
                  << std::setw(6) << s.acc(n_samples) << '|'
                  << std::setw(7) << s.precM()        << '|'
                  << std::setw(7) << s.recM()         << '|'
                  << std::setw(6) << s.f1M()          << '|'
                  << std::setw(8) << std::setprecision(6) << s.avg(n_samples) << "|\n";
        std::cout.unsetf(std::ios::floatfield);
    };

    pr("Row"      , S_row , N);
    pr("Column"   , S_col , N);
    pr("Cost"     , S_cost, N);
    pr("Hybrid"   , S_hopt, N);
    pr("Kernel"   , S_fann, N);
    pr("AI Model" , S_lgb , N);
    pr("Oracle"   , S_opt , N);

    std::cout << "\n=== AI Model where Cost-Rule is CORRECT (" << N_cost_ok << " samples) ===\n";
    pr("AI_cost_OK",  S_ai_cost_ok , N_cost_ok );

    std::cout << "\n=== AI Model where Cost-Rule is WRONG (" << N_cost_bad << " samples) ===\n";
    pr("AI_cost_BAD", S_ai_cost_bad, N_cost_bad);

    double loss_runtime = (S_lgb.avg(N) - S_opt.avg(N)) / S_opt.avg(N);
    std::cout << "\nRuntime-loss (ratio): " << loss_runtime + 1.0 << "\n";
    std::cout << "\n[CSV] test_metrics.csv written (macro P/R/F1 only, "
              << "includes AI_cost_OK / AI_cost_BAD)\n";
}


/*───────────────────────────────────────────────────────────────
 *  build_sample_from_sql()    – 即时 EXPLAIN 生成单条 Sample
 *  build_samples_from_csv()   – 按题给 CSV 批量生成 Samples
 *───────────────────────────────────────────────────────────────*/
#ifndef BUILD_SAMPLE_HELPERS_DEFINED
#define BUILD_SAMPLE_HELPERS_DEFINED

/* —— 默认连接信息，可按需改 / 也可以从 CLI 传进来 —— */
const std::string DEFAULT_HOST = "127.0.0.1";
const int         DEFAULT_PORT = 44444;
const std::string DEFAULT_USER = "root";
const std::string DEFAULT_PASS = "";

/* ------------------------------------------------------------ */
Sample
build_sample_from_sql(const std::string& sql,
                      const std::string& db_name,
                      bool need_col_plans,
                      const std::string& host,
                      int               port,
                      const std::string& user,
                      const std::string& pass)
{
    /* ---------- 0) 连接 MySQL 并切库 ---------- */
    MYSQL* conn = mysql_init(nullptr);
    mysql_options(conn, MYSQL_OPT_RECONNECT, "1");
    if (!mysql_real_connect(conn, host.c_str(), user.c_str(), pass.c_str(),
                            nullptr, port, nullptr, 0))
        throw std::runtime_error(std::string("MySQL connect failed: ") +
                                 mysql_error(conn));
    if (!db_name.empty()) {
        std::string use = "USE `" + db_name + "`";
        if (mysql_query(conn, use.c_str()) != 0)
            throw std::runtime_error("USE " + db_name +
                                     " failed: " + mysql_error(conn));
    }

    if (mysql_query(conn, "set use_imci_engine=off;") != 0)
        throw std::runtime_error("set use_imci_engine=off failed.");

    /* ---------- 1) 获取行存计划 (EXPLAIN FORMAT=JSON) ---------- */
    std::string q = "EXPLAIN FORMAT=JSON " + sql;
    if (mysql_query(conn, q.c_str()) != 0)
        throw std::runtime_error("EXPLAIN failed: " + std::string(mysql_error(conn)));

    MYSQL_RES* res = mysql_store_result(conn);
    if (!res) throw std::runtime_error("Empty EXPLAIN result");
    MYSQL_ROW row = mysql_fetch_row(res);
    if (!row || !row[0]) throw std::runtime_error("Bad EXPLAIN row");
    json plan = json::parse(row[0]);      /* row[0] = TEXT JSON */
    mysql_free_result(res);

    /* ---------- 2) 由 plan→102-维特征 ---------- */
    Sample s;
    float feat[NUM_FEATS]{};
    double qcost = 0.0;
    Graph  dummy;                         /* 行存不用列计划，dummy 占位 */
    if (!plan2feat(plan, feat))
        throw std::runtime_error("plan2feat() failed; not a SELECT?");

    std::copy(std::begin(feat), std::end(feat), s.feat.begin());
    s.qcost   = qcost;
    s.row_t   = 0;            /* 真实 runtime 不知道，用 0 占位 */
    s.col_t   = 0;
    s.label   = -1;           /* unknown ground-truth */
    s.fann_pred   = -1;
    s.hybrid_pred = -1;

    /* ---------- 3) 如需要列计划，再取一次 EXPLAIN ---------- */
    if (need_col_plans)
    {
        /* PolarDB / MySQL 没有真正的 column store；这里给演示实现
           —— 如果你有列存 Hint / System，可在此替换为对应 EXPLAIN。 */
        std::string q2 = "/*+ SET_VAR(optimizer_switch='materialization=on') */ "
                         "EXPLAIN FORMAT=JSON " + sql;
        if (mysql_query(conn, q2.c_str()) == 0 &&
            (res = mysql_store_result(conn)))
        {
            if (MYSQL_ROW r2 = mysql_fetch_row(res))
            {
                try {
                    json colPlan = json::parse(r2[0] ? r2[0] : "{}");
                    ColStats& st = get_col_stats_for_dir(db_name.empty()
                                       ? "." : db_name);      // fake dir key
                    s.colGraph    = parseColPlan(colPlan, st, global_op2id());
                } catch (...) { /* ignore bad column EXPLAIN */ }
            }
            mysql_free_result(res);
        }
    }

    mysql_close(conn);
    return s;
}

/* ------------------------------------------------------------ */
std::vector<Sample>
build_samples_from_csv(const std::string& csv_path,
                       bool need_col_plan,
                       const std::string& host,
                       int               port,
                       const std::string& user,
                       const std::string& pass)
{
    std::ifstream fin(csv_path);
    if (!fin) { logE("Cannot open CSV: " + csv_path); return {}; }

    /* ---------- 1) 读取表头，确定各字段位置 ---------- */
    std::string header;
    std::getline(fin, header);                   // <-- 只读一次
    std::vector<std::string> cols;
    {   std::stringstream ss(header);
        std::string tok;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
    }
    auto idx = [&](const std::string& name)->int {
        auto it = std::find(cols.begin(), cols.end(), name);
        return (it == cols.end()) ? -1 : int(it - cols.begin());
    };
    const int iDB   = idx("database");
    const int iSQL  = idx("sqlQuery");
    const int iRT   = idx("row_time");
    const int iCT   = idx("col_time");
    const int iFANN = idx("fann_model_label");
    const int iHOPT = idx("hybrid_optimizer_label");

    if (iDB < 0 || iSQL < 0 || iRT < 0 || iCT < 0) {
        logE("CSV header missing mandatory columns"); return {};
    }

    /* ---------- 2) 统计总数据行数（不含 header） ---------- */
    std::streampos data_pos = fin.tellg();       // 记录“数据区”起始位置
    std::size_t total_line =
        std::count(std::istreambuf_iterator<char>(fin),
                   std::istreambuf_iterator<char>(), '\n');
    fin.clear();                                 // 清 EOF
    fin.seekg(data_pos);                         // 回到数据区开头

    /* ---------- 3) 正式逐行解析 ---------- */
    std::vector<Sample> out;
    std::string line;
    std::size_t ln = 0;

    while (std::getline(fin, line))
    {
        progress("predicting", ln, total_line);
        ++ln;

        /* —— 非严格 CSV 拆分，仅示例用 —— */
        std::vector<std::string> v; v.reserve(cols.size());
        {
            bool inq = false;  std::string cur;
            for (char c : line) {
                if (c == '"')          { inq = !inq; continue; }
                if (c == ',' && !inq) { v.push_back(cur); cur.clear(); }
                else                   cur.push_back(c);
            }
            v.push_back(cur);
        }
        if (int(v.size()) <= iSQL) continue;     // 坏行，跳过

        try {
            Sample s = build_sample_from_sql(
                           v[iSQL],           // sql
                           v[iDB],            // db
                           need_col_plan,
                           host, port, user, pass);

            s.row_t       = std::stod(v[iRT]);
            s.col_t       = std::stod(v[iCT]);
            s.label       = (s.col_t < s.row_t);
            s.fann_pred   = (iFANN>=0 && iFANN<int(v.size())) ? std::stoi(v[iFANN]) : -1;
            s.hybrid_pred = (iHOPT>=0 && iHOPT<int(v.size())) ? std::stoi(v[iHOPT]) : -1;

            out.emplace_back(std::move(s));
        }
        catch (const std::exception& e) {
            logW("CSV row " + std::to_string(ln) + ": " + e.what() + ", skipped");
        }
    }

    logI("Parsed " + std::to_string(out.size()) + " samples from CSV");
    return out;
}


#endif /* BUILD_SAMPLE_HELPERS_DEFINED */