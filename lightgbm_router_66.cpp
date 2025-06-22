/*****************************************************************************************
 * lightgbm_router.cpp  –  63-feature LightGBM regressor (row-vs-column decision)
 *            ♦ portable 〈filesystem〉 shim
 *            ♦ identical feature extraction as train_dtree_enhanced.cpp
 *            ♦ LightGBM C-API (“regression”) → choose column when ŷ > 0
 *
 * Build (example):
 *   g++ -O3 -std=c++17 -I$LIGHTGBM/include \
 *       lightgbm_router.cpp -o lightgbm_router \
 *       -L$LIGHTGBM/lib -l_lightgbm -lmysqlclient -pthread -lstdc++fs
 *
 * Run:
 *   ./lightgbm_router --data_dirs=airline,hepatitis --base=/home/wuy/query_costs
 *****************************************************************************************/
#include <bits/stdc++.h>
#include <algorithm>

/* ---------- portable <filesystem> ---------- */
#if __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #error "Your compiler/libstdc++ lacks <filesystem>"
#endif
/* ------------------------------------------- */

#include <regex>
#include <mysql/mysql.h>
#include "json.hpp"
#include <LightGBM/c_api.h>

using json = nlohmann::json;
using namespace std;

/* ───────────────── helpers ───────────────── */
constexpr int    NUM_FEATS   = 66;
constexpr double EPS_RUNTIME = 1e-6;

static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }

static void progress(const string&tag,size_t cur,size_t tot,size_t W=40){
    double f=tot?double(cur)/tot:1.0; size_t filled=size_t(f*W);
    cerr<<"\r"<<tag<<" ["<<string(filled,'=')<<string(W-filled,' ')
        <<"] "<<setw(3)<<int(f*100)<<"% ("<<cur<<'/'<<tot<<')'<<flush;
    if(cur==tot) cerr<<'\n';
}

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

struct Agg {
    /* ------------ existing numeric accumulators ------------ */
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0,
           selSum=0,selMin=1e30,selMax=0,
           ratioSum=0,ratioMax=0,maxPrefix=0,minRead=1e30,
           fanoutMax=0;
    int    cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,
           idxUse=0,sumPK=0,coverCount=0,maxDepth=0;

    /* ------------ existing flags ------------ */
    bool   grp=false, ord=false, tmp=false;

    /* ------------ NEW flags (join-buffer / temp+filesort) -- */
    bool   jb=false;          // any  "using_join_buffer"
    bool   tf=false;          // temp table  or  filesort
};

/* Recursive DFS through a plan node */
static void walk(const json& n, Agg& a, int depth = 1)
{
    if (n.is_object())
    {
        /* ---- table node ------------------------------------------------ */
        if (n.contains("table") && n["table"].is_object())
        {
            const auto& t  = n["table"];
            const auto& ci = t.value("cost_info", json::object());

            /* --- numeric counters (unchanged) --- */
            double re = safe_f(t  , "rows_examined_per_scan");
            double rp = safe_f(t  , "rows_produced_per_join");
            double fl = safe_f(t  , "filtered");
            double rc = safe_f(ci , "read_cost");
            double ec = safe_f(ci , "eval_cost");
            double pc = safe_f(ci , "prefix_cost");
            double dr = ci.contains("data_read_per_join") &&
                        ci["data_read_per_join"].is_string()
                        ? str_size_to_num(ci["data_read_per_join"].get<string>())
                        : safe_f(ci, "data_read_per_join");

            a.re += re;  a.rp += rp;  a.f  += fl;
            a.rc += rc;  a.ec += ec;  a.pc += pc;  a.dr += dr;
            a.cnt++;

            a.maxPrefix = std::max(a.maxPrefix, pc);
            a.minRead   = std::min(a.minRead , rc);

            if (re > 0) {
                double sel = rp / re;
                a.selSum  += sel;
                a.selMin   = std::min(a.selMin, sel);
                a.selMax   = std::max(a.selMax, sel);
                a.fanoutMax= std::max(a.fanoutMax, sel);
            }
            if (ec > 0) {
                double ratio = rc / ec;
                a.ratioSum += ratio;
                a.ratioMax  = std::max(a.ratioMax, ratio);
            }

            /* --- access-type counters (unchanged) --- */
            std::string at = t.value("access_type", "ALL");
            if      (at == "range")   ++a.cRange;
            else if (at == "ref")     ++a.cRef;
            else if (at == "eq_ref")  ++a.cEq;
            else if (at == "index")   ++a.cIdx;
            else                      ++a.cFull;

            if (getBool(t, "using_index"))         ++a.idxUse;
            if (t.contains("possible_keys") &&
                t["possible_keys"].is_array())
                a.sumPK += int(t["possible_keys"].size());

            /* covering-index check (unchanged) */
            if (t.contains("used_columns") && t["used_columns"].is_array() &&
                t.contains("key") && t["key"].is_string())
            {
                std::string idx = t["key"];
                auto it = indexCols.find(idx);
                if (it != indexCols.end()) {
                    bool cover = true;
                    for (auto& u : t["used_columns"])
                        if (!u.is_string() ||
                            !it->second.count(u.get<std::string>()))
                        { cover = false; break; }
                    if (cover) ++a.coverCount;
                }
            }

            /* ---------- NEW FLAG: join-buffer ---------- */
            if (getBool(t, "using_join_buffer"))
                a.jb = true;
        }

        /* ------------- higher-level flags ------------- */
        if (n.contains("grouping_operation"))
            a.grp = true;

        if (n.contains("ordering_operation") || getBool(n, "using_filesort")) {
            a.ord = true;
            a.tf  = true;          // filesort implies temp+file flag
        }
        if (getBool(n, "using_temporary_table")) {
            a.tmp = true;
            a.tf  = true;
        }

        /* recurse into children (skip “table” key just visited) */
        for (auto& kv : n.items())
            if (kv.key() != "table")
                walk(kv.value(), a, depth + 1);
    }
    else if (n.is_array())
        for (auto& v : n)
            walk(v, a, depth);

    /* deepest depth seen so far */
    a.maxDepth = std::max(a.maxDepth, depth);
}


/* ---------- 取消 clip 的对数函数 ---------- */
static inline double lp(double v) { return std::log1p(std::max(0.0, v)); }


static bool plan2feat(const json& plan, float f[NUM_FEATS])
{
    if (!plan.contains("query_block")) return false;
    const json* qb = &plan["query_block"];

    /* unwrap UNION – keep behaviour identical */
    if (qb->contains("union_result")) {
        const auto& spec = (*qb)["union_result"]["query_specifications"];
        if (spec.is_array() && !spec.empty())
            qb = &spec[0]["query_block"];
    }

    Agg a;  walk(*qb, a);
    if (!a.cnt) return false;

    const double inv = 1.0 / a.cnt;
    int    k      = 0;
    double qCost  = safe_f(qb->value("cost_info", json::object()), "query_cost");
    double rootRow= safe_f(*qb, "rows_produced_per_join");

    auto lp = [](double v){ return std::log1p(std::max(0.0, v)); };
#define PUSH(x)  f[k++] = static_cast<float>(x)

    /* --------- 0-62: ***unchanged*** feature order --------- */
    /* 0-6 basics */               PUSH(lp(a.re*inv)); PUSH(lp(a.rp*inv)); PUSH(lp(a.f*inv));
                                   PUSH(lp(a.rc*inv)); PUSH(lp(a.ec*inv)); PUSH(lp(a.pc*inv)); PUSH(lp(a.dr*inv));
    /* 7-12 access */              PUSH(a.cRange*inv); PUSH(a.cRef*inv); PUSH(a.cEq*inv);
                                   PUSH(a.cIdx*inv);   PUSH(a.cFull*inv); PUSH(a.idxUse*inv);
    /* 13-17 selectivity/shape */  PUSH(a.selSum*inv); PUSH(a.selMin); PUSH(a.selMax);
                                   PUSH(a.maxDepth);   PUSH(a.fanoutMax);
    /* 18-20 flags */              PUSH(a.grp); PUSH(a.ord); PUSH(a.tmp);
    /* 21-22 ratio stats */        PUSH(a.ratioSum*inv); PUSH(a.ratioMax);
    /* 23-24 cost & rows */        PUSH(lp(qCost)); PUSH(lp(rootRow));
    /* 25-27 cost ratios */        PUSH(lp((a.pc*inv)/std::max(1e-6,a.rc*inv)));
                                   PUSH(lp((a.rc*inv)/std::max(1e-6,a.re*inv)));
                                   PUSH(lp((a.ec*inv)/std::max(1e-6,a.re*inv)));
    /* 28-31 misc */               PUSH(a.cnt==1); PUSH(a.cnt>1);
                                   PUSH(lp(a.maxDepth*(a.idxUse*inv)));
                                   PUSH(lp((a.idxUse*inv)/std::max(a.cFull*inv,1e-3)));
    /* 32-39 misc-2 */             PUSH(a.cnt);
                                   PUSH(a.cnt?double(a.sumPK)/a.cnt:0);
                                   PUSH(lp(a.maxPrefix));
                                   PUSH(lp(a.minRead<1e30?a.minRead:0));
                                   PUSH(a.cnt>1?double(a.cnt-1)/a.cnt:0);
                                   PUSH(rootRow>0?double(a.re)/rootRow:0);
                                   PUSH(a.selMax-a.selMin);
                                   PUSH(a.idxUse/double(std::max(1,a.cRange+a.cRef+a.cEq+a.cIdx)));
    /* 40-43 cover & big cost */   PUSH(qCost);
                                   PUSH(qCost>5e4);
                                   PUSH(a.cnt?double(a.coverCount)/a.cnt:0);
                                   PUSH(a.coverCount==a.cnt);
    /* 44-46 diff & counts */      PUSH(lp(a.re*inv)-lp(a.selSum*inv));
                                   PUSH(a.cnt);
                                   PUSH(lp(a.cnt));
    /* 47-50 PK/cover stats */     PUSH(a.sumPK);
                                   PUSH(a.cnt?double(a.sumPK)/a.cnt:0);
                                   PUSH(a.coverCount);
                                   PUSH(a.cnt?double(a.coverCount)/a.cnt:0);
    /* 51-56 access share again */ PUSH(a.idxUse*inv); PUSH(a.cRange*inv); PUSH(a.cRef*inv);
                                   PUSH(a.cEq*inv);    PUSH(a.cIdx*inv);  PUSH(a.cFull*inv);
    /* 57-59 prefix/read */        PUSH(lp(a.maxPrefix*inv));
                                   PUSH(lp(a.minRead<1e30?a.minRead:0));
                                   PUSH(a.selMax-a.selMin);
    /* 60-62 extremes */           PUSH(a.ratioMax); PUSH(a.fanoutMax);
                                   PUSH(a.selMin>0?double(a.selMax/a.selMin):0);

    /* --------- 63-65: NEW FEATURES (booleans + padding) ---- */
    PUSH(a.jb);        // 63  join-buffer used
    PUSH(a.tf);        // 64  temp-table OR filesort present
    PUSH(0.0f);        // 65  reserved / always 0

    return k == NUM_FEATS;        // NUM_FEATS should be 66
#undef PUSH
}

/* ─────────── sample struct & loaders ─────────── */
struct Sample {
    array<float, NUM_FEATS> feat{};
    int    label = 0;
    double row_t = 0, col_t = 0;
    double qcost = 0;
    int    fann_pred   = -1;
    int    hybrid_pred = -1;

    /* NEW */
    bool   joinbuf  = false;
    bool   tempfile = false;
};

static bool load_plan_file(const string&path,array<float,NUM_FEATS>&f,double&q){
    ifstream in(path); if(!in) return false; json j; try{ in>>j;}catch(...){return false;}
    if(!plan2feat(j,f.data())) return false;
    q= j.contains("query_block") ? safe_f(j["query_block"].value("cost_info",json::object()),"query_cost") : 0;
    return true;
}

static vector<Sample> load_dataset(const string&root,
                                   const vector<string>&dirs)
{
    vector<Sample> DS;

    for(const auto& d : dirs){
        string csv = root + "/" + d + "/query_costs.csv";
        fs::path planDir = root + "/" + d + "/row_plans";
        if(!fs::exists(csv) || !fs::is_directory(planDir)){
            logW("skip "+d); continue;
        }

        /* ---------- 读 meta ---------- */
        struct MetaRec{ int lab; double rt, ct; int fann, hybrid; };
        unordered_map<string,MetaRec> meta;

        ifstream fin(csv); string line;
        getline(fin,line);                     // 丢掉表头
        while(getline(fin,line)){
            stringstream ss(line);
            string qid,lab,rt,ct,fann,hyb;
            getline(ss,qid,','); getline(ss,lab,',');
            getline(ss,rt,',');  getline(ss,ct,',');
            getline(ss,hyb,',');getline(ss,fann,',');

            meta[qid] = {
                lab=="1",
                rt.empty()?0:stod(rt),
                ct.empty()?0:stod(ct),
                fann.empty()? -1 : stoi(fann),
                hyb .empty()? -1 : stoi(hyb )
            };
        }

        /* ---------- 扫 plan ---------- */
        vector<fs::path> files;
        for(auto&p:fs::directory_iterator(planDir))
            if(p.path().extension()==".json") files.push_back(p.path());

        size_t cur = 0, tot = files.size();
        for(auto& pf : files){
            ++cur; if(cur%5000==0) progress("scan "+d,cur,tot);

            string qid = pf.stem().string();
            auto it = meta.find(qid);
            if(it == meta.end()) continue;

            Sample s;
            if(!load_plan_file(pf.string(),s.feat,s.qcost)) continue;
            s.label      = it->second.lab;
            s.row_t      = it->second.rt;
            s.col_t      = it->second.ct;
            s.fann_pred  = it->second.fann;
            s.hybrid_pred= it->second.hybrid;
            DS.push_back(std::move(s));
        }
    }

    logI("Loaded "+to_string(DS.size())+" samples");
    return DS;
}

template <typename T>
static inline T my_clamp(T v, T lo, T hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

/*********************************************************************
 * gap-regression LightGBM router  ·  train_and_eval (no feature_weights)
 *********************************************************************/
static void train_and_eval(const std::vector<Sample>& DS,
                           const std::string&  model_path,
                           int     num_trees      = 400,
                           int     max_depth      = 10,
                           double  lr             = 0.06,
                           double  subsample      = 0.7,
                           double  colsample      = 0.8,
                           bool    skip_train     = false,
                           int     num_threads    = 100 /* 0 ⇒ auto */ )
{
    /* ---------- constants ---------- */
    constexpr double COST_THR = 5e4;   // MySQL cost rule
    constexpr double GAP_THR  = 0.0;   // disable fallback

    /* ---------- build VALID subset ---------- */
    std::vector<const Sample*> S;
    for (const auto& s : DS)
        if (s.hybrid_pred != -1 && s.fann_pred != -1 &&
            s.row_t > 0 && s.col_t > 0)
            S.push_back(&s);

    const int N = static_cast<int>(S.size());
    if (!N) { logE("no sample has full information"); return; }

    if (num_threads <= 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    /* ---------- allocate LightGBM matrices ---------- */
    std::vector<float>  X(static_cast<size_t>(N) * NUM_FEATS), y(N), w(N);
    double P = 0, N0 = 0;
    for (auto p : S) p->label ? ++P : ++N0;
    const double w_pos = P  ? N / (2 * P)  : 1.0;
    const double w_neg = N0 ? N / (2 * N0) : 1.0;

    for (int i = 0; i < N; ++i) {
        const Sample& s = *S[i];

        /* feature matrix ------------------------------------------------- */
        for (int j = 0; j < NUM_FEATS; ++j)
            X[i * NUM_FEATS + j] = s.feat[j];

        // /* regression label ------------------------------------------------ */
        double rt = std::max(s.row_t, EPS_RUNTIME);
        double ct = std::max(s.col_t, EPS_RUNTIME);
        y[i] = float(std::log(rt) - std::log(ct));   // Δ = ln(rt/ct)

        // /* base sample-weight (class-balance × gap × cost) ---------------- */
        // double gap     = std::fabs(rt - ct);
        // double w_gap   = 1.0 + std::log1p(gap);          // ≥ 1
        // double w_cost  = std::log1p(s.qcost);            // ≥ 0
        // double base_w  = (s.label ? w_pos : w_neg) * w_gap * w_cost;

        // /* feature-boost  (fanoutMax & ratioMax) --------------------------- */
        // double fanout  = s.feat[22];      // fanoutMax  (原 feat[22])
        // double ratioMx = s.feat[60];      // ratioMax   (原 feat[60])
        // /* tanh 抑制极端值；权重范围大致 1.0 – 3.0 */
        // double boost   = 1.0 + 2.0 * std::tanh(0.5 * fanout)
        //                       + 1.5 * std::tanh(0.5 * ratioMx);

        // w[i] = float(base_w * boost);

        // double base = (s.label ? w_pos : w_neg);

        // double sel_boost = 1.0
        //          + 1.0 * std::min<double>(3.0,  static_cast<double>(s.feat[22]))  // fanoutMax
        //          + 0.8 * std::min<double>(3.0,  static_cast<double>(s.feat[60])); // ratioMax


        // /* —— “巨大 qcost + fanout” 的 corner case再提权 —— */
        // double corner = (s.label && s.qcost > 1e5 && s.feat[22] > 1.0) ? 2.0 : 1.0;

        // w[i] = float(base * w_gap * sel_boost * corner);

        // -------------  inside the training loop  -------------
        // double base = (s.label ? w_pos : w_neg);          // class balance

        // double gap_ln  = std::log(rt) - std::log(ct);
        // double w_gap   = 1.0 + gap_ln * gap_ln;           // runtime gap (≥1)

        // double fanout     = static_cast<double>(s.feat[22]);      // fanoutMax
        // double ratio_max  = static_cast<double>(s.feat[60]);      // ratioMax
        // double idx_share  = static_cast<double>(s.feat[37]);      // idxUse / (range+ref+eq_ref+idx)

        // /* -------- positive-corner (column wins) --------------- */
        // double pos_corner = (s.label == 1 &&
        //                     s.qcost  > 1e5   &&   // very large estimate
        //                     fanout   > 1.0)  ? 2.0 : 1.0;   // wide fan-out

        // /* -------- negative-corner (row wins) ------------------ */
        // double neg_corner = (s.label == 0 &&
        //                     s.qcost  > 1e5 &&          // same huge estimate
        //                     fanout   < 1.1 &&          // but very narrow joins
        //                     idx_share > 0.7) ? 2.0 : 1.0;

        // /* -------- cost mis-estimate penalty ------------------- */
        // double underestimate_pen = (s.label == 1 && s.qcost < 1e4) ? 4.0 : 1.0;

        // /* -------- smoothed qcost factor ----------------------- */
        // double qcost_norm = std::log1p(s.qcost);          // grows slowly

        // double w_total = base * w_gap
        //             * (1.0 + std::min(3.0, fanout)   + 0.8*std::min(3.0,ratio_max))
        //             * qcost_norm
        //             * pos_corner * neg_corner * underestimate_pen;

        // /* clamp outliers */
        // w_total = std::min<double>(w_total, 1e6);
        // w[i] = static_cast<float>(w_total);

        // ---------- prepare shorthand ----------
        // double base   = (s.label ? w_pos : w_neg);        // 类别平衡
        // double q_ln   = std::log1p(s.qcost);              // 0–14

        // double w_i = base * (1.0 + 0.2 * q_ln);           // ① 主体
        // /* ② 温和列优势提权 (add-on) */
        // double fan   = std::min<double>(3.0, s.feat[22]);
        // double ratio = std::min<double>(3.0, s.feat[60]);
        // if (fan   > 1.5) w_i += 0.8 * base * fan;
        // if (ratio > 1.3) w_i += 0.5 * base * ratio;
        // /* ③ credit corner case */
        // if (fan > 2.5 && s.qcost < 8e3)
        //     w_i += 2.0 * base;

        // w[i] = static_cast<float>(w_i);

        // double base   = (s.label ? w_pos : w_neg);          // 类别平衡
        // double q_ln   = std::log1p(s.qcost);                // ~0–14

        // double w_i = base * (1.0 + 0.20 * q_ln);            // 主体 (×0.2)

        // // ---------- 列优势软加成 ----------
        // double fan   = std::min<double>(3.0, s.feat[22]);           // fanoutMax
        // double ratio = std::min<double>(3.0, s.feat[60]);           // ratioMax
        // if (fan   > 1.3) w_i += 0.6 * base * (fan   - 1.2);
        // if (ratio > 1.1) w_i += 0.4 * base * (ratio - 1.0);

        // // ---------- prefix_cost / read_cost 曝露更明显 ----------
        // double pc_rc = std::exp( std::min<double>(2.0, s.feat[25]) );   // Feat25 ≈ ln(pc/rc)
        // if (pc_rc > 2.0 && s.qcost < 1e4)
        //     w_i += 0.8 * base * (pc_rc - 1.5);

        // // ---------- credit-type corner case仍保留 ----------
        // if (fan > 2.5 && s.qcost < 8e3)
        //     w_i += 2.0 * base;

        // w[i] = static_cast<float>(w_i);

        /* ---------- inside the loop that fills w[i] ---------- */
        /* ---------- ① 先算一个“中性”权重 ---------------- */
        // 基础：类别平衡 × gap² × log1p(qcost)
        double gap_ln  = std::log(rt) - std::log(ct);          // Δ = ln(rt/ct)
        double w_base  = (s.label ? w_pos : w_neg);
        double w_gap   = 1.0 + gap_ln * gap_ln;                // ≥1
        double w_qc    = 1.0 + 0.15 * std::log1p(s.qcost);     // 1–3（平滑）

        double w_i = w_base * w_gap * w_qc;

        /* ---------- ② 针对「小 qcost 但列快」的提权 ------- */
        double fan    = std::min<double>(4.0, s.feat[22]);     // fanoutMax
        double ratioM = std::min<double>(4.0, s.feat[60]);     // ratioMax
        bool   heavyJ = (s.feat[63] > 0.5f) ||                 // join-buffer
                        (s.feat[64] > 0.5f);                  // temp/filesort

        if (s.label == 1 &&             // 真·列快
            s.qcost   < 1e4 &&          // Optimizer 低估
            (fan > 1.3 || ratioM > 1.2 || heavyJ))
        {
            double boost = 1.5 +
                        0.4 * (fan    - 1.0) +              // 最多 +1.2
                        0.3 * (ratioM - 1.0) +              // 最多 +1.2
                        (heavyJ ? 0.5 : 0.0);               // +0.5
            w_i *= std::min(4.0, boost);
        }

        /* ---------- ③ 针对「大 qcost 但行快」的惩罚 ------- */
        double idx_share = s.feat[37];                         // idxUse / (range+ref+…)
        if (s.label == 0 &&               // 真·行快
            s.qcost  > 5e4  &&            // Optimizer 高估
            fan      < 1.2  &&            // 无爆炸 fan-out
            idx_share > 0.6)
        {
            w_i *= 1.8;                   // 适度上调，促使模型记住
        }

        /* ---------- ④ join-buffer / temp-table 软加成 ---- */
        double jb_boost = (s.feat[63] > 0.5f) ? 1.25 : 1.0;
        double tf_boost = (s.feat[64] > 0.5f) ? 1.20 : 1.0;

        w_i *= jb_boost * tf_boost;

        /* ---------- ⑤ clamp 防炸 ------------------------- */
        w_i = my_clamp(w_i, 1e-4, 1e7);

        w[i] = static_cast<float>(w_i);




    }

    /* ---------- create / load LightGBM booster ---------- */
    BoosterHandle booster = nullptr;
    if (skip_train) {
        int iters = 0;
        if (LGBM_BoosterCreateFromModelfile(model_path.c_str(),
                                            &iters, &booster))
        { logE("model load failed"); return; }
        logI("Loaded model ← " + model_path +
             "  (" + std::to_string(iters) + " trees)");
    } else {
        DatasetHandle dtrain;
        if (LGBM_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT32,
                                      N, NUM_FEATS, 1, "", nullptr, &dtrain))
        { logE("DatasetCreate failed"); return; }

        LGBM_DatasetSetField(dtrain, "label" , y.data(), N, C_API_DTYPE_FLOAT32);
        LGBM_DatasetSetField(dtrain, "weight", w.data(), N, C_API_DTYPE_FLOAT32);

        std::string param =
            "objective=fair fair_c=0.5"
            " max_bin=127 num_leaves=256"
            " max_depth="        + std::to_string(max_depth)   +
            " num_iterations="   + std::to_string(num_trees)   +
            " learning_rate="    + std::to_string(lr)          +
            " feature_fraction=" + std::to_string(colsample)   +
            " bagging_fraction=" + std::to_string(subsample)   +
            " bagging_freq=1"
            " lambda_l2=1.0"
            " num_threads="      + std::to_string(num_threads) +
            " verbosity=-1";



        // std::string param =
        //     "objective=huber alpha=0.9"
        //     " max_bin=127 num_leaves=256"
        //     " max_depth="        + std::to_string(max_depth)   +
        //     " num_iterations="   + std::to_string(num_trees)   +
        //     " learning_rate="    + std::to_string(lr)          +
        //     " feature_fraction=" + std::to_string(colsample)   +
        //     " bagging_fraction=" + std::to_string(subsample)   +
        //     " bagging_freq=1"
        //     " lambda_l2=1.0"
        //     " num_threads="      + std::to_string(num_threads) +
        //     " verbosity=-1";


        // std::string param =
        //     "objective=regression metric=l2"
        //     " max_bin=127 num_leaves=256"
        //     " max_depth="        + std::to_string(max_depth)   +
        //     " num_iterations="   + std::to_string(num_trees)   +
        //     " learning_rate="    + std::to_string(lr)          +
        //     " feature_fraction=" + std::to_string(colsample)   +
        //     " bagging_fraction=" + std::to_string(subsample)   +
        //     " bagging_freq=1"
        //     " lambda_l2=1.0"
        //     " num_threads="      + std::to_string(num_threads) +
        //     " verbosity=-1";

        if (LGBM_BoosterCreate(dtrain, param.c_str(), &booster))
        { logE("BoosterCreate failed"); return; }

        for (int it = 0, fin; it < num_trees; ++it) {
            LGBM_BoosterUpdateOneIter(booster, &fin);
            if ((it + 1) % 10 == 0 || it + 1 == num_trees)
                progress("train", it + 1, num_trees);
        }
        std::cerr << '\n';
        LGBM_BoosterSaveModel(booster, 0, -1, 0, model_path.c_str());
        logI("Model saved → " + model_path);
        LGBM_DatasetFree(dtrain);
    }

    /* ---------- prediction ---------- */
    std::vector<double> pred(N); int64_t out_len = 0;
    if (LGBM_BoosterPredictForMat(booster, X.data(), C_API_DTYPE_FLOAT32,
                                  N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
                                  -1, 0, "", &out_len, pred.data()))
    { logE("PredictForMat failed"); LGBM_BoosterFree(booster); return; }

    /* ---------- metrics ---------- */
    int TP=0,FP=0,TN=0,FN=0;
    double r_row=0,r_col=0,r_opt=0,
           r_cost=0,r_hopt=0,r_fann=0,r_lgb=0;

    for (int i = 0; i < N; ++i) {
        const Sample& s = *S[i];

        bool col = pred[i] > 0.0;                 // no fallback (GAP_THR=0)

        (col ? (s.label?++TP:++FP) : (s.label?++FN:++TN));

        r_row  += s.row_t;
        r_col  += s.col_t;
        r_opt  += std::min(s.row_t, s.col_t);
        r_cost += (s.qcost > COST_THR) ? s.col_t : s.row_t;
        r_hopt += s.hybrid_pred ? s.col_t : s.row_t;
        r_fann += s.fann_pred   ? s.col_t : s.row_t;
        r_lgb  += col           ? s.col_t : s.row_t;
    }

    auto avg = [&](double s){ return s / N; };

    std::cout << "\n*** EVALUATED ON " << N << " COMPLETE SAMPLES ***\n";
    std::cout << std::fixed << std::setprecision(6)
              << "\n=== CONFUSION (LightGBM) ===\nTP=" << TP
              << " FP=" << FP << " TN=" << TN << " FN=" << FN
              << "\nAccuracy=" << double(TP+TN)/N
              << "  BalAcc="  << 0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
              << "  F1="      << (TP?2.0*TP/(2*TP+FP+FN):0.) << '\n';

    std::cout << "\n=== AVG RUNTIME (s) ===\n"
              << "Row only            : " << avg(r_row)  << '\n'
              << "Column only         : " << avg(r_col)  << '\n'
              << "Cost threshold rule : " << avg(r_cost) << '\n'
              << "Hybrid optimizer    : " << avg(r_hopt) << '\n'
              << "FANN model          : " << avg(r_fann) << '\n'
              << "LightGBM            : " << avg(r_lgb)  << '\n'
              << "Optimal (oracle)    : " << avg(r_opt)  << "\n\n";

    LGBM_BoosterFree(booster);
}




/* ─────────── CLI driver ─────────── */
int main(int argc,char*argv[]){
    string base="/home/wuy/query_costs", model="lgb_model.txt";
    vector<string> dirs;
    int trees=400, depth=10; double lr=0.06, subs=0.7, col=0.8;
    string host="127.0.0.1", user="root", pass=""; int port=44444; bool skip=false;

    for(int i=1;i<argc;++i){
        string a(argv[i]);
        if(a.rfind("--data_dirs=",0)==0){ string s=a.substr(12),t; stringstream ss(s); while(getline(ss,t,',')) dirs.push_back(t);}
        else if(a.rfind("--base=",0)==0) base=a.substr(7);
        else if(a.rfind("--trees=",0)==0) trees=stoi(a.substr(8));
        else if(a.rfind("--max_depth=",0)==0) depth=stoi(a.substr(12));
        else if(a.rfind("--lr=",0)==0) lr=stod(a.substr(5));
        else if(a.rfind("--subsample=",0)==0) subs=stod(a.substr(12));
        else if(a.rfind("--colsample=",0)==0) col =stod(a.substr(12));
        else if(a.rfind("--model=",0)==0) model=a.substr(8);
        else if(a=="--skip_train") skip=true;
        else if(a.rfind("--mysql_host=",0)==0) host=a.substr(13);
        else if(a.rfind("--mysql_port=",0)==0) port=stoi(a.substr(13));
        else if(a.rfind("--mysql_user=",0)==0) user=a.substr(13);
        else if(a.rfind("--mysql_pass=",0)==0) pass=a.substr(13);
    }
    if(dirs.empty()){ logE("need --data_dirs=..."); return 1; }

    load_all_index_defs(host,port,user,pass,dirs);
    auto DS=load_dataset(base,dirs);
    if(DS.size()<100){ logE("too few samples"); return 1; }

    train_and_eval(DS,model,trees,depth,lr,subs,col,skip);
    return 0;
}
