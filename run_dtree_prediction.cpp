//------------------------------------------------------------------------------
// run_dtree_prediction.cpp  ―  mirrors kernel feature extractor exactly
//------------------------------------------------------------------------------

#include <mysql/mysql.h>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>


using json = nlohmann::json;

/* ────────────── logging helpers ────────────── */
static void logInfo (const std::string& s){ std::cout  << "[INFO]  " << s << '\n'; }
static void logWarn (const std::string& s){ std::cerr  << "[WARN]  " << s << '\n'; }
static void logError(const std::string& s){ std::cerr  << "[ERROR] " << s << '\n'; }


constexpr int    NUM_FEATS    = 43;
constexpr double DEFAULT_TIME = 60.0;

/* 0 … no emphasis, 1 … linear, 2 … stronger, etc.                      */
constexpr double GAP_EMPHASIS = 2;   // you can tune this via CLI later


// Load a JSON file of the form:
// {
//   "table1": [ ["colA","colB"], ["colC"] ],
//   "table2": [ ["x","y","z"] ]
// }


/* ────────────── node (de)serialisation ────────────── */
struct DTNode{
    int   feat=-1,left=-1,right=-1; float thresh=0.f,prob=0.f;
};
inline void to_json (json& j, const DTNode& n){
    j = json{{"feat",n.feat},{"thresh",n.thresh},
             {"left",n.left},{"right",n.right},{"prob",n.prob}};
}
inline void from_json(const json& j, DTNode& n){
    j.at("feat").get_to(n.feat);   j.at("thresh").get_to(n.thresh);
    j.at("left").get_to(n.left);   j.at("right").get_to(n.right);
    j.at("prob").get_to(n.prob);
}

/* ────────────── tolerant numeric extractors ────────────── */
static double safe_f(const json& v){
    if(v.is_number()) return v.get<double>();
    if(v.is_string()){
        try{ return std::stod(v.get<std::string>()); }catch(...){}
    }
    return 0.0;
}

static double safe_f(const json& obj, const char* key){   // ← add back
    if(!obj.contains(key)) return 0.0;
    return safe_f(obj[key]);          // delegate to the 1-arg version
}
static inline double log1p_clip(double v){ return std::log1p(std::max(0.0,v)); }

/* -------- new: convert strings like "17.4M" / "2.3K" into numbers -------- */
static double convert_data_size_to_numeric(std::string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    if(s.empty()) return 0.0;
    double f=1.0; char suf=s.back();
    if(suf=='G'||suf=='g'){ f=1e9; s.pop_back(); }
    else if(suf=='M'||suf=='m'){ f=1e6; s.pop_back(); }
    else if(suf=='K'||suf=='k'){ f=1e3; s.pop_back(); }
    try{ return std::stod(s)*f; }catch(...){ return 0.0; }
}

/* ────────────── 32-d feature extractor (identical to kernel) ────────────── */
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0;
    double selSum=0,selMin=1e30,selMax=0,ratioSum=0,ratioMax=0;
    int cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,maxDepth=0;
    double fanoutMax=0; bool grp=0,ord=0,tmp=0,hasLimit=0,hasDistinct=0,hasUnion=0,hasConst=0;
    int numUnion=0;
    int    sumPossibleKeys = 0;      // total strings in `possible_keys` arrays
    double maxPrefix       = 0.0;    // maximum prefix_cost encountered
    double minRead         = 1e30;   // minimum  read_cost   encountered
    int    coverCount      = 0; // # of tables where used_columns ⊆ possible_keys
};
static bool getBool(const json& o,const char* k){
    if(!o.contains(k)) return false;
    const auto& v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){
        std::string s=v.get<std::string>(); std::transform(s.begin(),s.end(),s.begin(),::tolower);
        return s=="yes"||s=="true"||s=="1";
    }
    return false;
}

/* ❶  DFS over the plan tree – fills the Agg struct */
static void walk(const json &n, Agg &a, int depth)
{
    /* ---------- visit current node -------------------------------- */
    if (n.is_object())
    {
        if (n.contains("table") && n["table"].is_object())
        {
            const auto &t  = n["table"];
            const auto &ci = t.value("cost_info", json::object());

            /* numeric columns (all via safe_f(obj,key)!) */
            double re = safe_f(t , "rows_examined_per_scan");
            double rp = safe_f(t , "rows_produced_per_join");
            double fl = safe_f(t , "filtered");

            double rc = safe_f(ci, "read_cost");
            double ec = safe_f(ci, "eval_cost");
            double pc = safe_f(ci, "prefix_cost");

            double dr = 0.0;
            if (ci.contains("data_read_per_join"))
            {
                const auto &v = ci["data_read_per_join"];
                dr = v.is_string()
                       ? convert_data_size_to_numeric(v.get<std::string>())
                       : safe_f(v);
            }

            /* accumulate --------------------------------------------------- */
            a.re += re;  a.rp += rp;  a.f  += fl;
            a.rc += rc;  a.ec += ec;  a.pc += pc;  a.dr += dr;
            a.cnt++;

            /* ---- collect stats for new features ---- */
            if(t.contains("possible_keys") && t["possible_keys"].is_array())
                a.sumPossibleKeys += int(t["possible_keys"].size());
            a.maxPrefix = std::max(a.maxPrefix, pc);
            a.minRead   = std::min(a.minRead  , rc);

            if (re > 0)
            {
                double sel = rp / re;
                a.selSum   += sel;
                a.selMin    = std::min(a.selMin,  sel);
                a.selMax    = std::max(a.selMax,  sel);
                a.fanoutMax = std::max(a.fanoutMax, sel);
            }

            double ratio = (ec > 0.0) ? rc / ec : rc;
            a.ratioSum  += ratio;
            a.ratioMax   = std::max(a.ratioMax, ratio);

            /* access-type counters --------------------------------------- */
            std::string at = t.value("access_type", "ALL");
            if (at == "const")  a.hasConst = true;
            if (at == "range")       a.cRange++;
            else if (at == "ref")    a.cRef++;
            else if (at == "eq_ref") a.cEq++;
            else if (at == "index")  a.cIdx++;
            else                     a.cFull++;

            if (getBool(t, "using_index")) a.idxUse++;
            // ─── NEW: detect when the optimizer’s used_columns are fully covered by possible_keys
            if (t.contains("used_columns") && t["used_columns"].is_array()
             && t.contains("possible_keys") && t["possible_keys"].is_array()) {
                std::unordered_set<std::string> pk;
                for (auto &x : t["possible_keys"])
                    if (x.is_string()) pk.insert(x.get<std::string>());

                bool full = true;
                for (auto &u : t["used_columns"]) {
                    if (!u.is_string() || !pk.count(u.get<std::string>())) {
                        full = false;
                        break;
                    }
                }
                if (full) a.coverCount++;
            }
        }

        /* flags present at *any* level ------------------------------------ */
        if (n.contains("limit"))            a.hasLimit    = true;
        if (n.contains("distinct"))         a.hasDistinct = true;
        if (n.contains("union_result"))   { a.hasUnion    = true; a.numUnion++; }
        if (n.contains("grouping_operation"))                    a.grp = true;
        if (n.contains("ordering_operation") ||
            getBool(n, "using_filesort"))                       a.ord = true;
        if (getBool(n, "using_temporary_table"))                a.tmp = true;

        /* recurse into children (skip the “table” object itself) ---------- */
        for (const auto &kv : n.items())
            if (kv.key() != "table")
                walk(kv.value(), a, depth + 1);
    }
    else if (n.is_array())
    {
        for (const auto &el : n)
            walk(el, a, depth);
    }

    /* track maximum depth */
    a.maxDepth = std::max(a.maxDepth, depth);
}
// ───────────── ❷  top-level helper – converts a full EXPLAIN JSON into 32 features ─────────────
static bool plan2feat(const json &plan, float f[NUM_FEATS])
{
    if (!plan.contains("query_block")) return false;

    // unwrap UNION wrapper exactly like the kernel code
    const json *qb = &plan["query_block"];
    if (qb->contains("union_result")) {
        const auto &specs = (*qb)["union_result"]["query_specifications"];
        if (specs.is_array() && !specs.empty())
            qb = &specs[0]["query_block"];
    }

    // aggregate stats over the whole plan tree
    Agg a;
    walk(*qb, a, 1);
    if (a.cnt == 0) return false;
    const double inv = 1.0 / a.cnt;
    int k = 0;

    // pull out the two raw JSON fields we still need
    double qCost   = safe_f(qb->value("cost_info", json::object()), "query_cost");
    double rootRow = safe_f(*qb, "rows_produced_per_join");

    // ─────────── write the 32 features ───────────
    //  0–6: log-scaled cost/cardinals
    f[k++] = log1p_clip(a.re * inv);
    f[k++] = log1p_clip(a.rp * inv);
    f[k++] = log1p_clip(a.f  * inv);
    f[k++] = log1p_clip(a.rc * inv);
    f[k++] = log1p_clip(a.ec * inv);
    f[k++] = log1p_clip(a.pc * inv);
    f[k++] = log1p_clip(a.dr * inv);

    //  7–11: fractional access-types
    f[k++] = a.cRange * inv;
    f[k++] = a.cRef   * inv;
    f[k++] = a.cEq    * inv;
    f[k++] = a.cIdx   * inv;
    f[k++] = a.cFull  * inv;

    // 12: fraction of nodes using an index
    f[k++] = a.idxUse * inv;

    // 13–15: selection stats
    f[k++] = a.selSum * inv;    // mean selectivity
    f[k++] = a.selMin;          // min selectivity
    f[k++] = a.selMax;          // max selectivity

    // 16–17: shape metrics
    f[k++] = a.maxDepth;        // max recursion depth
    f[k++] = a.fanoutMax;       // max fan-out

    // 18–20: flags for group/order/temp
    f[k++] = a.grp ? 1.f : 0.f;
    f[k++] = a.ord ? 1.f : 0.f;
    f[k++] = a.tmp ? 1.f : 0.f;

    // 21–22: read/eval cost ratios
    f[k++] = a.ratioSum * inv;  // mean read/eval ratio
    f[k++] = a.ratioMax;        // max read/eval ratio

    // 23: total query cost
    f[k++] = log1p_clip(qCost);

    // 24: per-row output cardinality
    f[k++] = log1p_clip(rootRow);

    // 25–27: new per-cost features
    //   25 = prefix_cost / read_cost
    f[k++] = log1p_clip(static_cast<float>( (a.pc * inv) / std::max(1e-6, a.rc * inv) ));
    //   26 = read_cost per row scanned
    f[k++] = log1p_clip(static_cast<float>( (a.rc * inv) / std::max(1e-6, a.re * inv) ));
    //   27 = eval_cost per row scanned
    f[k++] = log1p_clip(static_cast<float>( (a.ec * inv) / std::max(1e-6, a.re * inv) ));

    // 28–29: single-table vs multi-join
    f[k++] = (a.cnt == 1 ? 1.f : 0.f);
    f[k++] = (a.cnt >  1 ? 1.f : 0.f);

    // 30: index_scan_depth = maxDepth × frac_index_access
    f[k++] = log1p_clip(static_cast<float>( a.maxDepth * (a.idxUse * inv) ));

    // 31: idx_use_to_fullscan_ratio = (idxUse/ cnt) / (fullScan/ cnt)
    {
      double fullFrac = (a.cFull ? (a.cFull * inv) : 1e-3);
      f[k++] = log1p_clip(static_cast<float>( (a.idxUse * inv) / fullFrac ));
    }

    /* ────────── 8 NEW FEATURES (32-39) ────────── */
    f[k++] = a.cnt;                                     // 32: #tables
    f[k++] = a.cnt ? float(a.sumPossibleKeys)/a.cnt : 0.f;   // 33: avg possible_keys
    f[k++] = log1p_clip(a.maxPrefix);                   // 34: max prefix_cost
    f[k++] = log1p_clip(a.minRead < 1e30 ? a.minRead : 0.0); // 35: min read_cost
    f[k++] = (a.cnt > 1 ? float(a.cnt-1)/a.cnt : 0.f);  // 36: join fraction
    f[k++] = (rootRow > 0 ? float(a.re) / rootRow : 0.f);    // 37: scans/root rows
    f[k++] = float(a.selMax - a.selMin);                // 38: selectivity range
    {
        float denom = float(a.cRange + a.cRef + a.cEq + a.cIdx);
        f[k++] = denom ? float(a.idxUse) / denom : 0.f; // 39: index-density
    }
    f[k++] = float(qCost);                 // 40: RAW query_cost  (no log!)
    f[k++] = (qCost > 5e4 ? 1.f : 0.f);    // 41: cost > 50 000 flag
    // 42: fraction of tables whose used_columns are fully covered by possible_keys
    f[k++] = a.cnt ? float(a.coverCount) / a.cnt : 0.f;
    return true;
}


/* ─────────── decision-tree loader / predictor ─────────── */
static std::vector<DTNode> TREE;
static bool load_tree(const std::string& p){
    std::ifstream in(p); if(!in) return false; json j; in>>j;
    if(!j.is_array()) return false; TREE.clear(); TREE.reserve(j.size());
    for(const auto& n:j){
        DTNode t; t.feat=n.value("feat",-1); t.thresh=n.value("thresh",0.f);
        t.left=n.value("left",-1); t.right=n.value("right",-1); t.prob=n.value("prob",0.f);
        TREE.push_back(t);
    } return !TREE.empty();
}
static float predict(const float f[NUM_FEATS]){
    // if covering-index feature is 1, force row
    if (f[42] >= 0.5f) return /* probability-of-row? */ 0.0f;
    int id=0; while(TREE[id].feat!=-1)
        id=(f[TREE[id].feat]<TREE[id].thresh)?TREE[id].left:TREE[id].right;
    return TREE[id].prob;
}

/* ───────────────────────────────────────────────────────── */
int main()
{

    /* connect */
    MYSQL* conn=mysql_init(nullptr);
    if(!mysql_real_connect(conn,"127.0.0.1","root","","task_info",44444,nullptr,0)){
        std::cerr<<"mysql: "<<mysql_error(conn)<<"\n"; return 1;
    }

    if(mysql_query(conn, "set use_imci_engine=off;")){ std::cerr<<"set off use_imci_engine failed\n"; return 1; }

    /* EXPLAIN */
    const char* qry=
      "EXPLAIN FORMAT=JSON "
      "SELECT count(*) AS cnt,module_code "
      "FROM task_info "
      "WHERE env='online' "
      "AND assignee_uid IN('2206529173543','cn011000001612304') "
      "AND platform_code IN('A@42gBZ','A@RhHWV');";
    // const char* qry = "explain format=json SELECT COUNT(DISTINCT MemberAcctId) as Count from  member_log WHERE  ShopId = 14823  AND IsGoShop = 1  AND CreateTime BETWEEN timestamp('2025-02-18 00:00:00.000000')  AND timestamp('2025-02-18 23:59:59.999000');";
    if(mysql_query(conn,qry)){ std::cerr<<"mysql: "<<mysql_error(conn)<<"\n"; return 1; }
    MYSQL_RES* res=mysql_store_result(conn);
    MYSQL_ROW  row=mysql_fetch_row(res);
    std::string js(row[0],mysql_fetch_lengths(res)[0]);

    /* parse JSON */
    json plan; try{ plan=json::parse(js); }
    catch(...){ std::cerr<<"JSON parse error\n"; return 1; }

    /* feature extraction */
    float feat[NUM_FEATS]{};
    if(!plan2feat(plan,feat)){ std::cerr<<"plan2feat failed\n"; return 1; }

    std::cout<<"[DEBUG] raw features:\n";
    for(int i=0;i<NUM_FEATS;++i)
        std::cout<<"  feat["<<i<<"] = "<<feat[i]<<'\n';

    /* load tree & predict */
    if(!load_tree("/home/wuy/row_column_routing/rowcol_dtree.json")){
        std::cerr<<"cannot load tree\n"; return 1;
    }
    float p=predict(feat);
    std::cout<<"P(column) = "<<p<<"  →  "<<(p>=0.5f?"COLUMN":"ROW")<<'\n';

    mysql_free_result(res); mysql_close(conn); return 0;
}
