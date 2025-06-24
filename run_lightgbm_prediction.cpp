/*****************************************************************************************
 * run_lightgbm_prediction.cpp  – offline predictor (row-vs-column) using a trained
 *                                LightGBM model + the 65-feature extractor you
 *                                just patched (outerRows / eqChainDepth).
 *
 * Build (example):
 *   g++ -O3 -std=c++17 -I$LIGHTGBM/include \
 *       run_lightgbm_prediction.cpp -o run_lgb_predict \
 *       -L$LIGHTGBM/lib -l_lightgbm -lmysqlclient -pthread
 *
 * Run (basic):
 *   ./run_lgb_predict --sql="SELECT …" --model=lgb_model.txt \
 *                     --mysql_db=hybench_sf10 --mysql_tbl=customer
 *
 * Or read the SQL from a file:
 *   ./run_lgb_predict --sql_file=q1.sql …
 *****************************************************************************************/
#include <bits/stdc++.h>
#include <mysql/mysql.h>
#include <LightGBM/c_api.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;
constexpr int    NUM_FEATS   = 77;
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

            a.maxPrefix = std::max(a.maxPrefix, pc);
            a.minRead   = std::min(a.minRead , rc);

            if (re > 0) {
                double sel = rp / re;
                a.selSum   += sel;
                a.selMin    = std::min(a.selMin, sel);
                a.selMax    = std::max(a.selMax, sel);
                a.fanoutMax = std::max(a.fanoutMax, sel);
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

/* ----------------------------------------------------------- */
/*  plan → 77-dim feature (63 old + 2 signals + 12 histograms) */
/* ----------------------------------------------------------- */
static bool plan2feat(const json& plan, float f[NUM_FEATS])
{
    if (!plan.contains("query_block")) return false;
    const json* qb = &plan["query_block"];

    /* If UNION, only inspect the first branch (same as before) */
    if (qb->contains("union_result")) {
        const auto& specs = (*qb)["union_result"]["query_specifications"];
        if (specs.is_array() && !specs.empty())
            qb = &specs[0]["query_block"];
    }

    /* ---------- first pass: old 65-dim logic ---------- */
    Agg a; walk(*qb, a);
    if (!a.cnt) return false;

    const double inv = 1.0 / a.cnt;
    int   k      = 0;
    double qCost = safe_f(qb->value("cost_info", json::object()), "query_cost");
    double rootRow = safe_f(*qb, "rows_produced_per_join");

#define PUSH(x) f[k++] = static_cast<float>(x)

    /* ……………………… (the original PUSH sequence 0-64 unchanged) …………………… */
    /* Copy your existing 0-64 feature pushes here exactly as before          */
    /* They were already present in your file – snip for brevity              */
    /* --------------------------------------------------------------------- */
    /* ( keep everything up to: PUSH(lp(a.outerRows));  PUSH(a.eqChainDepth); ) */
    /* After those two lines k==65                                            */
    /* --------------------------------------------------------------------- */
    /* --- 0‒6 basic costs / rows ------------------------------------ */
    PUSH(lp(a.re * inv)); PUSH(lp(a.rp * inv)); PUSH(lp(a.f * inv));
    PUSH(lp(a.rc * inv)); PUSH(lp(a.ec * inv)); PUSH(lp(a.pc * inv));
    PUSH(lp(a.dr * inv));

    /* --- 7‒12 access-type counters (ratio) ------------------------- */
    PUSH(a.cRange * inv); PUSH(a.cRef * inv); PUSH(a.cEq * inv);
    PUSH(a.cIdx   * inv); PUSH(a.cFull* inv); PUSH(a.idxUse* inv);

    /* --- 13‒17 selectivity / shape -------------------------------- */
    PUSH(a.selSum * inv); PUSH(a.selMin); PUSH(a.selMax);
    PUSH(a.maxDepth); PUSH(a.fanoutMax);

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

    /* --- 40‒43 covering-index & big-cost flags --------------------- */
    PUSH(qCost);
    PUSH(qCost > 5e4);
    PUSH(a.cnt ? double(a.coverCount) / a.cnt : 0);
    PUSH(a.coverCount == a.cnt);

    /* --- 44‒46 log-diffs & counts --------------------------------- */
    PUSH(lp(a.re * inv) - lp(a.selSum * inv));
    PUSH(a.cnt);
    PUSH(lp(a.cnt));

    /* --- 47‒50 PK / cover counters -------------------------------- */
    PUSH(a.sumPK);
    PUSH(a.cnt ? double(a.sumPK) / a.cnt : 0);
    PUSH(a.coverCount);
    PUSH(a.cnt ? double(a.coverCount) / a.cnt : 0);

    /* --- 51‒56 repeated access shares ------------------------------ */
    PUSH(a.idxUse * inv); PUSH(a.cRange * inv); PUSH(a.cRef * inv);
    PUSH(a.cEq   * inv); PUSH(a.cIdx   * inv); PUSH(a.cFull  * inv);

    /* --- 57‒59 prefix / read extremes ------------------------------ */
    PUSH(lp(a.maxPrefix * inv));
    PUSH(lp(a.minRead < 1e30 ? a.minRead : 0));
    PUSH(a.selMax - a.selMin);

    /* --- 60‒62 extremes & ratios ----------------------------------- */
    PUSH(a.ratioMax);
    PUSH(a.fanoutMax);
    PUSH(a.selMin > 0 ? double(a.selMax / a.selMin) : 0);

    /* --- 63‒64 row-wins signals ------------------------------------ */
    PUSH(lp(a.outerRows));         // 63
    PUSH(a.eqChainDepth);          // 64

    /* ---------- second pass: collect column IDs used by the query ---------- */
    unordered_set<string> touched;   // "db.tbl.col"
    function<void(const json&)> collect = [&](const json& n) {
        if (n.is_object()) {
            if (n.contains("table") && n["table"].is_object()) {
                const auto& t = n["table"];
                if (t.contains("used_columns") && t["used_columns"].is_array()) {
                    string tbl = t.value("table_name", "");
                    string db  = "";      // fill if your plan includes DB
                    for (const auto& c : t["used_columns"])
                        if (c.is_string())
                            touched.insert(db + "." + tbl + "." + c.get<string>());
                }
            }
            for (const auto& kv : n.items())
                collect(kv.value());
        } else if (n.is_array()) {
            for (const auto& v : n) collect(v);
        }
    };
    collect(*qb);

    /* ---------- build 3 histograms (width / NDV / dtype) ---------- */
    array<float,4>  width_hist = {0,0,0,0};     // 0-4, 5-16, 17-64, 65+
    array<float,3>  ndv_hist   = {0,0,0};       // 1-1e3, 1e3-1e5, 1e5+
    array<float,5>  type_hist  = {0,0,0,0,0};   // enum ColDType

    for (const auto& id : touched) {
        ColStats s = lookup_col_stats(id);

        /* width bucket */
        if      (s.avg_width <= 4)   width_hist[0] += 1;
        else if (s.avg_width <= 16)  width_hist[1] += 1;
        else if (s.avg_width <= 64)  width_hist[2] += 1;
        else                         width_hist[3] += 1;

        /* NDV bucket */
        if      (s.ndv <= 1e3)       ndv_hist[0]   += 1;
        else if (s.ndv <= 1e5)       ndv_hist[1]   += 1;
        else                         ndv_hist[2]   += 1;

        /* data-type bucket */
        if (s.dtype < COL_DTYPE_N)   type_hist[s.dtype] += 1;
    }

    // auto normalise = [](auto& h) {
    //     float sum = accumulate(h.begin(), h.end(), 0.f);
    //     if (sum < 1.f) sum = 1.f;
    //     for (float& v : h) v /= sum;
    // };
    normalise(width_hist);
    normalise(ndv_hist);
    normalise(type_hist);

    /* ---------- append 12 new features ---------- */
    for (float v : width_hist) PUSH(v);
    for (float v : ndv_hist)   PUSH(v);
    for (float v : type_hist)  PUSH(v);

#undef PUSH
    return k == NUM_FEATS;      // should now be 77
}


/* ────────────────────────────────────────────────────────────────────────────
 *  NEW main()  —  robust CSV reader + resilient index-load
 * ────────────────────────────────────────────────────────────────────────── */

struct Rec { std::string db, qid, sql; int use_imci; };

/* --- tiny CSV reader that tolerates quoted commas and embedded new-lines --- */
static bool read_csv_row(std::istream& in, std::vector<std::string>& out)
{
    out.clear(); std::string cell; bool in_q = false;
    for (int ch; (ch = in.get()) != EOF; ) {
        if (ch == '"')                   in_q = !in_q;           /* toggle quote */
        else if (ch == ',' && !in_q)     { out.emplace_back(std::move(cell)); cell.clear(); }
        else if (ch == '\n' && !in_q)    { out.emplace_back(std::move(cell)); return true; }
        else if (ch != '\r')             cell.push_back(char(ch));             /* ignore CR */
    }
    if (!cell.empty() || in_q)           out.emplace_back(std::move(cell));
    return !out.empty();
}
/* --------------- helper: get query_block or diagnose why not ------------- */
static bool extract_query_block(const json& plan, const json*& qb, std::string& why)
{
    if (plan.contains("query_block")) { qb = &plan["query_block"]; return true; }
    if (plan.contains("errmsg"))      { why = plan["errmsg"].get<std::string>(); return false; }
    why = plan.dump(120);             /* first 120 chars of whatever we got */
    return false;
}

    constexpr double COST_THR = 5e4;   // MySQL cost rule



/* --------------------------- new main() ---------------------------------- */
int main(int argc,char*argv[])
{

    /* ----- CLI (same flags as before) ----- */
    std::string csv="problem_sqls.csv", model="lgb_model_column_histogram.txt";
    std::string host="127.0.0.1",user="root",pass=""; int port=44444;
    for(int i=1;i<argc;++i){
        std::string a(argv[i]);
        auto eat=[&](const char*s){ size_t L=strlen(s); return a.rfind(s,0)==0? a.substr(L):"";};
        if(auto v=eat("--csv=");!v.empty())   csv=v;
        else if(auto v=eat("--model=");!v.empty()) model=v;
        else if(auto v=eat("--mysql_host=");!v.empty()) host=v;
        else if(auto v=eat("--mysql_port=");!v.empty()) port=stoi(v);
        else if(auto v=eat("--mysql_user=");!v.empty()) user=v;
        else if(auto v=eat("--mysql_pass=");!v.empty()) pass=v;
        else { std::cerr<<"unknown arg "<<a<<'\n'; return 1; }
    }

    /* ----- read CSV (same robust reader as previous message) ----- */
    std::ifstream fin(csv); if(!fin) logE("cannot open "+csv);
    std::vector<std::string> row; read_csv_row(fin,row);          /* header */
    struct Rec{std::string db,qid,sql;int use_imci;};
    std::vector<Rec> Q; std::unordered_set<std::string> dbs;
    while(read_csv_row(fin,row) && row.size()>=10){
        Rec r{row[0],row[1],row[9],std::stoi(row[5])};
        if(!r.sql.empty()&&r.sql.front()=='"') r.sql.erase(0,1);
        if(!r.sql.empty()&&r.sql.back()=='"')  r.sql.pop_back();
        if(!r.db.empty()){ Q.push_back(r); dbs.insert(r.db);}      /* drop empty-db rows */
    }
    logI("loaded "+std::to_string(Q.size())+" queries from CSV");

    /* ----- index definitions (unchanged) ----- */
    load_all_index_defs(host,port,user,pass,
                        std::vector<std::string>(dbs.begin(),dbs.end()));
    logI("index map size = "+std::to_string(indexCols.size()));

    /* ----- LightGBM model ----- */
    BoosterHandle booster=nullptr; int iters=0;
    if(LGBM_BoosterCreateFromModelfile(model.c_str(),&iters,&booster))
        logE("cannot load model "+model);
    
    float TAU_STAR = 0.0f;
    
    int correct=0, processed=0;
    for(const auto& r:Q)
    {
        MYSQL* c=mysql_init(nullptr);
        if(!mysql_real_connect(c,host.c_str(),user.c_str(),pass.c_str(),
                               r.db.c_str(),port,nullptr,0))
        { logW("connect "+r.db+" failed"); mysql_close(c); continue; }

        mysql_query(c, "set use_imci_engine=off;");

        /* enlarge packet + relax ONLY_FULL_GROUP_BY */
        mysql_query(c,"SET SESSION max_allowed_packet=67108864");
        mysql_query(c,"SET SESSION sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''))");

        std::string ex="EXPLAIN FORMAT=JSON "+r.sql;
        if(mysql_query(c,ex.c_str()))
        { logW("EXPLAIN fail "+r.qid+": "+mysql_error(c)); mysql_close(c); continue; }

        MYSQL_RES* res=mysql_store_result(c); MYSQL_ROW row_=mysql_fetch_row(res);
        unsigned long* len=mysql_fetch_lengths(res);
        std::string js=(row_&&len)?std::string(row_[0],len[0]):"";
        mysql_free_result(res); mysql_close(c);

        json plan; try{ plan=json::parse(js);}catch(...){
            logW("bad JSON for "+r.qid); continue; }

        const json* qb=nullptr; std::string why;
        if(!extract_query_block(plan,qb,why)){
            logW("plan missing query_block ("+r.qid+") "+why.substr(0,80));
            /*  fallback = cost threshold  */
            bool act_col = r.use_imci==1;
            bool pred_col = plan.contains("cost_info") &&
                            safe_f(plan["cost_info"],"query_cost") > COST_THR;
            if(pred_col==act_col) ++correct;
            ++processed;
            std::cout<<r.db<<':'<<r.qid<<"  pred="<<(pred_col?"COLUMN":"ROW")
                     <<"*  actual="<<(act_col?"COLUMN":"ROW")<<"  (fallback)\n";
            continue;
        }

        float feat[NUM_FEATS]={};
        if(!plan2feat(plan,feat)){ logW("unexpected extract failure "+r.qid); continue; }

        double prob=0; int64_t out_len=0;
        LGBM_BoosterPredictForMat(booster,feat,C_API_DTYPE_FLOAT32,
                                  1,NUM_FEATS,1,C_API_PREDICT_NORMAL,
                                  -1,0,"",&out_len,&prob);

        
        bool pred_col= prob>TAU_STAR, act_col=r.use_imci==1;

        if(pred_col==act_col) ++correct;
        ++processed;

        std::cout<<r.db<<':'<<r.qid<<"  pred="<<(pred_col?"COLUMN":"ROW")
                 <<"  actual="<<(act_col?"COLUMN":"ROW")
                 <<"  prob="<<std::fixed<<std::setprecision(3)<<prob<<'\n';
    }

    std::cout<<"\nProcessed "<<processed<<" / "<<Q.size()
             <<"   Accuracy "<<correct<<'/'<<processed<<" ("
             <<std::fixed<<std::setprecision(2)
             <<(processed?100.0*correct/processed:0.0)<<"%)\n";
    LGBM_BoosterFree(booster);
    return 0;
}
