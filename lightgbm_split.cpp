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
constexpr int    NUM_FEATS   = 102;
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



/* ─────────── sample struct & loaders ─────────── */
struct Sample{
    array<float,NUM_FEATS> feat{};
    int    label = 0;              // 真实最优: 1=column 更快, 0=row 更快
    double row_t = 0, col_t = 0;   // 两种执行路径真实耗时 (s)
    double qcost = 0;              // Optimizer 估计的 query_cost
    int    fann_pred   = -1;       // 0/1；若 CSV 中缺失则保持 -1
    int    hybrid_pred = -1;       // 0/1；若 CSV 中缺失则保持 -1
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
            ++cur; if(cur%500==0) progress("scan "+d,cur,tot);

            string qid = pf.stem().string();
            auto it = meta.find(qid);
            if(it == meta.end()) continue;

            Sample s;
            if(!load_plan_file(pf.string(),s.feat,s.qcost)) {
                continue;
            }
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



/* ---------- tiny C++11 clamp (std::clamp back-port) ---------- */
template<typename T>
static inline const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v ? hi : v);
}

/*********************************************************************
 * gap-regression LightGBM router · train_and_eval  (two-phase version)
 *********************************************************************/
static void train_and_eval(const std::vector<Sample>& DS,
                           const std::string&  model_path,
                           int     num_trees      = 800,
                           int     max_depth      = 20,
                           double  lr             = 0.06,
                           double  subsample      = 0.7,
                           double  colsample      = 0.8,
                           bool    skip_train     = false,
                           int     num_threads    = 20 /* 0 ⇒ auto */ )
{
    /* ---------- constants ---------- */
    constexpr double COST_THR = 5e4;          // old cost rule
    constexpr double GAP_THR  = 0.0;          // disable fallback
    constexpr int    ROWS_RAW_IDX = 99;       // new feature indices
    constexpr int    CUM_FAN_IDX  = 97;

    constexpr double BIG_TABLE = 3e6;         // you can tune
    constexpr double FP_MULT2  = 4;         // extra penalty
    constexpr float  DECISION  = 0.0f;        // τ*

    /* ---------- build VALID subset ---------- */
    std::vector<const Sample*> S;
    constexpr double GAP_MIN = 0;

    for (const auto& s : DS)
        if (s.hybrid_pred != -1 && s.fann_pred != -1 &&
            s.row_t > 0 && s.col_t > 0 && std::fabs(s.row_t - s.col_t) >= GAP_MIN)
            S.push_back(&s);

    const int N = static_cast<int>(S.size());
    if (!N) { logE("no sample has full information"); return; }

    if (num_threads <= 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    /* ---------- allocate matrices ---------- */
    std::vector<float>  X(static_cast<size_t>(N) * NUM_FEATS), y(N), w(N);
    double P = 0, N0 = 0;
    for (auto p : S) p->label ? ++P : ++N0;
    const double w_pos = P  ? N / (2 * P)  : 1.0;
    const double w_neg = N0 ? N / (2 * N0) : 1.0;

    constexpr double Q_SMALL = 1e4;
    constexpr double Q_LARGE = 5e4;
    constexpr double BOOST   = 1.0;

    for (int i = 0; i < N; ++i) {
        const Sample& s = *S[i];

        for (int j = 0; j < NUM_FEATS; ++j)
            X[i * NUM_FEATS + j] = s.feat[j];

        /* regression label  Δ = ln(rt/ct) */
        double rt = std::max(s.row_t, EPS_RUNTIME);
        double ct = std::max(s.col_t, EPS_RUNTIME);
        y[i] = float(std::log(rt) - std::log(ct));

        /* base weight + heuristics (same as before) */
        double base = (s.label ? w_pos : w_neg);
        double q_ln = std::log1p(s.qcost);
        double w_i  = base * (1.0 + 0.20 * q_ln);

        double fan   = std::min<double>(3.0, s.feat[22]);
        double ratio = std::min<double>(3.0, s.feat[60]);
        if (fan   > 1.3) w_i += 0.6 * base * (fan   - 1.2);
        if (ratio > 1.1) w_i += 0.4 * base * (ratio - 1.0);

        double pc_rc = std::exp(std::min<double>(2.0, s.feat[25]));
        if (pc_rc > 2.0 && s.qcost < 1e4) w_i += 0.8 * base * (pc_rc - 1.5);
        if (fan > 2.5 && s.qcost < 8e3)   w_i += 2.0 * base;

        auto rel_gap = [](double fast, double slow) {
            double g = std::log(slow / std::max(1e-6, fast));  // ≥0
            double n = clamp(g / 3.0, 0.0, 1.0);
            return n;
        };
        if (s.label == 1 && s.qcost < Q_SMALL)
            w_i *= (BOOST + 0 * rel_gap(s.col_t, s.row_t));
        if (s.label == 0 && s.qcost > Q_LARGE)
            w_i *= (BOOST + 0 * rel_gap(s.row_t, s.col_t));

        w[i] = static_cast<float>(w_i);
    }

    DatasetHandle dtrain = nullptr;
    BoosterHandle booster = nullptr;

    /* ---------- SKIP-TRAIN path ---------- */
     if (skip_train) {
        int num_iters = 0;                        /* must be valid storage */
        if (LGBM_BoosterCreateFromModelfile(model_path.c_str(),
                                            &num_iters,           /*  OK   */
                                            &booster))
        {
            logE("model load failed");            /* booster stays null   */
            return;
        }
    }
    /* ---------- TRAIN path (two-phase) ---------- */
    else {
        /* 1)  create dataset -------------------------------------- */
        if (LGBM_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT32,
                                      N, NUM_FEATS, 1, "", nullptr, &dtrain))
        { logE("DatasetCreate failed"); return; }
        LGBM_DatasetSetField(dtrain, "label" , y.data(), N, C_API_DTYPE_FLOAT32);
        LGBM_DatasetSetField(dtrain, "weight", w.data(), N, C_API_DTYPE_FLOAT32);

        /* 2)  create booster + parameters ------------------------- */
        // std::string param =
        //     "objective=fair fair_c=0.9"
        //     " max_bin=127 num_leaves=512"
        //     " max_depth="        + std::to_string(max_depth) +
        //     " learning_rate="    + std::to_string(lr)        +
        //     " feature_fraction=" + std::to_string(colsample) +
        //     " bagging_fraction=" + std::to_string(subsample) +
        //     " bagging_freq=1"
        //     " lambda_l2=1.0"
        //     " num_threads="      + std::to_string(num_threads) +
        //     " verbosity=-1";


        std::string param =
            "boosting=goss"                 // Gradient One-Side Sampling
            " top_rate=0.2 other_rate=0.05" // keep 20 % biggest gradients each iter
            " objective=fair fair_c=1.2"    // like yours but softer (bigger C)
            " metric=l1,l2"                 // track both
            " max_bin=127"
            " num_leaves=512"
            " min_data_in_leaf=20"
            " learning_rate="    + std::to_string(lr * 0.75) +  // smaller LR for GOSS
            " feature_fraction=" + std::to_string(colsample) +
            " lambda_l2=0.5"
            " num_threads="      + std::to_string(num_threads) +
            " verbosity=-1";



        /* monotone constraints (unchanged) */
        {
            std::string mono; mono.reserve(NUM_FEATS * 2);
            for (int i = 0; i < NUM_FEATS; ++i) {
                int m = 0;
                if (i == 22 || i == 60 || i == 101) m = +1;
                if (i == 65 || i == 97 || i == 98 || i == 99 || i == 100) m = -1;
                mono += std::to_string(m);
                if (i + 1 < NUM_FEATS) mono += ',';
            }
            param += " monotone_constraints=" + mono;
        }

        if (LGBM_BoosterCreate(dtrain, param.c_str(), &booster))
        { logE("BoosterCreate failed"); return; }

        /* 3)  phase-1 training ----------------------------------- */
        int stage1_trees = int(num_trees * 0.75);       // e.g. 600/800
        for (int it = 0, fin; it < stage1_trees; ++it) {
            LGBM_BoosterUpdateOneIter(booster, &fin);
            if ((it + 1) % 10 == 0 || it + 1 == stage1_trees)
                progress("train-1", it + 1, stage1_trees);
        }
        std::cerr << '\n';

        /* 4)  predict & re-weight -------------------------------- */
        std::vector<double> pred_tmp(N); int64_t out_len = 0;
        if (LGBM_BoosterPredictForMat(booster, X.data(), C_API_DTYPE_FLOAT32,
                                      N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
                                      -1, 0, "", &out_len, pred_tmp.data()))
        { logE("PredictForMat failed"); return; }

        for (int i = 0; i < N; ++i) {
            bool col_pred = pred_tmp[i] > DECISION;
            bool row_fast = S[i]->label == 0;
            double big = S[i]->feat[ROWS_RAW_IDX];
            if (col_pred && row_fast && big > BIG_TABLE)
                w[i] *= FP_MULT2;
        }
        LGBM_DatasetSetField(dtrain, "weight", w.data(), N, C_API_DTYPE_FLOAT32);
        /* NOTE: keep dtrain alive! */
        LGBM_BoosterResetTrainingData(booster, dtrain);

        /* 5)  phase-2 training (remaining trees) ------------------ */
        int stage2_trees = num_trees - stage1_trees;    // e.g. 200/800
        for (int it = 0, fin; it < stage2_trees; ++it) {
            LGBM_BoosterUpdateOneIter(booster, &fin);
            if ((it + 1) % 10 == 0 || it + 1 == stage2_trees)
                progress("train-2", it + 1, stage2_trees);
        }
        std::cerr << '\n';

        /* 6)  save final model ----------------------------------- */
        LGBM_BoosterSaveModel(booster, 0, -1, 0, model_path.c_str());
        logI("Model saved → " + model_path);
    }

    /* ---------- evaluation ---------- */
    std::vector<double> pred(N);  int64_t out_len = 0;
    if (LGBM_BoosterPredictForMat(booster, X.data(), C_API_DTYPE_FLOAT32,
                                  N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
                                  -1, 0, "", &out_len, pred.data()))
    { logE("PredictForMat failed"); }

    float TAU_STAR = 0.0f;

    int TP=0,FP=0,TN=0,FN=0;
    double r_row=0,r_col=0,r_opt=0,
           r_cost=0,r_hopt=0,r_fann=0,r_lgb=0,
           loss_a=0,cnt_a=0,loss_b=0,cnt_b=0;

    for (int i = 0; i < N; ++i) {
        const Sample& s = *S[i];
        bool col = pred[i] > TAU_STAR;

        (col ? (s.label?++TP:++FP) : (s.label?++FN:++TN));

        r_row  += s.row_t;
        r_col  += s.col_t;
        r_opt  += std::min(s.row_t, s.col_t);
        r_cost += (s.qcost > COST_THR) ? s.col_t : s.row_t;
        r_hopt += s.hybrid_pred ? s.col_t : s.row_t;
        r_fann += s.fann_pred   ? s.col_t : s.row_t;
        r_lgb  += col           ? s.col_t : s.row_t;

        bool isA = (s.label==1 && s.qcost < Q_SMALL);
        bool isB = (s.label==0 && s.qcost > Q_LARGE);
        if (isA) { loss_a += (col ? s.col_t : s.row_t); ++cnt_a; }
        if (isB) { loss_b += (col ? s.col_t : s.row_t); ++cnt_b; }
    }

    std::cout << "── Group (A) avg runtime : "
              << (cnt_a ? loss_a / cnt_a : -1) << '\n'
              << "── Group (B) avg runtime : "
              << (cnt_b ? loss_b / cnt_b : -1) << "\n\n";

    auto avg = [&](double s){ return s / N; };

    std::cout << "*** EVALUATED ON " << N << " COMPLETE SAMPLES ***\n"
              << std::fixed << std::setprecision(6)
              << "\n=== CONFUSION (LightGBM) ===\nTP=" << TP
              << " FP=" << FP << " TN=" << TN << " FN=" << FN
              << "\nAccuracy=" << double(TP+TN)/N
              << "  BalAcc="  << 0.5*(double(TP)/(TP+FN)+double(TN)/(TN+FP))
              << "  F1="      << (TP?2.0*TP/(2*TP+FP+FN):0.) << "\n\n"
              << "=== AVG RUNTIME (s) ===\n"
              << "Row only            : " << avg(r_row)  << '\n'
              << "Column only         : " << avg(r_col)  << '\n'
              << "Cost threshold rule : " << avg(r_cost) << '\n'
              << "Hybrid optimizer    : " << avg(r_hopt) << '\n'
              << "FANN model          : " << avg(r_fann) << '\n'
              << "LightGBM            : " << avg(r_lgb)  << '\n'
              << "Optimal (oracle)    : " << avg(r_opt)  << "\n\n";

    /* ---------- tidy-up ---------- */
    if (!skip_train) LGBM_DatasetFree(dtrain);
    LGBM_BoosterFree(booster);
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

/* ——  very small helper —— */
static double compute_bal_acc(const std::string& model_path,
                              const std::vector<Sample>& DS)
{
    if (DS.empty()) return 0.0;

    /* 1) load model */
    BoosterHandle booster = nullptr;
    int num_iters = 0;
    if (LGBM_BoosterCreateFromModelfile(model_path.c_str(),
                                        &num_iters, &booster))
    {
        logE("cannot open "+model_path);
        return 0.0;
    }

    /* 2) features → matrix */
    const int N = DS.size();
    std::vector<float> X(size_t(N) * NUM_FEATS);
    std::vector<double> pred(N);
    for (int i=0;i<N;++i)
        std::copy(DS[i].feat.begin(),
                  DS[i].feat.end(),
                  X.begin()+i*NUM_FEATS);

    int64_t out_len = 0;
    LGBM_BoosterPredictForMat(
        booster, X.data(), C_API_DTYPE_FLOAT32,
        N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
        -1, 0, "", &out_len, pred.data());

    /* 3) balanced accuracy */
    int TP=0,TN=0,FP=0,FN=0;
    for(int i=0;i<N;++i){
        bool col = pred[i] > 0.0;        // τ*=0
        if(col &&  DS[i].label) ++TP;
        if(col && !DS[i].label) ++FP;
        if(!col && DS[i].label) ++FN;
        if(!col &&!DS[i].label) ++TN;
    }
    LGBM_BoosterFree(booster);

    double tpr = (TP+FN)? double(TP)/(TP+FN):0;
    double tnr = (TN+FP)? double(TN)/(TN+FP):0;
    return 0.5*(tpr+tnr);
}

/* ───── new: 全量一次性加载 ───── */
using DirSamples = unordered_map<string, vector<Sample>>;

/* 递归扫描每个目录 ⇒ 存进 map[directory] */
static DirSamples load_all_datasets(const string& base,
                                    const vector<string>& dirs)
{
    DirSamples M;

    for (const auto& d : dirs) {
        string csv = base + "/" + d + "/query_costs.csv";
        fs::path planDir = base + "/" + d + "/row_plans";
        if (!fs::exists(csv) || !fs::is_directory(planDir)) {
            logW("skip " + d); continue;
        }

        /* --- meta CSV → unordered_map<qid,MetaRec> (与旧逻辑一致) --- */
        struct MetaRec { int lab; double rt, ct; int fann, hybrid; };
        unordered_map<string, MetaRec> meta;
        ifstream fin(csv); string line; getline(fin, line);       // drop header
        while (getline(fin, line)) {
            stringstream ss(line);
            string qid, lab, rt, ct, fann, hyb;
            getline(ss, qid, ','); getline(ss, lab, ',');
            getline(ss, rt, ',');  getline(ss, ct, ',');
            getline(ss, hyb, ','); getline(ss, fann, ',');
            meta[qid] = {
                lab == "1",
                rt.empty() ? 0 : stod(rt),
                ct.empty() ? 0 : stod(ct),
                fann.empty() ? -1 : stoi(fann),
                hyb .empty() ? -1 : stoi(hyb )
            };
        }

        /* --- scan *.json once --- */
        size_t cur = 0, tot = 0;
        for (auto &p : fs::directory_iterator(planDir))
            if (p.path().extension() == ".json") ++tot;

        vector<Sample> vec;
        vec.reserve(tot);

        for (auto &p : fs::directory_iterator(planDir)) {
            if (p.path().extension() != ".json") continue;
            ++cur; if (cur % 500 == 0) progress("scan " + d, cur, tot);

            string qid = p.path().stem().string();
            auto it = meta.find(qid);
            if (it == meta.end()) continue;

            Sample s;
            if (!load_plan_file(p.path().string(), s.feat, s.qcost)) continue;
            s.label       = it->second.lab;
            s.row_t       = it->second.rt;
            s.col_t       = it->second.ct;
            s.fann_pred   = it->second.fann;
            s.hybrid_pred = it->second.hybrid;
            vec.push_back(std::move(s));
        }
        cerr << "\n";
        logI("dir " + d + "  -> " + to_string(vec.size()) + " samples");
        M.emplace(d, std::move(vec));
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



/* ───────── single-model → 0/1 预测 ───────── */
static vector<int> predict_binary(const string& model_path,
                                  const vector<Sample>& DS,
                                  float tau = 0.0f)        // threshold
{
    const int N = DS.size();
    vector<float> X(size_t(N) * NUM_FEATS);
    for (int i = 0; i < N; ++i)
        std::copy(DS[i].feat.begin(), DS[i].feat.end(),
                  X.begin() + i * NUM_FEATS);

    BoosterHandle booster = nullptr; int iters = 0;
    if (LGBM_BoosterCreateFromModelfile(model_path.c_str(),
                                        &iters, &booster))
    { logE("load "+model_path+" failed"); exit(1); }

    vector<double> pred(N); int64_t out_len = 0;
    LGBM_BoosterPredictForMat(
        booster, X.data(), C_API_DTYPE_FLOAT32,
        N, NUM_FEATS, 1, C_API_PREDICT_NORMAL,
        -1, 0, "", &out_len, pred.data());
    LGBM_BoosterFree(booster);

    vector<int> bin(N);
    for (int i = 0; i < N; ++i) bin[i] = pred[i] > tau;
    return bin;
}

/* ---------- 评估 6 组策略 ---------- */
struct Stat {
    long TP=0,FP=0,TN=0,FN=0;
    double runtime_sum=0;          // 秒
    void add(bool pred_col, bool gt_col, double row_t, double col_t){
        /* confusion count */
        if      ( pred_col &&  gt_col) ++TP;
        else if ( pred_col && !gt_col) ++FP;
        else if (!pred_col &&  gt_col) ++FN;
        else                           ++TN;
        runtime_sum += pred_col ? col_t : row_t;
    }
    double acc (long tot) const { return double(TP+TN)/tot; }
    double recall()       const { return (TP+FN)? double(TP)/(TP+FN):0; }
    double f1()           const {
        return (TP)? 2.0*TP/(2*TP+FP+FN) : 0;
    }
    double avg_rt(long tot) const { return runtime_sum/tot; }
};

/* ─────────────────────────────────────────────────────────────
 *  report_and_evaluate
 *  — 统计 7 种策略 (含 Oracle) 的混淆矩阵与平均运行时
 *    并写入 CSV + 打印控制台表格
 * ───────────────────────────────────────────────────────────── */
void report_metrics(const std::vector<int>& pred_lgb,
                         const std::vector<Sample>& DS_test)
{
    constexpr double COST_THR = 5e4;

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
    for (size_t i=0;i<DS_test.size();++i){
        const Sample &s = DS_test[i];
        bool gt  = s.label;                  // 1 ⇒ column faster
        bool lgb = pred_lgb[i];

        S_row .add(false            ,gt,s.row_t,s.col_t);
        S_col .add(true             ,gt,s.row_t,s.col_t);
        S_cost.add(s.qcost>COST_THR ,gt,s.row_t,s.col_t);
        S_hopt.add(s.hybrid_pred    ,gt,s.row_t,s.col_t);
        S_fann.add(s.fann_pred      ,gt,s.row_t,s.col_t);
        S_lgb .add(lgb              ,gt,s.row_t,s.col_t);

        /* Oracle: always pick the faster of the two paths */
        bool oracle_pred = s.label;          // same definition as gt
        S_opt.add(oracle_pred   ,gt,s.row_t,s.col_t);
    }
    const long N = DS_test.size();

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
    dump("FANN"       ,S_fann);
    dump("LightGBM"   ,S_lgb );
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
    pr("FANN"    ,S_fann);
    pr("LGBM"    ,S_lgb );
    pr("Oracle"  ,S_opt );

    std::cout<<"\n[CSV] test_metrics.csv written (includes Oracle)\n";
}



/* ───────────────────────── UPDATED  main() ───────────────────────── */
int main(int argc, char* argv[])
{
    /* ---------- hyper-params & CLI ---------- */
    int    trees = 400, depth = 10;
    double lr = 0.06, subs = 0.7, col = 0.8;
    string base = "/home/wuy/query_costs";
    uint32_t seed = 42;
    bool skip_train = false;                 // ← 新增

    vector<string> all_dirs;
    for (int i = 1; i < argc; ++i) {
        string a(argv[i]);
        if (a.rfind("--data_dirs=",0)==0){
            string s=a.substr(12), t; stringstream ss(s);
            while(getline(ss,t,',')) all_dirs.push_back(t);
        } else if (a.rfind("--base=",0)==0)        base  = a.substr(7);
        else if (a.rfind("--seed=",0)==0)          seed  = stoi(a.substr(7));
        else if (a.rfind("--trees=",0)==0)         trees = stoi(a.substr(8));
        else if (a.rfind("--max_depth=",0)==0)     depth = stoi(a.substr(12));
        else if (a.rfind("--lr=",0)==0)            lr    = stod(a.substr(5));
        else if (a.rfind("--subsample=",0)==0)     subs  = stod(a.substr(12));
        else if (a.rfind("--colsample=",0)==0)     col   = stod(a.substr(12));
        else if (a=="--skip_train")                skip_train = true;
    }
    if (!skip_train && all_dirs.size() < 8) {
        logE("need ≥8 data dirs for training (omit --skip_train to run inference only)");
        return 1;
    }
    if (skip_train && all_dirs.empty()) {
        logE("need at least 1 data dir when --skip_train is present");
        return 1;
    }

    /* ---------- 一次性扫描磁盘 ---------- */
    DirSamples ALL = load_all_datasets(base, all_dirs);

    if (skip_train)
    {
        const vector<string> model_paths = {"lgb_fold1.txt",
                                            "lgb_fold2.txt",
                                            "lgb_fold3.txt"};
        /* 构造测试集（用户给的所有目录即测试集） */
        auto DS_test = build_subset(all_dirs, ALL);
        if (DS_test.empty()) { logE("no samples found in given data_dirs"); return 1; }

        /* 多数投票 */
        vector<int> vote_sum(DS_test.size(), 0);
        size_t usable_models = 0;
        for (const auto& m : model_paths) {
            if (!fs::exists(m)) { logW("model "+m+" missing, skip"); continue; }
            auto pred = predict_binary(m, DS_test);
            for (size_t i=0;i<pred.size();++i) vote_sum[i] += pred[i];
            ++usable_models;
        }
        if (usable_models == 0) { logE("no usable model files"); return 1; }
        int maj = usable_models/2 + 1;
        vector<int> final(DS_test.size());
        for (size_t i=0;i<final.size();++i) final[i] = vote_sum[i] >= maj;

        report_metrics(final, DS_test);

        return 0;
    }

    /* ---------- 选 3 个库做 Test ---------- */
    vector<string> test_dirs = pick_test3(all_dirs, seed);
    vector<string> cv_dirs;
    for (auto& d: all_dirs)
        if (!count(test_dirs.begin(), test_dirs.end(), d))
            cv_dirs.push_back(d);

    cerr << "\n[Test ] "; for (auto& d:test_dirs) cerr<<d<<' ';
    cerr << "\n[Pool ] "; for (auto& d:cv_dirs)  cerr<<d<<' '; cerr<<"\n\n";

    /* ---------- 5-fold & 分数 ---------- */
    auto folds = make_cv5(cv_dirs, seed);
    vector<pair<double,string>> scores;           // (BalAcc, path)

    int fid = 0;
    for (auto& f : folds) {
        ++fid;
        string mp = "lgb_fold"+to_string(fid)+".txt";

        if (!skip_train) {                        // 需要训练
            auto DS_tr = build_subset(f.tr_dirs, ALL);
            auto DS_va = build_subset(f.val_dirs, ALL);
            if (DS_tr.empty()||DS_va.empty()){ logW("fold skip"); continue; }
            train_and_eval(DS_tr, mp, trees, depth, lr, subs, col, false);
            double bal = compute_bal_acc(mp, DS_va);
            cout << "[Fold"<<fid<<"] BalAcc="<<bal<<'\n';
            scores.push_back({bal, mp});
        } else {                                 // 直接加载现成模型
            double bal = compute_bal_acc(mp, build_subset(f.val_dirs, ALL));
            cout << "[Fold"<<fid<<"] (skip-train) BalAcc="<<bal<<'\n';
            scores.push_back({bal, mp});
        }
    }
    if (scores.size() < 3) { logE("need ≥3 models"); return 1; }

    /* ---------- 选最优一半 ---------- */
    sort(scores.rbegin(), scores.rend());
    size_t keep = (scores.size()+1)/2;
    vector<string> ens_models;
    for (size_t i=0;i<keep;++i) ens_models.push_back(scores[i].second);

    cout << "\n[INFO] ensemble with top "<<keep<<" model(s):\n";
    for (auto& m:ens_models) cout<<"  "<<m<<'\n';

    /* ---------- 预测 ---------- */
    auto DS_test = build_subset(test_dirs, ALL);
    vector<int> vote_sum(DS_test.size(), 0);
    for (auto& m: ens_models){
        auto p = predict_binary(m, DS_test);
        for(size_t i=0;i<p.size();++i) vote_sum[i]+=p[i];
    }
    const int maj = ens_models.size()/2 + 1;
    vector<int> final(DS_test.size());
    for(size_t i=0;i<final.size();++i) final[i] = vote_sum[i] >= maj;

    report_metrics(final, DS_test);
    return 0;
}
