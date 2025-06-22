//------------------------------------------------------------------------------
// run_fann_prediction_enhanced.cpp
//
// • Connects to MySQL, runs EXPLAIN FORMAT=JSON on one query
// • Extracts the **same 32-dimensional feature vector** used during training
// • Scales each feature into [0,1] using the per-column maxima/minima that
//   were learned (row_plan_statistics.json) – falls back to hard-coded caps
//   when the JSON is missing.
// • Feeds the scaled vector into the 32 → hidden → 1 FANN network
// • Prints the probability and the row-vs-column decision
//
//   g++ -std=c++11 -lfann -lmysqlclient -o run_fann_prediction_enhanced \
//       run_fann_prediction_enhanced.cpp
//------------------------------------------------------------------------------

#include <mysql/mysql.h>
#include <fann.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cctype>
#include "json.hpp"
using json = nlohmann::json;

/* -------------------------------------------------------------------------- */
static void logWarn (const std::string& s){ std::cerr<<"[WARN] "<<s<<'\n'; }
static void logError(const std::string& s){ std::cerr<<"[ERR ] "<<s<<'\n'; }

constexpr int NUM_FEATS = 32;

/* -------------------------------------------------------------------------- */
/*                    ❶  feature scaling helpers (0-1)                       */
/* -------------------------------------------------------------------------- */
static double featMin[NUM_FEATS]{};
static double featMax[NUM_FEATS]{};
static bool   scalingLoaded=false;

/* names that appear in row_plan_statistics.json (nullptr = derived features) */
static const char* STAT_NAME[NUM_FEATS] = {
  /* 0-6  */ "rows_examined_per_scan","rows_produced_per_join","filtered",
             "read_cost","eval_cost","prefix_cost","data_read_per_join",
  /* 7-11 */ nullptr,nullptr,nullptr,nullptr,nullptr,
  /* 12   */ nullptr,                       // using-idx
  /* 13-15*/ "filtered","filtered","filtered",
  /* 16-17*/ nullptr,nullptr,               // depth / fan-out
  /* 18-20*/ nullptr,nullptr,nullptr,       // grp / ord / tmp
  /* 21-22*/ "read_cost","read_cost",       // ratio mean/max
  /* 23   */ "query_cost",
  /* 24   */ "rows_produced_per_join",      // root cardinality
  /* 25-27*/ nullptr,nullptr,nullptr,       // const / limit / distinct
  /* 28-29*/ nullptr,nullptr,               // union flag & #union
  /* 30-31*/ nullptr,nullptr                // cross features
};

/* pessimistic hard caps if stats file missing */
static const double DEFAULT_MAX[NUM_FEATS] = {
  /* 0-6  */ 20,20,10,20,20,20,30,
  /* 7-11 */ 1,1,1,1,1,
  /* 12   */ 1,
  /* 13-15*/ 5,0,5,
  /* 16-17*/ 10,10,
  /* 18-20*/ 1,1,1,
  /* 21-22*/ 10,10,
  /* 23   */ 30,
  /* 24   */ 20,
  /* 25-27*/ 1,1,1,
  /* 28-29*/ 1,10,
  /* 30-31*/ 10,10
};

static bool fileExists(const std::string& p){ std::ifstream f(p); return f.good(); }

/* load per-feature min/max from JSON (only max matters; min=0) */
static void loadScaling()
{
    if (scalingLoaded) return;
    for (int i = 0; i < NUM_FEATS; ++i) {
        featMin[i] = 0.0;
        featMax[i] = DEFAULT_MAX[i];
    }
    const std::string statPath = "/home/wuy/query_costs/hybench_sf1/row_plan_statistics.json";
    if (!fileExists(statPath)) {
        logWarn("stats file missing – using hard-coded caps");
        scalingLoaded = true;
        return;
    }
    json st;
    try { std::ifstream(statPath) >> st; }
    catch (...) {
        logWarn("stats JSON parse error – using defaults");
        scalingLoaded = true;
        return;
    }
    for (int i = 0; i < NUM_FEATS; ++i) {
        if (!STAT_NAME[i]) continue;
        const std::string key(STAT_NAME[i]);
        if (st.contains(key))
            featMax[i] = std::max(st[key].value("max", DEFAULT_MAX[i]), 1e-6);
    }
    scalingLoaded = true;
}

/* clamp & scale raw→[0,1] */
static float scaleFeat(int idx, double v)
{
    if (!scalingLoaded) loadScaling();
    v = std::min(std::max(v, featMin[idx]), featMax[idx]);
    return static_cast<float>((v - featMin[idx]) /
                              (featMax[idx] - featMin[idx] + 1e-12));
}

/* -------------------------------------------------------------------------- */
/*                 ❷  helpers for feature extraction                          */
/* -------------------------------------------------------------------------- */
double convert_data_size_to_numeric(std::string s)
{
    if (s.empty()) return 0.0;
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    if (s.empty()) return 0.0;
    double f = 1.0; char suf = s.back();
    if (suf=='G'||suf=='g'){ f=1e9; s.pop_back(); }
    else if (suf=='M'||suf=='m'){ f=1e6; s.pop_back(); }
    else if (suf=='K'||suf=='k'){ f=1e3; s.pop_back(); }
    try { return std::stod(s) * f; } catch (...) { return 0.0; }
}

inline double safeLog1p(double v){ return std::log1p(std::max(0.0, v)); }

double parseNumber(const json &j, const std::string &k)
{
    if (!j.contains(k)) return 0.0;
    try {
        if (j[k].is_string()) return std::stod(j[k].get<std::string>());
        if (j[k].is_number()) return j[k].get<double>();
    } catch (...) {}
    return 0.0;
}

struct FeatureAgg {
    double reSum=0,rpSum=0,fSum=0,rcSum=0,ecSum=0,pcSum=0,drSum=0;
    double selMin=1e30, selMax=0, selSum=0;
    double ratioSum=0, ratioMax=0;
    int    cnt=0;
    int cntRange=0,cntRef=0,cntEq=0,cntIdx=0,cntFull=0;
    int usingIdx=0;
    int maxDepth=0; double fanoutMax=0;
    bool hasGroup=false, hasOrder=false, hasTemp=false;
    /* new for 32 dims */
    bool hasConst=false, hasLimit=false, hasDistinct=false, hasUnion=false;
    int  numUnion=0;
};

void recursePlan(const json &node, FeatureAgg &A, int depth)
{
    if (node.is_object()) {
        if (node.contains("table") && node["table"].is_object()) {
            const auto &tbl = node["table"];
            double re = parseNumber(tbl, "rows_examined_per_scan");
            double rp = parseNumber(tbl, "rows_produced_per_join");
            double f  = parseNumber(tbl, "filtered");
            json ci = tbl.value("cost_info", json::object());
            double rc = parseNumber(ci, "read_cost");
            double ec = parseNumber(ci, "eval_cost");
            double pc = parseNumber(ci, "prefix_cost");
            double dr = ci.contains("data_read_per_join")
                      ? (ci["data_read_per_join"].is_string()
                         ? convert_data_size_to_numeric(ci["data_read_per_join"].get<std::string>())
                         : parseNumber(ci, "data_read_per_join"))
                      : 0.0;
            A.reSum+=re; A.rpSum+=rp; A.fSum+=f;
            A.rcSum+=rc; A.ecSum+=ec; A.pcSum+=pc; A.drSum+=dr; A.cnt++;
            if (re>0) {
                double sel = rp/re;
                A.selSum+=sel;
                A.selMin = std::min(A.selMin, sel);
                A.selMax = std::max(A.selMax, sel);
                A.fanoutMax = std::max(A.fanoutMax, sel);
            }
            double ratio = ec>0 ? rc/ec : rc;
            A.ratioSum+=ratio;
            A.ratioMax=std::max(A.ratioMax, ratio);

            std::string at = tbl.value("access_type","all");
            if      (at=="range")  A.cntRange++;
            else if (at=="ref")    A.cntRef++;
            else if (at=="eq_ref") A.cntEq++;
            else if (at=="index")  A.cntIdx++;
            else                   A.cntFull++;
            if (tbl.value("using_index",false)) A.usingIdx++;

            if (at=="const")       A.hasConst = true;
        }
        if (node.contains("grouping_operation"))         A.hasGroup    = true;
        if (node.contains("ordering_operation"))         A.hasOrder    = true;
        if (node.value("using_filesort",false))          A.hasOrder    = true;
        if (node.value("using_temporary_table",false))   A.hasTemp     = true;
        if (node.contains("limit"))                      A.hasLimit    = true;
        if (node.contains("distinct"))                   A.hasDistinct = true;
        if (node.contains("union_result")) {              A.hasUnion    = true; A.numUnion++; }

        for (auto &kv : node.items())
            if (kv.key() != "table")
                recursePlan(kv.value(), A, depth+1);
    }
    else if (node.is_array()) {
        for (auto &el : node) recursePlan(el, A, depth);
    }
    A.maxDepth = std::max(A.maxDepth, depth);
}

bool extractFeaturesFromPlan(const json &plan, double raw[NUM_FEATS])
{
    if (!plan.contains("query_block")) return false;
    FeatureAgg A;
    recursePlan(plan["query_block"], A, 1);
    double qCost = parseNumber(plan["query_block"]
                    .value("cost_info", json::object()), "query_cost");

    std::fill(raw, raw+NUM_FEATS, 0.0);
    if (A.cnt == 0) return true;
    double inv = 1.0 / A.cnt; int k = 0;

    raw[k++] = safeLog1p(A.reSum * inv);
    raw[k++] = safeLog1p(A.rpSum * inv);
    raw[k++] = safeLog1p(A.fSum  * inv);
    raw[k++] = safeLog1p(A.rcSum * inv);
    raw[k++] = safeLog1p(A.ecSum * inv);
    raw[k++] = safeLog1p(A.pcSum * inv);
    raw[k++] = safeLog1p(A.drSum * inv);

    raw[k++] = A.cntRange * inv;
    raw[k++] = A.cntRef   * inv;
    raw[k++] = A.cntEq    * inv;
    raw[k++] = A.cntIdx   * inv;
    raw[k++] = A.cntFull  * inv;

    raw[k++] = A.usingIdx * inv;

    raw[k++] = A.selSum * inv;
    raw[k++] = A.selMin;
    raw[k++] = A.selMax;

    raw[k++] = A.maxDepth;
    raw[k++] = A.fanoutMax;

    raw[k++] = A.hasGroup ? 1.0 : 0.0;
    raw[k++] = A.hasOrder ? 1.0 : 0.0;
    raw[k++] = A.hasTemp  ? 1.0 : 0.0;

    raw[k++] = A.ratioSum * inv;
    raw[k++] = A.ratioMax;

    raw[k++] = safeLog1p(qCost);              // feature 23

    /* 24-31 new */
    raw[k++] = safeLog1p(parseNumber(plan["query_block"], "rows_produced_per_join"));
    raw[k++] = A.hasConst    ? 1.0 : 0.0;
    raw[k++] = A.hasLimit    ? 1.0 : 0.0;
    raw[k++] = A.hasDistinct ? 1.0 : 0.0;
    raw[k++] = A.hasUnion    ? 1.0 : 0.0;
    raw[k++] = std::min(A.numUnion, 10);
    double selMean  = A.selSum * inv;
    double fullFrac = (A.cntFull ? A.cntFull * inv : 1e-3);
    raw[k++] = selMean * A.maxDepth;
    raw[k++] = (A.usingIdx * inv) / fullFrac;

    return true;
}

/* -------------------------------------------------------------------------- */
/*                               ❸  main                                      */
/* -------------------------------------------------------------------------- */
int main()
{
    // 1) connect
    const char *host="127.0.0.1", *user="root", *pass="", *db="member_log";
    unsigned int port=44444;
    MYSQL *conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn,host,user,pass,db,port,nullptr,0)) {
        logError(mysql_error(conn)); return 1;
    }

    // 2) run EXPLAIN FORMAT=JSON your query
    const char *qry =
        "EXPLAIN FORMAT=JSON SELECT COUNT(DISTINCT MemberAcctId) as Count from  member_log WHERE ShopId = 14823 AND IsGoShop = 1 AND CreateTime BETWEEN timestamp('2025-02-18 00:00:00.000000') AND timestamp('2025-02-18 23:59:59.999000');";
    if (mysql_query(conn,qry)) {
        logError(mysql_error(conn)); mysql_close(conn); return 1;
    }
    MYSQL_RES *res = mysql_store_result(conn);
    MYSQL_ROW row  = mysql_fetch_row(res);
    unsigned long *len = mysql_fetch_lengths(res);
    std::string jsonText(row[0], len[0]);

    // 3) parse JSON
    json plan;
    try { plan = json::parse(jsonText); }
    catch (...) { logError("JSON parse error"); return 1; }

    // 4) extract raw features
    double raw[NUM_FEATS];
    if (!extractFeaturesFromPlan(plan, raw)) {
        logError("feature extraction failed"); return 1;
    }

    // 5) scale
    fann_type input[NUM_FEATS];
    std::cout<<"[DEBUG] Scaled features (0-1):\n";
    for(int i=0;i<NUM_FEATS;i++){
        input[i] = scaleFeat(i, raw[i]);
        std::cout<<"  f["<<i<<"] = "<<input[i]<<'\n';
    }

    // 6) run FANN
    const char *modelPath = "checkpoints/best_mlp_enhanced.net";
    struct fann *ann = fann_create_from_file(modelPath);
    if (!ann){ logError("cannot load model"); return 1; }
    fann_type *out = fann_run(ann, input);
    if (!out){ logError("fann_run failed"); return 1; }

    std::cout<<"Probability(col) = "<<out[0]
             <<"  →  "<< (out[0] >= 0.5f ? "Column Store" : "Row Store") <<'\n';

    // cleanup
    fann_destroy(ann);
    mysql_free_result(res);
    mysql_close(conn);
    return 0;
}
