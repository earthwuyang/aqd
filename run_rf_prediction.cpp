// run_rf_prediction.cpp
// mirrors kernel feature extractor with covering‐index lookup and debug prints

#include <bits/stdc++.h>
#include <mysql/mysql.h>
#include <regex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

/* ────────────── logging helpers ────────────── */
static void logInfo (const string& s){ cout  << "[INFO]  " << s << '\n'; }
static void logWarn (const string& s){ cerr  << "[WARN]  " << s << '\n'; }
static void logError(const string& s){ cerr  << "[ERROR] " << s << '\n'; }

/* ────────────── MySQL‐based index loader ────────────── */
static unordered_map<string, unordered_set<string>> indexCols;

static void load_index_defs_from_db(const string &host,
                                    int port,
                                    const string &user,
                                    const string &pass,
                                    const string &db,
                                    const string &table_name)
{
    MYSQL *conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn,
                            host.c_str(), user.c_str(), pass.c_str(),
                            db.c_str(), port, nullptr, 0))
    {
        logWarn("MySQL connect failed: " + string(mysql_error(conn)));
        return;
    }
    string q = "SHOW CREATE TABLE `" + table_name + "`";
    if (mysql_query(conn, q.c_str()))
    {
        logWarn("SHOW CREATE TABLE failed: " + string(mysql_error(conn)));
        mysql_close(conn);
        return;
    }
    MYSQL_RES *res = mysql_store_result(conn);
    MYSQL_ROW row = res ? mysql_fetch_row(res) : nullptr;
    string ddl = (row && row[1]) ? row[1] : "";
    mysql_free_result(res);
    mysql_close(conn);

    regex r(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))", regex::icase);
    for (sregex_iterator it(ddl.begin(), ddl.end(), r), end; it != end; ++it)
    {
        string idx  = (*it)[1].str();
        string cols = (*it)[2].str();
        unordered_set<string> S;
        string tok;
        stringstream ss(cols);
        while (getline(ss, tok, ','))
        {
            tok.erase(remove(tok.begin(), tok.end(), '`'), tok.end());
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t")+1);
            if (!tok.empty()) S.insert(tok);
        }
        indexCols.emplace(idx, move(S));
    }
    logInfo("Loaded " + to_string(indexCols.size())
         + " indexes from `" + table_name + "`");
}

/* ─────────── feature constants ─────────── */
constexpr int NUM_FEATS  = 63;
constexpr double GAP_EMPH=2.0;

/* ─────────── numeric helpers ─────────── */
static double safe_f(const json& v) {
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) {
        try { return stod(v.get<string>()); } catch(...) {}
    }
    return 0.0;
}
static double safe_f(const json& obj, const char* key) {
    return obj.contains(key) ? safe_f(obj[key]) : 0.0;
}
static inline double log1p_clip(double v) { return log1p(max(0.0, v)); }
static double str_size_to_num(string s) {
    if (s.empty()) return 0.0;
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    double m = 1; char suf = s.back();
    if (suf=='G'||suf=='g'){ m=1e9; s.pop_back(); }
    else if(suf=='M'||suf=='m'){ m=1e6; s.pop_back(); }
    else if(suf=='K'||suf=='k'){ m=1e3; s.pop_back(); }
    try { return stod(s)*m; } catch(...) { return 0.0; }
}
static bool getBool(const json& o, const char* k) {
    if (!o.contains(k)) return false;
    const auto& v = o[k];
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_string()) {
        string s = v.get<string>();
        transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s=="yes"||s=="true"||s=="1";
    }
    return false;
}
// ─────────── per-plan DFS aggregation ───────────
struct Agg {
    double re=0, rp=0, f=0, rc=0, ec=0, pc=0, dr=0;
    double selSum=0, selMin=1e30, selMax=0, ratioSum=0, ratioMax=0;
    int cnt=0;
    int cRange=0, cRef=0, cEq=0, cIdx=0, cFull=0, idxUse=0;
    int sumPossibleKeys=0;
    double maxPrefix=0, minRead=1e30;
    double fanoutMax=0;
    bool grp=false, ord=false, tmp=false;
    int coverCount=0;
    // new used_columns stats
    int usedColsSum=0, usedColsMax=0;
    double usedColsRatioSum=0, usedColsRatioMax=0;
    int maxDepth=0;  // ← ensure this field is present
};



// ─────────── per-plan DFS aggregation ───────────
static void walk(const json& n, Agg& a, int depth)
{
    if (n.is_object()) {
        if (n.contains("table") && n["table"].is_object()) {
            const auto& t  = n["table"];
            const auto& ci = t.value("cost_info", json::object());

            double re = safe_f(t, "rows_examined_per_scan");
            double rp = safe_f(t, "rows_produced_per_join");
            double fl = safe_f(t, "filtered");
            double rc = safe_f(ci, "read_cost");
            double ec = safe_f(ci, "eval_cost");
            double pc = safe_f(ci, "prefix_cost");
            double dr = 0;
            if (ci.contains("data_read_per_join")) {
                const auto& v = ci["data_read_per_join"];
                dr = v.is_string()
                   ? str_size_to_num(v.get<string>())
                   : safe_f(v);
            }

            a.re += re;  a.rp += rp;  a.f  += fl;
            a.rc += rc;  a.ec += ec;  a.pc += pc;  a.dr += dr;
            a.cnt++;

            if (t.contains("possible_keys") && t["possible_keys"].is_array())
                a.sumPossibleKeys += int(t["possible_keys"].size());
            a.maxPrefix = max(a.maxPrefix, pc);
            a.minRead   = min(a.minRead, rc < 1e30 ? rc : 1e30);

            if (re > 0) {
                double sel = rp / re;
                a.selSum   += sel;
                a.selMin    = min(a.selMin, sel);
                a.selMax    = max(a.selMax, sel);
                a.fanoutMax = max(a.fanoutMax, sel);
            }
            double ratio = (ec > 0) ? rc / ec : rc;
            a.ratioSum += ratio;
            a.ratioMax  = max(a.ratioMax, ratio);

            string at = t.value("access_type", "ALL");
            if      (at == "range")  a.cRange++;
            else if (at == "ref")    a.cRef++;
            else if (at == "eq_ref") a.cEq++;
            else if (at == "index")  a.cIdx++;
            else                     a.cFull++;
            if (getBool(t, "using_index")) a.idxUse++;

            // covering‐index check
            if (t.contains("used_columns") && t["used_columns"].is_array()
             && t.contains("key")         && t["key"].is_string())
            {
                string idxName = t["key"].get<string>();
                auto it = indexCols.find(idxName);
                if (it != indexCols.end()) {
                    bool fullCover = true;
                    for (auto &u : t["used_columns"]) {
                        if (!u.is_string() ||
                            !it->second.count(u.get<string>()))
                        {
                            fullCover = false;
                            break;
                        }
                    }
                    if (fullCover) a.coverCount++;
                }
            }
        }

        if (n.contains("grouping_operation"))                  a.grp = true;
        if (n.contains("ordering_operation") || getBool(n,"using_filesort")) a.ord = true;
        if (getBool(n, "using_temporary_table"))               a.tmp = true;

        for (const auto& kv : n.items())
            if (kv.key() != "table")
                walk(kv.value(), a, depth + 1);
    }
    else if (n.is_array()) {
        for (const auto& el : n)
            walk(el, a, depth);
    }

    a.maxDepth = max(a.maxDepth, depth);
}


// ─────────── updated plan2feat() w/ 63 features ───────────
static bool plan2feat(const json& plan, float f[NUM_FEATS])
{
    if (!plan.contains("query_block")) return false;

    const json* qb = &plan["query_block"];
    if (qb->contains("union_result")) {
        const auto& specs = (*qb)["union_result"]["query_specifications"];
        if (specs.is_array() && !specs.empty())
            qb = &specs[0]["query_block"];
    }

    Agg a{};
    walk(*qb, a, 1);
    if (a.cnt == 0) return false;

    double inv     = 1.0 / a.cnt;
    int    k       = 0;
    double qCost   = safe_f(qb->value("cost_info", json::object()), "query_cost");
    double rootRow = safe_f(*qb, "rows_produced_per_join");

    // 0–6: basic costs/cardinals
    f[k++] = log1p_clip(a.re * inv);
    f[k++] = log1p_clip(a.rp * inv);
    f[k++] = log1p_clip(a.f  * inv);
    f[k++] = log1p_clip(a.rc * inv);
    f[k++] = log1p_clip(a.ec * inv);
    f[k++] = log1p_clip(a.pc * inv);
    f[k++] = log1p_clip(a.dr * inv);

    // 7–12: access-type fractions
    f[k++] = a.cRange * inv;
    f[k++] = a.cRef   * inv;
    f[k++] = a.cEq    * inv;
    f[k++] = a.cIdx   * inv;
    f[k++] = a.cFull  * inv;
    f[k++] = a.idxUse * inv;

    // 13–17: selectivity stats & shape
    f[k++] = a.selSum * inv;
    f[k++] = a.selMin;
    f[k++] = a.selMax;
    f[k++] = a.maxDepth;
    f[k++] = a.fanoutMax;

    // 18–20: boolean flags
    f[k++] = a.grp ? 1.f : 0.f;
    f[k++] = a.ord ? 1.f : 0.f;
    f[k++] = a.tmp ? 1.f : 0.f;

    // 21–22: cost ratios
    f[k++] = a.ratioSum * inv;
    f[k++] = a.ratioMax;

    // 23–24: query cost & root rows
    f[k++] = log1p_clip(qCost);
    f[k++] = log1p_clip(rootRow);

    // 25–27: per-cost cost/cardinal ratios
    f[k++] = log1p_clip((a.pc * inv) / max(1e-6, a.rc * inv));
    f[k++] = log1p_clip((a.rc * inv) / max(1e-6, a.re * inv));
    f[k++] = log1p_clip((a.ec * inv) / max(1e-6, a.re * inv));

    // 28–29: single vs multi-block
    f[k++] = (a.cnt == 1) ? 1.f : 0.f;
    f[k++] = (a.cnt >  1) ? 1.f : 0.f;

    // 30–31: deeper interactions
    f[k++] = log1p_clip(a.maxDepth * (a.idxUse * inv));
    {
        double fullFrac = a.cFull ? (a.cFull * inv) : 1e-3;
        f[k++] = log1p_clip((a.idxUse * inv) / fullFrac);
    }

    // 32–39: structural features
    f[k++] = float(a.cnt);
    f[k++] = a.cnt ? float(a.sumPossibleKeys) / a.cnt : 0.f;
    f[k++] = log1p_clip(a.maxPrefix);
    f[k++] = log1p_clip(a.minRead < 1e30 ? a.minRead : 0.0);
    f[k++] = (a.cnt > 1) ? float(a.cnt - 1) / a.cnt : 0.f;
    f[k++] = (rootRow > 0) ? float(a.re) / rootRow : 0.f;
    f[k++] = float(a.selMax - a.selMin);
    {
        float denom = float(a.cRange + a.cRef + a.cEq + a.cIdx);
        f[k++]   = denom ? float(a.idxUse) / denom : 0.f;
    }

    // 40–41: raw cost & threshold flag
    f[k++] = float(qCost);
    f[k++] = (qCost > 5e4) ? 1.f : 0.f;

    // 42–43: covering-index coverage
    f[k++] = a.cnt ? float(a.coverCount) / a.cnt : 0.f;
    f[k++] = (a.coverCount == a.cnt) ? 1.f : 0.f;

    // 44: sequentiality proxy
    f[k++] = log1p_clip(a.re * inv) - log1p_clip(a.selSum * inv);

    // 45: raw table count & log(table count)
    f[k++] = float(a.cnt);
    f[k++] = log1p_clip(a.cnt);

    // 47–48: possible-keys stats
    f[k++] = float(a.sumPossibleKeys);
    f[k++] = a.cnt ? float(a.sumPossibleKeys) / a.cnt : 0.f;

    // 49–50: cover-count stats
    f[k++] = float(a.coverCount);
    f[k++] = a.cnt ? float(a.coverCount) / a.cnt : 0.f;

    // 51–56: fine-grained access-type ratios & index usage
    f[k++] = a.idxUse * inv;
    f[k++] = a.cRange * inv;
    f[k++] = a.cRef   * inv;
    f[k++] = a.cEq    * inv;
    f[k++] = a.cIdx   * inv;
    f[k++] = a.cFull  * inv;

    // 57–59: prefix/read skew & selectivity range
    f[k++] = log1p_clip(a.maxPrefix * inv);
    f[k++] = log1p_clip(a.minRead < 1e30 ? a.minRead : 0.0);
    f[k++] = float(a.selMax - a.selMin);

    // 60: maximum ratio
    f[k++] = float(a.ratioMax);

    // 61: fanout skew
    f[k++] = float(a.fanoutMax);

    // 62: selectivity skew (max/min)
    f[k++] = a.selMin > 0 ? float(a.selMax / a.selMin) : 0.f;

    // sanity check
    assert(k == NUM_FEATS);
    return true;
}

/* ─────────── RF loader & predictor ─────────── */
struct RFNode { int feat,left,right; float thresh,prob; };
static vector<vector<RFNode>> FOREST;

static bool load_forest(const string& p) {
    ifstream in(p); if(!in) return false;
    json j; in >> j;
    if(!j.contains("forest")) return false;
    FOREST.clear();
    for(auto& t: j["forest"]) {
        vector<RFNode> tree;
        for(auto& n: t) {
            RFNode nd;
            nd.feat   = n.value("feat",-1);
            nd.thresh = n.value("thr", n.value("thresh",0.f));
            nd.left   = n.value("left",-1);
            nd.right  = n.value("right",-1);
            nd.prob   = n.value("prob",0.f);
            tree.push_back(nd);
        }
        FOREST.push_back(move(tree));
    }
    return !FOREST.empty();
}

static float predict_rf(const float f[NUM_FEATS]) {
    // if (f[43]==1.0f) return 0.0f;
    double sum = 0;
    for (auto& tree: FOREST) {
        int id = 0;
        while (tree[id].feat != -1) {
            id = (f[tree[id].feat] < tree[id].thresh)
               ? tree[id].left
               : tree[id].right;
        }
        sum += tree[id].prob;
    }
    return float(sum / FOREST.size());
}

/* ─────────── main ─────────── */
int main(int argc, char** argv) {
    // MySQL + config
    string host="127.0.0.1", user="root", pass="", db="task_info", tbl="task_info";
    int port=44444;
    string model = "/home/wuy/row_column_routing/rf_model_more_features.json";

    // 1) load index definitions
    load_index_defs_from_db(host, port, user, pass, db, tbl);

    // 2) get EXPLAIN JSON
    MYSQL* conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn, host.c_str(), user.c_str(), pass.c_str(),
                            tbl.c_str(), port, nullptr, 0))
    {
        logError(mysql_error(conn));
        return 1;
    }
    mysql_query(conn, "SET use_imci_engine=off;");
    const char* qry =
      "EXPLAIN FORMAT=JSON "
      "SELECT count(*) AS cnt,module_code "
      "FROM task_info "
      "WHERE env='online' "
      "AND assignee_uid IN('2206529173543','cn011000001612304') "
      "AND platform_code IN('A@42gBZ','A@RhHWV');";
    if (mysql_query(conn, qry)) {
        logError(mysql_error(conn));
        mysql_close(conn);
        return 1;
    }
    MYSQL_RES* res = mysql_store_result(conn);
    MYSQL_ROW  row = mysql_fetch_row(res);
    unsigned long* len = mysql_fetch_lengths(res);
    string js = row && len ? string(row[0], len[0]) : "";
    mysql_free_result(res);
    mysql_close(conn);

    // 3) parse JSON
    json plan;
    try { plan = json::parse(js); }
    catch(...) {
        logError("JSON parse error");
        return 1;
    }

    // 4) extract features
    float feat[NUM_FEATS] = {};
    if (!plan2feat(plan, feat)) {
        logError("plan2feat failed");
        return 1;
    }
    for (int i = 0; i < NUM_FEATS; ++i)
        cout << "feat["<<setw(2)<<i<<"]="<<feat[i]<<"\n";

    // 5) load model & predict
    if (!load_forest(model)) {
        logError("cannot load RF model");
        return 1;
    }
    float p = predict_rf(feat);
    cout << "P(column)="<<p<<" -> "<<(p>=0.5f?"COLUMN":"ROW")<<"\n";

    return 0;
}
