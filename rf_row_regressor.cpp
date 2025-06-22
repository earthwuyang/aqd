/************************************************************************************
 * train_rf_regressor.cpp
 *
 *  ▸ Bootstrap-based Random-Forest regressor (default 32 trees)
 *  ▸ Regularised CART for regression (minimize MSE)
 *  ▸ CLI flags (all optional):
 *      --data_dirs=DIR1,DIR2,...      • required – training folders
 *      --base=/path/to/query_costs    • default /home/wuy/query_costs
 *      --trees=32                     • forest size
 *      --max_depth=15                 • tree max depth
 *      --min_samples=40               • minimum samples per leaf
 *      --min_gain=1e-4                • minimum MSE reduction to split
 *      --model=rf_regressor.json      • output model file
 *
 ************************************************************************************/

#include <bits/stdc++.h>
#include "json.hpp"
#include <chrono>
#include <omp.h>
#include <mysql/mysql.h>
#include <regex>

using json = nlohmann::json;
using namespace std;

// ─────────── logging ───────────
static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }

// ─────────── progress bar ───────────
static void progress(const string& tag, size_t cur, size_t tot, size_t w=40) {
    double f = tot? double(cur)/tot : 1.0;
    size_t filled = size_t(f*w);
    cout << '\r' << tag << " ["
         << string(filled,'=') << string(w-filled,' ')
         << "] " << setw(3) << int(f*100) << "% ("
         << cur << '/' << tot << ')' << flush;
    if(cur==tot) cout<<"\n";
}

// ─────────── feature constants & helpers ───────────
constexpr int NUM_FEATS = 63;
static double safe_f(const json&v){
    if(v.is_number()) return v.get<double>();
    if(v.is_string()){ try{ return stod(v.get<string>());} catch(...){} }
    return 0.0;
}
static double safe_f(const json&obj,const char*key){
    return obj.contains(key)? safe_f(obj[key]) : 0.0;
}
static double log1p_clip(double v){ return log1p(max(0.0, v)); }

// ─────────── extract indexes from MySQL ───────────
static unordered_map<string, unordered_set<string>> indexCols;

static void load_index_defs_from_db(
    const string& host, int port,
    const string& user, const string& pass,
    const string& db, const string& table)
{
    MYSQL *conn = mysql_init(nullptr);
    if(!mysql_real_connect(conn, host.c_str(), user.c_str(), pass.c_str(),
                           db.c_str(), port, nullptr, 0))
    {
        logW("MySQL connect failed: " + string(mysql_error(conn)));
        return;
    }
    string q = "SHOW CREATE TABLE `" + table + "`";
    if(mysql_query(conn,q.c_str())){
        logW("SHOW CREATE TABLE failed: "+string(mysql_error(conn)));
        mysql_close(conn); return;
    }
    MYSQL_RES *res = mysql_store_result(conn);
    if(!res){ mysql_close(conn); return; }
    MYSQL_ROW row = mysql_fetch_row(res);
    string ddl = row && row[1] ? row[1] : "";
    mysql_free_result(res);
    mysql_close(conn);

    static const regex re(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))", regex::icase);
    auto it = sregex_iterator(ddl.begin(), ddl.end(), re);
    auto end = sregex_iterator();
    for(; it!=end; ++it){
        auto &m = *it;
        string idx = m[1].str();
        string cols = m[2].str();
        unordered_set<string> S;
        istringstream ss(cols); string tok;
        while(getline(ss,tok,',')){
            tok.erase(remove(tok.begin(),tok.end(),'`'),tok.end());
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t")+1);
            if(!tok.empty()) S.insert(tok);
        }
        indexCols[idx] = move(S);
    }
}

static void load_all_index_defs(
    const string& host, int port,
    const string& user, const string& pass,
    const vector<string>& dbs)
{
    for(auto &db: dbs){
        MYSQL *conn = mysql_init(nullptr);
        if(!mysql_real_connect(conn,host.c_str(),user.c_str(),pass.c_str(),
                               db.c_str(),port,nullptr,0))
        {
            logW("MySQL connect to "+db+" failed");
            mysql_close(conn); continue;
        }
        if(mysql_query(conn,("USE `"+db+"`").c_str())){ mysql_close(conn); continue;}
        if(mysql_query(conn,"SHOW TABLES")){ mysql_close(conn); continue;}
        MYSQL_RES *tables = mysql_store_result(conn);
        MYSQL_ROW row;
        while((row=mysql_fetch_row(tables))){
            load_index_defs_from_db(host,port,user,pass,db,row[0]);
        }
        mysql_free_result(tables);
        mysql_close(conn);
    }
}

// ─────────── plan → features ───────────
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

static bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    auto &v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){
        string s=v.get<string>(); transform(s.begin(),s.end(),s.begin(),::tolower);
        return s=="yes"||s=="true"||s=="1";
    }
    return false;
}

static double str_size_to_num(string s){
    if(s.empty()) return 0;
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    double m=1; char suf=s.back();
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return stod(s)*m;}catch(...){return 0;}
}



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

// ─────────── regression tree ───────────
struct RTNode {
    int feat=-1, left=-1, right=-1;
    float thr=0, value=0;
};
class RegressionTree {
    vector<RTNode> nodes_;
    int max_depth_, min_samples_;
    double min_gain_;
public:
    RegressionTree(int md,int ms,double mg)
      : max_depth_(md), min_samples_(ms), min_gain_(mg) {}

    int build(const vector<int>& idx,
              const vector<array<float,NUM_FEATS>>& X,
              const vector<double>& y,
              int depth)
    {
        // compute mean & mse
        double sum=0, sumsq=0;
        for(int i:idx){ sum+=y[i]; sumsq+=y[i]*y[i]; }
        int m = idx.size();
        double mean = sum/m;
        double mse  = sumsq/m - mean*mean;

        RTNode node;
        node.value = mean;
        int node_id = nodes_.size();
        nodes_.push_back(node);

        if(depth>=max_depth_ || m<=min_samples_) return node_id;

        // search best split
        int best_f=-1; float best_thr=0;
        double best_gain=0;
        for(int f=0;f<NUM_FEATS;f++){
            vector<float> vals;
            for(int i:idx) vals.push_back(X[i][f]);
            sort(vals.begin(),vals.end());
            vals.erase(unique(vals.begin(),vals.end()),vals.end());
            if(vals.size()<2) continue;
            int steps = min<int>(9, vals.size()-1);
            for(int s=1;s<=steps;s++){
                float thr = vals[vals.size()*s/(steps+1)];
                vector<int> L,R;
                for(int i:idx){
                    (X[i][f]<thr?L:R).push_back(i);
                }
                if((int)L.size()<min_samples_||(int)R.size()<min_samples_) continue;
                // compute child mse
                double sumL=0,sumsqL=0, sumR=0,sumsqR=0;
                for(int i:L){ sumL+=y[i]; sumsqL+=y[i]*y[i]; }
                for(int i:R){ sumR+=y[i]; sumsqR+=y[i]*y[i]; }
                double mL=L.size(), mR=R.size();
                double mseL = sumsqL/mL - (sumL/mL)*(sumL/mL);
                double mseR = sumsqR/mR - (sumR/mR)*(sumR/mR);
                double gain = mse - (mL/m)*(mseL) - (mR/m)*(mseR);
                if(gain>best_gain){ best_gain=gain; best_f=f; best_thr=thr; }
            }
        }

        if(best_gain<min_gain_) return node_id;

        // commit split
        nodes_[node_id].feat = best_f;
        nodes_[node_id].thr  = best_thr;

        vector<int> L,R;
        for(int i:idx){
            (X[i][best_f]<best_thr?L:R).push_back(i);
        }
        int l = build(L,X,y,depth+1);
        int r = build(R,X,y,depth+1);
        nodes_[node_id].left = l;
        nodes_[node_id].right= r;
        return node_id;
    }

    void fit(const vector<array<float,NUM_FEATS>>& X,
             const vector<double>& y)
    {
        nodes_.clear();
        vector<int> idx(X.size());
        iota(idx.begin(),idx.end(),0);
        build(idx, X, y, 0);
    }

    double predict_one(const float *f) const {
        int id=0;
        while(nodes_[id].feat>=0){
            id = (f[nodes_[id].feat] < nodes_[id].thr)
               ? nodes_[id].left : nodes_[id].right;
        }
        return nodes_[id].value;
    }

    json to_json() const {
        json arr = json::array();
        for(auto &n:nodes_){
            arr.push_back({
                {"feat", n.feat},
                {"thr",  n.thr},
                {"left", n.left},
                {"right",n.right},
                {"value",n.value}
            });
        }
        return arr;
    }

    void from_json(const json& arr){
        nodes_.clear();
        for(auto &e:arr){
            RTNode n;
            n.feat  = e["feat"];
            n.thr   = e["thr"];
            n.left  = e["left"];
            n.right = e["right"];
            n.value = e["value"];
            nodes_.push_back(n);
        }
    }
};

// ─────────── Random Forest regressor ───────────
class RFRegressor {
    vector<RegressionTree> trees_;
    double sampleRatio_;
public:
    RFRegressor(int n,int md,int ms,double mg,double sr)
      : sampleRatio_(sr)
    {
        trees_.reserve(n);
        for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
    }

    void fit(const vector<array<float,NUM_FEATS>>& X,
             const vector<double>& y)
    {
        mt19937 rng(42);
        int T = trees_.size();
        #pragma omp parallel for schedule(dynamic)
        for(int t=0;t<T;++t){
            auto &tree = trees_[t];
            mt19937 loc(rng()+t);
            uniform_int_distribution<int> uni(0,X.size()-1);
            int m = sampleRatio_*X.size();
            vector<array<float,NUM_FEATS>> BX; BX.reserve(m);
            vector<double> By; By.reserve(m);
            for(int i=0;i<m;++i){
                int j = uni(loc);
                BX.push_back(X[j]);
                By.push_back(y[j]);
            }
            tree.fit(BX,By);
            #pragma omp critical
            progress("RF-train", t+1, T);
        }
        logI("Forest trained");
    }

    double predict(const float *f) const {
        double sum=0;
        for(auto &tr:trees_) sum += tr.predict_one(f);
        return sum/trees_.size();
    }

    json to_json() const {
        json arr=json::array();
        for(auto &tr:trees_) arr.push_back(tr.to_json());
        return arr;
    }

    void from_json(const json& j){
        trees_.clear();
        for(auto &t: j){
            RegressionTree tr(0,0,0);
            tr.from_json(t);
            trees_.push_back(tr);
        }
    }
};

struct Sample {
    array<float,NUM_FEATS> feat;
    double row_t;
};

static bool load_plan(const string& path, array<float,NUM_FEATS>& f){
    ifstream in(path);
    if(!in) return false;
    json j;
    try{ in>>j; } catch(...){ return false; }
    return plan2feat(j,f.data());
}



int main(int argc, char* argv[]){
    // CLI defaults
    string baseDir    = "/home/wuy/query_costs";
    vector<string> dataDirs;
    int    nTrees     = 32;
    int    maxDepth   = 15;
    int    minSamples = 40;
    double minGain    = 1e-4;
    double sampleRatio= 0.6;
    string modelPath  = "rf_row_regressor.json";
    bool   skip_train = false;

    // parse flags
    for(int i=1;i<argc;++i){
        string a(argv[i]);
        if      (a.rfind("--data_dirs=",0)==0){
            string s=a.substr(12), tmp;
            stringstream ss(s);
            while(getline(ss,tmp,',')) dataDirs.push_back(tmp);
        }
        else if (a.rfind("--base=",0)==0)        baseDir    = a.substr(7);
        else if (a.rfind("--trees=",0)==0)       nTrees     = stoi(a.substr(8));
        else if (a.rfind("--max_depth=",0)==0)   maxDepth   = stoi(a.substr(12));
        else if (a.rfind("--min_samples=",0)==0) minSamples = stoi(a.substr(14));
        else if (a.rfind("--min_gain=",0)==0)    minGain    = stod(a.substr(11));
        else if (a.rfind("--sample_ratio=",0)==0)sampleRatio= stod(a.substr(14));
        else if (a.rfind("--model=",0)==0)       modelPath  = a.substr(8);
        else if (a == "--skip_train")            skip_train = true;
    }
    if(dataDirs.empty()){
        logE("need --data_dirs=...");
        return 1;
    }

    // load index definitions
    load_all_index_defs("127.0.0.1", 44444, "root", "", dataDirs);

    // load samples
    vector<Sample> DS;
    size_t skipped = 0;
    for(auto &d: dataDirs){
        ifstream fin(baseDir+"/"+d+"/query_costs.csv");
        if(!fin){ logW("missing CSV: "+d); continue; }
        string line; getline(fin,line);
        vector<string> lines;
        while(getline(fin,line)) lines.push_back(line);
        size_t cur=0, tot=lines.size();
        for(auto &l: lines){
            progress("load", ++cur, tot);
            string qid,lab,rt,ct,hyb,fann;
            stringstream ss(l);
            getline(ss,qid,','); getline(ss,lab,',');
            getline(ss,rt,',');  getline(ss,ct,',');
            getline(ss,hyb,','); getline(ss,fann,',');
            if(qid.empty()){ skipped++; continue; }
            Sample s;
            try { s.row_t = stod(rt); }
            catch(...) { skipped++; continue; }
            if(!load_plan(baseDir+"/"+d+"/row_plans/"+qid+".json", s.feat)){
                skipped++;
                continue;
            }
            DS.push_back(s);
        }
    }
    logI("Loaded "+to_string(DS.size())+" samples, skipped="+to_string(skipped));
    if(DS.size() < size_t(minSamples*2)){
        logE("too few samples");
        return 1;
    }

    // build X, y
    vector<array<float,NUM_FEATS>> X; X.reserve(DS.size());
    vector<double>                 Y; Y.reserve(DS.size());
    for(auto &s: DS){
        X.push_back(s.feat);
        Y.push_back(s.row_t);
    }

    // train or load forest
    RFRegressor rf(nTrees, maxDepth, minSamples, minGain, sampleRatio);
    if(!skip_train){
        rf.fit(X, Y);
        json outj; outj["forest"] = rf.to_json();
        ofstream out(modelPath);
        out << outj.dump(2);
        logI("Model saved → " + modelPath);
    } else {
        logI("Loading model from " + modelPath);
        ifstream in(modelPath);
        json inj; in >> inj;
        rf.from_json(inj["forest"]);
    }

    // evaluate MAE & RMSE
    double mae = 0, mse = 0;
    for(size_t i=0; i<DS.size(); ++i){
        double pred = rf.predict(X[i].data());
        double err  = pred - DS[i].row_t;
        mae += fabs(err);
        mse += err*err;
    }
    mae /= DS.size();
    mse = sqrt(mse / DS.size());
    cout << "\n=== REGRESSION METRICS ===\n"
         << fixed << setprecision(6)
         << "MAE  = " << mae << " sec\n"
         << "RMSE = " << mse << " sec\n";

    // compute Q-errors
    vector<double> qerrs;
    qerrs.reserve(DS.size());
    for(size_t i=0; i<DS.size(); ++i){
        double pred  = rf.predict(X[i].data());
        double truth = DS[i].row_t;
        if(pred>0 && truth>0){
            double q = pred/truth;
            if(q<1.0) q = 1.0/q;
            qerrs.push_back(q);
        }
    }
    sort(qerrs.begin(), qerrs.end());
    size_t N = qerrs.size();
    double q_min = qerrs.front();
    double q_max = qerrs.back();
    double q50   = qerrs[N/2];
    double q95   = qerrs[size_t(0.95 * N)];

    cout << "\n=== Q-ERROR METRICS ===\n"
         << fixed << setprecision(3)
         << "min Q-error     : " << q_min << "\n"
         << "50th percentile : " << q50  << "\n"
         << "95th percentile : " << q95  << "\n"
         << "max Q-error     : " << q_max << "\n";

    double sum_q = 0.0;
    for (double q : qerrs) {
        sum_q += q;
    }
    double q_mean = sum_q / qerrs.size();

    cout << "Mean Q-error    : " << fixed << setprecision(3)
        << q_mean << "\n";

    return 0;
}
