/************************************************************************************
 * train_dtree.cpp – Train a small CART decision-tree (32-feature) and export JSON
 * g++ -std=c++17 -O3 -I./json_single_include -o train_dtree train_dtree.cpp
 ************************************************************************************/

#include <bits/stdc++.h>
#include "json.hpp"
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


/* ────────────── lightweight CART with balanced weights ────────────── */
class DecisionTree{
    std::vector<DTNode> nodes_;

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
        /* ---------------------------------------------------------------- */
        /* special-case root: imitate the hand “cost > 50 k” threshold      */
        /* feature 40  =  raw query_cost  (see plan2feat)                   */
        /* ---------------------------------------------------------------- */
        if(depth == 0 && NUM_FEATS > 40){
            best_f   = 40;
            best_thr = 5e4f;          // 50 000
        }

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
        node.feat = best_f; node.thresh = best_thr;
        int self = int(nodes_.size());
        nodes_.push_back(node);               // placeholder

        int l = build(left ,X,y,w,depth+1,max_depth,min_samples);
        int r = build(right,X,y,w,depth+1,max_depth,min_samples);
        nodes_[self].left  = l;
        nodes_[self].right = r;
        return self;
    }
public:
    float predict(const float* f) const{
        int id = 0;
        while(nodes_[id].feat != -1)
            id = (f[nodes_[id].feat] < nodes_[id].thresh) ? nodes_[id].left
                                                           : nodes_[id].right;
        return nodes_[id].prob;
    }

    void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
             const std::vector<int>&  y,
             const std::vector<float>& w,
             int max_depth,int min_samples)
    {
        nodes_.clear();
        std::vector<int> idx(X.size()); std::iota(idx.begin(),idx.end(),0);
        build(idx,X,y,w,0,max_depth,min_samples);
    }

    void save(const std::string& path) const
    {
        json j = json::array();
        for(const auto& n : nodes_) j.push_back(n);
        std::ofstream(path) << j.dump(2);
    }
};

/* ────────────── tiny progress bar ────────────── */
static void progress(std::size_t cur,std::size_t tot,std::size_t w=50){
    double f = tot? double(cur)/tot : 1.0;
    std::size_t filled = std::size_t(f*w);
    std::cout << '\r' << '[' << std::string(filled,'=')
              << std::string(w-filled,' ') << "] "
              << std::setw(3) << int(f*100) << "% (" << cur << '/' << tot << ')'
              << std::flush;
    if(cur==tot) std::cout << '\n';
}


/* ───────────────────────────────  MAIN  ───────────────────────────── */
/* ───────────────────────────────  MAIN  ───────────────────────────── */
int main(int argc,char*argv[])
{
    /* ---------- CLI ---------- */
    std::string baseDir = "/home/wuy/query_costs";
    std::vector<std::string> dataDirs;
    std::string modelPath = "rowcol_dtree_new.json";
    int    max_depth   = 20;
    int    min_samples = 20;
    double gap_emph    = GAP_EMPHASIS;

    for(int i=1;i<argc;++i){
        std::string a(argv[i]);
        if      (a.rfind("--data_dirs=",0)==0)   dataDirs.push_back(a.substr(12));
        else if (a.rfind("--base=",0)==0)       baseDir    = a.substr(7);
        else if (a.rfind("--model=",0)==0)      modelPath  = a.substr(8);
        else if (a.rfind("--max_depth=",0)==0)  max_depth   = std::stoi(a.substr(12));
        else if (a.rfind("--min_samples=",0)==0)min_samples = std::stoi(a.substr(14));
        else if (a.rfind("--gap_emph=",0)==0)   gap_emph    = std::stod(a.substr(11));
    }
    if(dataDirs.empty()){
        logError("need --data_dirs=...");
        return 1;
    }

    /* ---------- load dataset ---------- */
    struct Sample{
        std::array<float,NUM_FEATS> feat;
        int    label;
        double row_t, col_t, qry_cost;
        bool   hyb_t;    // hybrid optimizer decision: 1=column, 0=row
    };
    std::vector<Sample> DS;
    std::size_t skipped = 0;

    auto load_plan = [&](const std::string& p,
                         std::array<float,NUM_FEATS>& f,
                         double& qcost)->bool{
        std::ifstream in(p);
        if(!in) return false;
        json j;
        try { in >> j; }
        catch(...){ return false; }
        if(!plan2feat(j,f.data())) return false;
        qcost = 0.0;
        if(j.contains("query_block"))
            qcost = safe_f(j["query_block"]
                             .value("cost_info", json::object()),
                           "query_cost");
        return true;
    };

    for(const auto& dirName: dataDirs){
        std::string csv = baseDir + '/' + dirName + "/query_costs.csv";
        std::string row = baseDir + '/' + dirName + "/row_plans/";
        std::ifstream fin(csv);
        if(!fin){
            logWarn("missing CSV: " + csv);
            continue;
        }

        std::string head; std::getline(fin, head);
        std::vector<std::string> lines;
        for(std::string l; std::getline(fin,l); )
            lines.push_back(std::move(l));

        std::size_t cur = 0, tot = lines.size();
        for(const auto& l: lines){
            progress(++cur, tot);
            std::stringstream ss(l);
            std::string qid, lab, rt, ct, hyb;
            std::getline(ss, qid, ',');
            std::getline(ss, lab, ',');
            std::getline(ss, rt,  ',');
            std::getline(ss, ct,  ',');
            std::getline(ss, hyb, ',');

            if(qid.empty() || lab.empty()){
                ++skipped;
                continue;
            }

            double rowT=0, colT=0;
            bool rowOk=false, colOk=false;
            try{ rowT = std::stod(rt);  rowOk = true; } catch(...){}
            try{ colT = std::stod(ct);  colOk = true; } catch(...){}

            // hyb is a 0/1 flag
            bool hybFlag = (hyb == "1");

            if(!colOk){
                ++skipped;
                continue;
            }
            if(!rowOk) rowT = DEFAULT_TIME;

            Sample s{};
            if(!load_plan(row + qid + ".json", s.feat, s.qry_cost)){
                ++skipped;
                continue;
            }

            s.label = std::stoi(lab);
            s.row_t = rowT;
            s.col_t = colT;
            s.hyb_t = hybFlag;
            DS.push_back(std::move(s));
        }
    }

    logInfo("usable samples = " + std::to_string(DS.size()) +
            ", skipped = " + std::to_string(skipped));
    if(DS.size() < std::size_t(min_samples*2)){
        logError("too few");
        return 1;
    }

    /* ---------- prepare ML arrays ---------- */
    std::vector<std::array<float,NUM_FEATS>> X;
    std::vector<int> Y;
    X.reserve(DS.size());
    Y.reserve(DS.size());
    for(const auto& s: DS){
        X.push_back(s.feat);
        Y.push_back(s.label == 1 ? 1 : 0);
    }

    /* ---------- compute gap‐weighted sample weights ---------- */
    std::size_t P=0, N=0;
    for(int y: Y) (y?P:N)++;
    const float S  = float(P+N);
    const float w1 = P? S/(2.f*P) : 1.f;
    const float w0 = N? S/(2.f*N) : 1.f;

    std::vector<float> W; W.reserve(DS.size());
    for(const auto& s: DS){
        double gap   = std::fabs(s.row_t - s.col_t);
        double base  = (s.label==1? w1 : w0);
        // double w_gap = 1.0 + gap_emph * gap/(s.row_t + s.col_t + 1e-6);
        // W.push_back(float(base * w_gap));
        double w_gap  = 1.0 + gap_emph * gap/(s.row_t + s.col_t + 1e-6);
        /* extra: make expensive mistakes hurt more ------------------- */
        double w_cost = 1.0 + std::min(s.qry_cost, 1e6) / 1e5;  // saturates at ×11
        W.push_back(float(base * w_gap * w_cost));
    }

    /* ---------- train & save tree ---------- */
    DecisionTree tree;
    tree.fit(X, Y, W, max_depth, min_samples);
    tree.save(modelPath);
    logInfo("model saved → " + modelPath);


    /* ---------- choose τ to minimise average runtime ---------- */
    double best_tau = 0.5, best_rt = 1e100;
    // for(double tau = 0.05; tau <= 0.95; tau += 0.01){
    //     double rt = 0; std::size_t n = 0;
    //     for(const auto& s : DS){
    //         bool col = tree.predict(s.feat.data()) >= tau;
    //         double t = col ? s.col_t : s.row_t;
    //         ++n;  rt += (t - rt)/n;
    //     }
    //     if(rt < best_rt){ best_rt = rt; best_tau = tau; }
    // }
    // logInfo("picked τ = " + std::to_string(best_tau) +
    //         "  (avg-runtime = " + std::to_string(best_rt) + " s)");

    /* ---------- evaluation @best_tau ---------- */


    /* ---------- evaluation ---------- */
    std::size_t TP=0, FP=0, TN=0, FN=0;
    for(const auto& s: DS){
        bool pred = tree.predict(s.feat.data()) >= best_tau;
        if(s.label==1 &&  pred) ++TP;
        if(s.label==0 &&  pred) ++FP;
        if(s.label==0 && !pred)++TN;
        if(s.label==1 && !pred)++FN;
    }
    auto div = [](double a,double b){ return b? a/b : 0; };
    double precision = div(TP, TP+FP),
           recall    = div(TP, TP+FN),
           accuracy  = div(TP+TN, TP+TN+FP+FN),
           f1        = (precision+recall)?
                          2*precision*recall/(precision+recall) : 0.0;


    std::cout<<"\n=== CONFUSION MATRIX @0.5 ===\n"
             <<"TP="<<TP<<"  FP="<<FP
             <<"  TN="<<TN<<"  FN="<<FN<<"\n"
             <<"Acc="<<accuracy
             <<"  Prec="<<precision
             <<"  Rec="<<recall
             <<"  F1="<<f1<<"\n\n";

    /* ---------- average runtimes ---------- */
    double rt_row=0, rt_col=0, rt_rule=0, rt_ai=0, rt_opt=0, rt_hyb=0;
    std::size_t n=0;
    for(const auto& s: DS){
        ++n;
        rt_row   += (s.row_t - rt_row)/n;
        rt_col   += (s.col_t - rt_col)/n;
        double rule = (s.qry_cost > 5e4 ? s.col_t : s.row_t);
        rt_rule  += (rule    - rt_rule)/n;
        double ai  = (tree.predict(s.feat.data()) >= best_tau
                       ? s.col_t : s.row_t);
        rt_ai    += (ai      - rt_ai)/n;
        double hyb = s.hyb_t ? s.col_t : s.row_t;
        rt_hyb   += (hyb     - rt_hyb)/n;
        double opt = std::min(s.row_t, s.col_t);
        rt_opt   += (opt     - rt_opt)/n;
    }

    std::cout<<"=== AVG RUNTIME (s) ===\n"
             <<"Row only        : "<<rt_row  <<'\n'
             <<"Column only     : "<<rt_col  <<'\n'
             <<"Cost threshold  : "<<rt_rule <<'\n'
             <<"Decision tree   : "<<rt_ai   <<'\n'
             <<"Hybrid optimizer: "<<rt_hyb  <<'\n'
             <<"Optimal oracle  : "<<rt_opt  <<'\n';

    return 0;
}


