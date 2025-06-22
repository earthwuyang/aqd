/************************************************************************************
 * rf_even_more.cpp  -– full source
 *
 *  ▸ Bootstrap-based Random-Forest (default 32 trees)
 *  ▸ Regularised CART (depth penalty, min-samples, min-gain)
 *  ▸ 68-dim feature vector  ← 63 old  + 5 new (see plan2feat())
 *  ▸ Gap- and cost-aware sample weighting
 *  ▸ Optional training skipped with --skip_train
 *
 *  CLI flags (all optional):
 *    --data_dirs=DIR1,DIR2,...      • required – training folders
 *    --base=/path/to/query_costs    • default /home/wuy/query_costs
 *    --trees=32                     • forest size
 *    --max_depth=15                 • CART regularisation
 *    --min_samples=40
 *    --min_gain=0.0005
 *    --sample_ratio=0.6
 *    --model=rf_model.json
 *    --skip_train
 *
 ************************************************************************************/

#include <bits/stdc++.h>
#include "json.hpp"
#include <chrono>                   // ⬅ timing
#include <omp.h>
#include <mysql/mysql.h>
#include <regex>

using json = nlohmann::json;
using namespace std;

/* ───────── logging ───────── */
static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }

/* ───────── progress-bar ───────── */
static void progress(const string& tag,size_t cur,size_t tot,size_t w=40)
{
    double f = tot? double(cur)/tot : 1.0;
    size_t filled = size_t(f*w);
    cout << '\r' << tag << " ["
         << string(filled,'=') << string(w-filled,' ')
         << "] " << setw(3) << int(f*100) << "% (" << cur << '/' << tot << ')'
         << flush;
    if(cur==tot) cout << '\n';
}

/* ─────────────────────── global INDEX dictionary ─────────────────────── */
static unordered_map<string, unordered_set<string>> indexCols;

/* ---------- helper: pull SHOW CREATE TABLE & parse indexes ---------- */
static void load_index_defs_from_db(const string &host,int port,
                                    const string &user,const string &pass,
                                    const string &db,const string &tbl)
{
    MYSQL *conn = mysql_init(nullptr);
    if(!mysql_real_connect(conn,host.c_str(),user.c_str(),pass.c_str(),
                           db.c_str(),port,nullptr,0))
    { logW("MySQL connect fail: "+string(mysql_error(conn))); return; }

    string q = "SHOW CREATE TABLE `" + tbl + "`";
    if(mysql_query(conn,q.c_str()))
    { logW("SHOW CREATE TABLE fail: "+string(mysql_error(conn)));
      mysql_close(conn); return; }

    MYSQL_RES *res = mysql_store_result(conn);
    if(!res){ mysql_close(conn); return; }
    MYSQL_ROW row = mysql_fetch_row(res);
    if(!row||!row[1]){ mysql_free_result(res); mysql_close(conn); return; }
    string ddl = row[1];
    mysql_free_result(res); mysql_close(conn);

    static const regex re(R"(KEY\s+`([^`]+)`\s*\(\s*([^)]+)\))",
                          regex::icase);
    smatch m; auto it=ddl.cbegin(), ed=ddl.cend();
    while(regex_search(it,ed,m,re))
    {
        string idx=m[1].str(), cols=m[2].str();
        unordered_set<string> S;
        istringstream ss(cols); string tok;
        while(getline(ss,tok,',')){
            tok.erase(remove(tok.begin(),tok.end(),'`'),tok.end());
            tok.erase(0,tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t")+1);
            if(!tok.empty()) S.insert(tok);
        }
        if(!S.empty()) indexCols.emplace(move(idx),move(S));
        it = m.suffix().first;
    }
}

/* ---------- load indexes for every TABLE in every DB ---------- */
static void load_all_index_defs(const string &host,int port,const string& user,
                                const string& pass,const vector<string>& dbs)
{
    indexCols.clear();
    for(const auto& db:dbs){
        MYSQL *c=mysql_init(nullptr);
        if(!mysql_real_connect(c,host.c_str(),user.c_str(),pass.c_str(),
                               db.c_str(),port,nullptr,0))
        { logW("MySQL conn "+db+" fail"); mysql_close(c); continue; }
        if(mysql_query(c,"SHOW TABLES"))
        { mysql_close(c); continue; }
        MYSQL_RES *rs=mysql_store_result(c);
        MYSQL_ROW r;
        while((r=mysql_fetch_row(rs))){
            load_index_defs_from_db(host,port,user,pass,db,r[0]);
        }
        mysql_free_result(rs); mysql_close(c);
    }
}

/* ───────── constants & helpers ───────── */
constexpr int    NUM_FEATS  = 72;   // 63 old + 5 new
constexpr double GAP_EMPH   = 2.0;

static double safe_f(const json&v){
    if(v.is_number()) return v.get<double>();
    if(v.is_string()) { try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
static double safe_f(const json&o,const char*key){
    return o.contains(key)? safe_f(o[key]) : 0.0;
}
static double log1p_clip(double v){ return log1p(max(0.0,v)); }
static double str_size_to_num(string s){
    if(s.empty()) return 0.0;
    while(!s.empty()&&isspace((unsigned char)s.back())) s.pop_back();
    double m=1; char suf=s.back();
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return stod(s)*m;}catch(...){return 0.0;}
}
static bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    const auto&v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){
        string s=v.get<string>(); transform(s.begin(),s.end(),s.begin(),::tolower);
        return s=="yes"||s=="true"||s=="1";
    }
    return false;
}

/* ───────── DFS aggregation struct ───────── */
/* ─────────── per-plan aggregation record ───────────
 * Every field is additive or “max/flag” so the DFS walk( ) can
 * just update in-place.  All values start at the neutral element. */
struct Agg
{
    /* ❶ base cost & cardinality sums (later averaged with inv = 1/cnt) */
    double re = 0.0;     // rows_examined_per_scan   Σ
    double rp = 0.0;     // rows_produced_per_join   Σ
    double f  = 0.0;     // filtered (%)             Σ
    double rc = 0.0;     // read_cost                Σ
    double ec = 0.0;     // eval_cost                Σ
    double pc = 0.0;     // prefix_cost              Σ
    double dr = 0.0;     // data_read_per_join (B)   Σ
    double drSum = 0.0; // Σ (no normalisation) – filled below
    double drMax = 0.0; //max bytes read by *one* table

    /* ❷ selectivity & cost-ratio statistics */
    double selSum  = 0.0;     // Σ selectivity (rp/re)
    double selMin  = 1e30;    // min selectivity
    double selMax  = 0.0;     // max selectivity
    double ratioSum= 0.0;     // Σ rc/ec (per table)
    double ratioMax= 0.0;     // max rc/ec

    /* ❸ counters */
    int cnt        = 0;       // #tables visited
    int cRange     = 0;
    int cRef       = 0;
    int cEq        = 0;
    int cIdx       = 0;
    int cFull      = 0;
    int idxUse     = 0;       // USING INDEX flag count
    int sumPossibleKeys = 0;  // total possible_keys entries

    /* ❹ extremes / maxima / minima */
    double maxPrefix = 0.0;   // largest prefix_cost
    double minRead   = 1e30;  // smallest read_cost
    double fanoutMax = 0.0;   // biggest rp/re

    /* ❺ plan-shape / operator flags */
    bool grp = false;         // any GROUP BY
    bool ord = false;         // any ORDER BY / filesort
    bool tmp = false;         // uses temporary table
    bool grpTmp = false;   // new

    /* ❻ covering-index statistics */
    int coverCount = 0;       // #tables fully covered by chosen index

    /* ❼ placeholders for future column-usage stats */
    int    usedColsSum        = 0;
    int    usedColsMax        = 0;
    double usedColsRatioSum   = 0.0;
    double usedColsRatioMax   = 0.0;

    /* ❽ depth of deepest subtree (set in walk()) */
    int maxDepth = 0;

    
};


/* ───────── walk plan tree ───────── */
static void walk(const json& n,Agg&a,int depth){
    if(n.is_object()){
        if(n.contains("table")&&n["table"].is_object()){
            const auto&t=n["table"];
            const auto&ci=t.value("cost_info",json::object());
            double re=safe_f(t,"rows_examined_per_scan");
            double rp=safe_f(t,"rows_produced_per_join");
            double fl=safe_f(t,"filtered");
            double rc=safe_f(ci,"read_cost");
            double ec=safe_f(ci,"eval_cost");
            double pc=safe_f(ci,"prefix_cost");
            double dr=0;
            if(ci.contains("data_read_per_join")){
                const auto&v=ci["data_read_per_join"];
                dr=v.is_string()?str_size_to_num(v.get<string>()) : safe_f(v);
            }
            a.re+=re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr; a.drSum+=dr; a.drMax = std::max(a.drMax, dr); a.cnt++;

            if(t.contains("possible_keys")&&t["possible_keys"].is_array())
                a.sumPossibleKeys+=int(t["possible_keys"].size());
            a.maxPrefix=max(a.maxPrefix,pc);
            a.minRead=min(a.minRead,rc<1e30?rc:1e30);

            if(re>0){
                double sel=rp/re;
                a.selSum+=sel; a.selMin=min(a.selMin,sel); a.selMax=max(a.selMax,sel);
                a.fanoutMax=max(a.fanoutMax,sel);
            }
            double ratio=(ec>0)? rc/ec : rc;
            a.ratioSum+=ratio; a.ratioMax=max(a.ratioMax,ratio);

            string at=t.value("access_type","ALL");
            if(at=="range") a.cRange++; else if(at=="ref") a.cRef++;
            else if(at=="eq_ref") a.cEq++; else if(at=="index") a.cIdx++; else a.cFull++;
            if(getBool(t,"using_index")) a.idxUse++;

            /* covering index check */
            if(t.contains("used_columns")&&t["used_columns"].is_array()
               && t.contains("key")&&t["key"].is_string()){
                string idx=t["key"].get<string>();
                auto it=indexCols.find(idx);
                if(it!=indexCols.end()){
                    bool cover=true;
                    for(auto&u:t["used_columns"]){
                        if(!u.is_string()||!it->second.count(u.get<string>()))
                        { cover=false; break; }
                    }
                    if(cover) a.coverCount++;
                }
            }
        }
        if(n.contains("grouping_operation")) a.grp=true;
        if(n.contains("ordering_operation")||getBool(n,"using_filesort")) a.ord=true;
        if(getBool(n,"using_temporary_table")) a.tmp=true;
        if(a.grp && a.tmp)                                a.grpTmp=true;

        for(const auto&kv:n.items())
            if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }else if(n.is_array()){
        for(const auto&el:n) walk(el,a,depth);
    }
    a.maxDepth=max(a.maxDepth,depth);
}


/* ───────── plan → feature vector ───────── */
static bool plan2feat(const json &plan, float f[NUM_FEATS])
{
    /* ----------   sanity & union-flattener   ---------- */
    if (!plan.contains("query_block")) return false;
    const json &rootQB = plan["query_block"];

    double qCost   = 0.0;
    double rootRow = 0.0;
    std::vector<const json *> blocks;

    if (rootQB.contains("union_result"))
    {
        const auto &specs = rootQB["union_result"]["query_specifications"];
        if (!specs.is_array() || specs.empty()) return false;
        for (const auto &sp : specs)
        {
            const json &qb = sp["query_block"];
            blocks.push_back(&qb);
            qCost   += safe_f(qb.value("cost_info", json::object()), "query_cost");
            rootRow += safe_f(qb, "rows_produced_per_join");
        }
    }
    else
    {
        blocks.push_back(&rootQB);
        qCost   = safe_f(rootQB.value("cost_info", json::object()), "query_cost");
        rootRow = safe_f(rootQB, "rows_produced_per_join");
    }

    /* ----------   aggregate   ---------- */
    Agg a{};
    for (const json *qb : blocks) walk(*qb, a, 1);
    if (a.cnt == 0) return false;

    const double inv = 1.0 / a.cnt;
    int k = 0;

    /* ----------   retained high-value features   ---------- */
    f[k++] = log1p_clip(a.re * inv);                 //  0
    f[k++] = log1p_clip(a.rp * inv);                 //  1
    f[k++] = log1p_clip(a.f  * inv);                 //  2
    f[k++] = log1p_clip(a.rc * inv);                 //  3
    f[k++] = log1p_clip(a.ec * inv);                 //  4
    f[k++] = log1p_clip(a.pc * inv);                 //  5
    f[k++] = log1p_clip(a.dr * inv);                 //  6
    f[k++] = a.cRange * inv;                         //  7
    f[k++] = a.cRef   * inv;                         //  8
    f[k++] = a.cEq    * inv;                         //  9
    f[k++] = a.cIdx   * inv;                         // 10
    f[k++] = a.cFull  * inv;                         // 11
    f[k++] = a.idxUse * inv;                         // 12
    f[k++] = a.selSum * inv;                         // 13
    f[k++] = a.selMin;                               // 14
    f[k++] = a.selMax;                               // 15
    f[k++] = a.maxDepth;                             // 16
    f[k++] = a.fanoutMax;                            // 17
    f[k++] = a.grp;                                  // 18
    f[k++] = a.tmp;                                  // 19
    f[k++] = a.ratioSum * inv;                       // 20
    f[k++] = a.ratioMax;                             // 21
    f[k++] = log1p_clip((a.pc*inv) / std::max(1e-6, a.rc*inv));   // 22
    f[k++] = log1p_clip((a.rc*inv) / std::max(1e-6, a.re*inv));   // 23
    f[k++] = log1p_clip((a.ec*inv) / std::max(1e-6, a.re*inv));   // 24
    f[k++] = log1p_clip(a.maxDepth*(a.idxUse*inv));               // 25
    { double fullFrac = a.cFull ? a.cFull*inv : 1e-3;
      f[k++] = log1p_clip((a.idxUse*inv) / fullFrac); }           // 26
    f[k++] = float(a.cnt);                                        // 27
    f[k++] = a.cnt ? float(a.sumPossibleKeys)/a.cnt : 0.f;        // 28
    f[k++] = log1p_clip(a.minRead < 1e30 ? a.minRead : 0.0);      // 29
    f[k++] = log1p_clip(a.re*inv) - log1p_clip(a.selSum*inv);     // 30
    f[k++] = float(a.idxUse) / std::max(1, a.cRange+a.cRef+a.cEq+a.cIdx); //31
    f[k++] = float(a.sumPossibleKeys);                            // 32
    f[k++] = float(a.coverCount);                                 // 33
    f[k++] = a.cnt ? float(a.coverCount)/a.cnt : 0.f;             // 34
    f[k++] = log1p_clip(a.maxPrefix);                             // 35
    f[k++] = log1p_clip(a.maxPrefix * inv);                       // 36
    f[k++] = log1p_clip(a.drMax / 1048576.0);                     // 37
    f[k++] = log1p_clip(a.drSum / 1048576.0);                     // 38
    f[k++] = (a.cnt > 1);                                         // 39
    f[k++] = float(a.selMax - a.selMin);                          // 40
    f[k++] = float(a.ratioMax);                                   // 41
    f[k++] = float(a.fanoutMax);                                  // 42
    f[k++] = a.selMin>0 ? float(a.selMax/a.selMin) : 0.f;         // 43
    f[k++] = log1p_clip(qCost);                                   // 44
    f[k++] = log1p_clip(rootRow);                                 // 45
    f[k++] = float(qCost > 5e4);                                  // 46
    f[k++] = a.grpTmp ? 1.f : 0.f;                                // 47

    /* ----------   brand-new predictive features   ---------- */
    const double cost_per_row = qCost / std::max(1.0, rootRow);
    f[k++] = float(cost_per_row);                                 // 48  (raw)
    f[k++] = float(cost_per_row > 10000.0);                       // 49  flag
    f[k++] = float(a.maxPrefix > 50000.0);                        // 50  flag
    double tmp_io_ratio = (a.tmp && a.drMax>0)?
        std::log1p( (a.drMax/1048576.0) / (qCost + 1e-6) ) : 0.0;
    f[k++] = float(tmp_io_ratio);                                 // 51

    /* ----------   sanity   ---------- */
    static_assert(NUM_FEATS == 72, "update NUM_FEATS / feature list");
    if (k != NUM_FEATS) return false;
    return true;
}



/* ───────── CART tree ───────── */
struct DTNode{ int feat=-1,left=-1,right=-1; float thr=0,prob=0; };
class DecisionTree{
    vector<DTNode> nodes_;
    double min_gain_;
    int max_depth_,min_samples_;
    /* depth-penalised Gini */
    static inline double gini(double pos,double tot){
        if(tot<=0) return 1.0;
        double q=pos/tot; return 2.0*q*(1.0-q);
    }
    int build(const vector<int>& idx,
              const vector<array<float,NUM_FEATS>>&X,
              const vector<int>& y,const vector<float>& w,
              int depth)
    {
        double pos_w=0,tot_w=0;
        for(int i:idx){ tot_w+=w[i]; pos_w+=w[i]*y[i]; }
        DTNode node; node.prob=float(pos_w/max(1e-12,tot_w));
        if(depth>=max_depth_||idx.size()<=min_samples_||
           node.prob<=0.02||node.prob>=0.98){
            nodes_.push_back(node); return int(nodes_.size())-1;
        }

        int best_f=-1; float best_thr=0,best_g=FLT_MAX;
        for(int f=0;f<NUM_FEATS;++f){
            vector<float> vals; vals.reserve(idx.size());
            for(int i:idx) vals.push_back(X[i][f]);
            sort(vals.begin(),vals.end());
            vals.erase(unique(vals.begin(),vals.end()),vals.end());
            if(vals.size()<2) continue;
            vector<float> cand;
            int steps=min<int>(9,vals.size()-1);
            for(int s=1;s<=steps;++s) cand.push_back(vals[vals.size()*s/(steps+1)]);
            for(float thr:cand){
                double l_tot=0,l_pos=0,r_tot=0,r_pos=0;
                int l_cnt=0,r_cnt=0;
                for(int i:idx){
                    if(X[i][f]<thr){ l_tot+=w[i]; l_pos+=w[i]*y[i]; ++l_cnt;}
                    else           { r_tot+=w[i]; r_pos+=w[i]*y[i]; ++r_cnt;}
                }
                if(l_cnt<min_samples_||r_cnt<min_samples_) continue;
                float g=gini(l_pos,l_tot)+gini(r_pos,r_tot)+0.002f*depth; // depth penalty
                if(g<best_g){ best_g=g; best_f=f; best_thr=thr; }
            }
        }
        if(best_f==-1){ nodes_.push_back(node); return int(nodes_.size())-1; }

        vector<int> left,right;
        for(int i:idx){
            (X[i][best_f]<best_thr? left:right).push_back(i);
        }
        node.feat=best_f; node.thr=best_thr;
        int self=int(nodes_.size()); nodes_.push_back(node);
        int l=build(left ,X,y,w,depth+1);
        int r=build(right,X,y,w,depth+1);
        nodes_[self].left=l; nodes_[self].right=r;
        return self;
    }
public:
    DecisionTree(int md,int ms,double mg)
      :min_gain_(mg),max_depth_(md),min_samples_(ms){}
    void fit(const vector<array<float,NUM_FEATS>>&X,
             const vector<int>&y,const vector<float>&w){
        nodes_.clear(); vector<int> idx(X.size());
        iota(idx.begin(),idx.end(),0); build(idx,X,y,w,0);
    }
    float predict(const float*f)const{
        int id=0;
        while(nodes_[id].feat!=-1)
            id=(f[nodes_[id].feat]<nodes_[id].thr)?nodes_[id].left:nodes_[id].right;
        return nodes_[id].prob;
    }
    json to_json()const{
        json arr=json::array();
        for(const auto&n:nodes_)
            arr.push_back({{"feat",n.feat},{"thr",n.thr},
                           {"left",n.left},{"right",n.right},{"prob",n.prob}});
        return arr;
    }
    void from_json(const json&arr){
        nodes_.clear();
        for(const auto&e:arr){
            DTNode n; n.feat=e.at("feat"); n.thr=e.at("thr");
            n.left=e.at("left"); n.right=e.at("right"); n.prob=e.at("prob");
            nodes_.push_back(n);
        }
    }
};

/* ───────── Random-Forest ───────── */
class RandomForest{
    vector<DecisionTree> trees_;
    double sampleRatio_;
public:
    RandomForest(int n,int md,int ms,double mg,double sr):sampleRatio_(sr){
        trees_.reserve(n); for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
    }
    void fit(const vector<array<float,NUM_FEATS>>&X,
             const vector<int>&y,const vector<float>&w,mt19937&rng)
    {
        size_t T=trees_.size();
        #pragma omp parallel for schedule(dynamic)
        for(int t=0;t<int(T);++t){
            mt19937 rng_local(rng()+t);
            uniform_int_distribution<int> uni(0,int(X.size()-1));
            size_t m=size_t(sampleRatio_*X.size());
            vector<array<float,NUM_FEATS>> BX; BX.reserve(m);
            vector<int> By; By.reserve(m); vector<float> Bw; Bw.reserve(m);
            for(size_t i=0;i<m;++i){
                int j=uni(rng_local);
                BX.push_back(X[j]); By.push_back(y[j]); Bw.push_back(w[j]);
            }
            trees_[t].fit(BX,By,Bw);
            #pragma omp critical
            progress("RF-train",t+1,T);
        }
    }
    float predict(const float*f)const{
        double s=0; for(const auto&tr:trees_) s+=tr.predict(f);
        return float(s/trees_.size());
    }
    json to_json()const{ json arr=json::array();
        for(const auto&tr:trees_) arr.push_back(tr.to_json()); return arr; }
    void from_json(const json&arr){
        trees_.clear();
        for(const auto&jt:arr){ DecisionTree tr(0,0,0); tr.from_json(jt);
            trees_.push_back(move(tr)); }
    }
};

/* ───────── sample struct ───────── */
struct Sample{
    array<float,NUM_FEATS> feat{};
    int label=0; double row_t=0,col_t=0,qcost=0; bool hyb=0,fann=0;
};

/* ───────── plan loader ───────── */
static bool load_plan(const string&path,array<float,NUM_FEATS>&f,double&qcost){
    ifstream in(path); if(!in) return false;
    json j; try{ in>>j;}catch(...){return false;}
    if(!plan2feat(j,f.data())) return false;
    qcost=j.contains("query_block")? safe_f(j["query_block"].value("cost_info",json::object()),"query_cost"):0;
    return true;
}

// /* ───────── main ───────── */
// int main(int argc,char*argv[]){
//     string baseDir="/home/wuy/query_costs";
//     vector<string> dataDirs;
//     int nTrees=32,maxDepth=15,minSamples=40;
//     double minGain=0.0005,sampleRatio=0.6;
//     bool skip_train=false; string modelPath="rf_model_even_more.json";
//     double delta=0.2,costThreshold=5e4;
//     string mysql_host="127.0.0.1"; int mysql_port=44444;
//     string mysql_user="root", mysql_pass="";

//     for(int i=1;i<argc;++i){
//         string a(argv[i]);
//         if(a.rfind("--data_dirs=",0)==0){ string s=a.substr(12),tmp; stringstream ss(s);
//             while(getline(ss,tmp,',')) dataDirs.push_back(tmp);}
//         else if(a.rfind("--base=",0)==0) baseDir=a.substr(7);
//         else if(a.rfind("--trees=",0)==0) nTrees=stoi(a.substr(8));
//         else if(a.rfind("--max_depth=",0)==0) maxDepth=stoi(a.substr(12));
//         else if(a.rfind("--min_samples=",0)==0) minSamples=stoi(a.substr(14));
//         else if(a.rfind("--min_gain=",0)==0) minGain=stod(a.substr(11));
//         else if(a.rfind("--sample_ratio=",0)==0) sampleRatio=stod(a.substr(14));
//         else if(a.rfind("--model=",0)==0) modelPath=a.substr(8);
//         else if(a=="--skip_train") skip_train=true;
//         else if(a.rfind("--mysql_host=",0)==0) mysql_host=a.substr(13);
//         else if(a.rfind("--mysql_port=",0)==0) mysql_port=stoi(a.substr(13));
//         else if(a.rfind("--mysql_user=",0)==0) mysql_user=a.substr(13);
//         else if(a.rfind("--mysql_pass=",0)==0) mysql_pass=a.substr(13);
//     }
//     if(dataDirs.empty()){ logE("need --data_dirs"); return 1; }
//     if(sampleRatio<=0||sampleRatio>1){ logE("sample_ratio in (0,1]"); return 1;}

//     load_all_index_defs(mysql_host,mysql_port,mysql_user,mysql_pass,dataDirs);

//     vector<Sample> DS; size_t skipped=0;
//     for(auto&d:dataDirs){
//         ifstream fin(baseDir+"/"+d+"/query_costs.csv");
//         if(!fin){ logW("missing CSV "+d); continue; }
//         string line; getline(fin,line);
//         vector<string> lines; while(getline(fin,line)) lines.push_back(line);
//         size_t cur=0,tot=lines.size();
//         for(auto&l:lines){
//             progress("load",++cur,tot);
//             stringstream ss(l); string qid,lab,rt,ct,gap,use,hyb,fann,sql;
//             getline(ss,qid,','); getline(ss,lab,',');
//             getline(ss,rt,','); getline(ss,ct,','); getline(ss,gap,',');
//             getline(ss,use,','); getline(ss,hyb,','); getline(ss,fann,',');
//             getline(ss,sql); // rest of line

//             if(qid.empty()){ skipped++; continue;}
//             Sample s; try{s.row_t=stod(rt);}catch(...){s.row_t=60;}
//             try{s.col_t=stod(ct);}catch(...){skipped++; continue;}
//             s.label=(lab=="1"); s.hyb=(hyb=="1"); s.fann=(fann=="1");
//             string planPath=baseDir+"/"+d+"/row_plans/"+qid+".json";
//             double qc; if(!load_plan(planPath,s.feat,qc)) s.qcost=0; else s.qcost=qc;
//             DS.push_back(s);
//         }
//     }
//     logI("Loaded "+to_string(DS.size())+" samples, skipped "+to_string(skipped));
//     if(DS.size()<size_t(minSamples*2)){ logE("too few samples"); return 1; }

//     vector<array<float,NUM_FEATS>> X; vector<int> Y; vector<float> W;
//     double P=0,N=0; for(auto&s:DS) s.label?++P:++N; double S=P+N;
//     double w1=S/(2*P), w0=S/(2*N);

//     for(auto&s:DS){
//         X.push_back(s.feat); Y.push_back(s.label);
//         double gap=fabs(s.row_t-s.col_t);
//         double base=s.label? w1:w0;
//         double wgap=1+GAP_EMPH*pow(gap/(s.row_t+s.col_t+1e-6),1.5);
//         // double wcost=1+min(s.qcost,1e6)/1e5;
//         double wscan = 1 + log1p( (s.col_t + s.row_t) * 0.2 );   // ~20 % of runtime
//         double wcost = wscan;
//         W.push_back(base*wgap*wcost);
//     }

//     RandomForest rf(nTrees,maxDepth,minSamples,minGain,sampleRatio);
//     mt19937 rng(42);
//     if(!skip_train){
//         rf.fit(X,Y,W,rng);
//         json j; j["forest"]=rf.to_json();
//         ofstream(modelPath)<<j.dump(2); logI("model saved "+modelPath);
//     }else{
//         ifstream in(modelPath); json j; in>>j; rf.from_json(j["forest"]);
//     }

//     size_t TP=0,FP=0,TN=0,FN=0; double rt_row=0,rt_col=0,rt_rule=0,rt_rf=0,rt_hyb=0,rt_fann=0,rt_opt=0; size_t n=0;
//     for(auto&s:DS){
//         ++n; rt_row+=(s.row_t-rt_row)/n; rt_col+=(s.col_t-rt_col)/n;
//         double rule=s.qcost>costThreshold? s.col_t:s.row_t;
//         rt_rule+=(rule-rt_rule)/n;
//         float p=rf.predict(s.feat.data());
//         bool dec=(fabs(p-0.5f)<delta? (s.qcost>costThreshold):(p>=0.5f));
//         rt_rf+=((dec?s.col_t:s.row_t)-rt_rf)/n;
//         rt_hyb+=((s.hyb?s.col_t:s.row_t)-rt_hyb)/n;
//         rt_fann+=((s.fann?s.col_t:s.row_t)-rt_fann)/n;
//         rt_opt+=(min(s.row_t,s.col_t)-rt_opt)/n;
//         if(s.label&&dec) ++TP; else if(!s.label&&dec) ++FP;
//         else if(!s.label&&!dec) ++TN; else ++FN;
//     }
//     double prec=TP?double(TP)/(TP+FP):0, rec=TP?double(TP)/(TP+FN):0;
//     double f1=(prec+rec)?2*prec*rec/(prec+rec):0;
//     cout<<"\n=== CONFUSION (RF) ===\nTP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN
//         <<"\nPrecision="<<prec<<" Recall="<<rec<<" F1="<<f1<<"\n\n";
//     cout<<"=== AVG RUNTIME (s) ===\nRow          : "<<rt_row
//         <<"\nColumn       : "<<rt_col
//         <<"\nCost thresh  : "<<rt_rule
//         <<"\nRandom Forest: "<<rt_rf
//         <<"\nHybrid opt   : "<<rt_hyb
//         <<"\nFANN model   : "<<rt_fann
//         <<"\nOptimal      : "<<rt_opt<<'\n';
//     return 0;
// }


int main(int argc, char* argv[]) {
    // ─ CLI defaults ─
    string baseDir       = "/home/wuy/query_costs";
    vector<string> dataDirs;
    int    nTrees        = 32;
    int    maxDepth      = 15;
    int    minSamples    = 40;
    double minGain       = 0.0005;
    double sampleRatio   = 0.6;
    bool   skip_train    = false;
    string modelPath     = "rf_model_more_features.json";
    double delta         = 0.2;
    double costThreshold = 5e4;

    // MySQL for index defs
    string mysql_host = "127.0.0.1";
    int    mysql_port = 44444;
    string mysql_user = "root";
    string mysql_pass = "";
    // (we now pass the list of dataDirs as databases)
    
    // ─ Parse flags ─
    for (int i = 1; i < argc; ++i) {
        string a(argv[i]);
        if (a.rfind("--data_dirs=", 0) == 0) {
            string s = a.substr(12), tmp;
            stringstream ss(s);
            while (getline(ss, tmp, ',')) dataDirs.push_back(tmp);
        } else if (a.rfind("--base=", 0) == 0)        baseDir       = a.substr(7);
        else if (a.rfind("--trees=", 0) == 0)        nTrees        = stoi(a.substr(8));
        else if (a.rfind("--max_depth=", 0) == 0)    maxDepth      = stoi(a.substr(12));
        else if (a.rfind("--min_samples=", 0) == 0)  minSamples    = stoi(a.substr(14));
        else if (a.rfind("--min_gain=", 0) == 0)     minGain       = stod(a.substr(11));
        else if (a.rfind("--sample_ratio=", 0) == 0) sampleRatio   = stod(a.substr(14));
        else if (a.rfind("--model=", 0) == 0)        modelPath     = a.substr(8);
        else if (a == "--skip_train")                skip_train    = true;
        else if (a.rfind("--mysql_host=", 0) == 0)   mysql_host    = a.substr(13);
        else if (a.rfind("--mysql_port=", 0) == 0)   mysql_port    = stoi(a.substr(13));
        else if (a.rfind("--mysql_user=", 0) == 0)   mysql_user    = a.substr(13);
        else if (a.rfind("--mysql_pass=", 0) == 0)   mysql_pass    = a.substr(13);
    }

    if (dataDirs.empty()) {
        logE("need --data_dirs=...");
        return 1;
    }
    if (sampleRatio <= 0 || sampleRatio > 1) {
        logE("sample_ratio must be in (0,1]");
        return 1;
    }

    // ─ Load full index definitions for all databases in dataDirs ─
    load_all_index_defs(mysql_host,
                        mysql_port,
                        mysql_user,
                        mysql_pass,
                        dataDirs);

    // ─ Load samples ─
    struct Sample {
        array<float,NUM_FEATS> feat;
        int    label;
        double row_t, col_t, qcost;
        bool   hyb;
        bool   fann;    // ← NEW: FANN-model decision flag
    };
    vector<Sample> DS;
    size_t skipped = 0;

    for (auto& d: dataDirs) {
        ifstream fin(baseDir + "/" + d + "/query_costs.csv");
        if (!fin) {
            logW("missing CSV: " + d);
            continue;
        }
        string line;
        getline(fin, line);  // header
        vector<string> lines;
        while (getline(fin, line)) lines.push_back(line);

        size_t cur = 0, tot = lines.size();
        for (auto& l: lines) {
            progress("load", ++cur, tot);
            stringstream ss(l);
            string qid, lab, rt, ct, hyb, fann;
            getline(ss, qid,  ','); 
            getline(ss, lab,  ',');
            getline(ss, rt,   ',');
            getline(ss, ct,   ',');
            getline(ss, hyb,  ',');
            getline(ss, fann, ',');

            if (qid.empty()) { skipped++; continue; }
            Sample s;
            try { s.row_t = stod(rt); } catch(...) { s.row_t = 60.0; }
            try { s.col_t = stod(ct); } catch(...) { skipped++; continue; }
            s.label = (lab  == "1");
            s.hyb   = (hyb  == "1");
            s.fann  = (fann == "1");  // ← NEW

            string planPath = baseDir + "/" + d + "/row_plans/" + qid + ".json";
            double qc;
            if (!load_plan(planPath, s.feat, qc)) {
                s.qcost = 0;
            } else {
                s.qcost = qc;
            }

            DS.push_back(s);
        }
    }
    logI("loaded " + to_string(DS.size()) + " samples, skipped=" + to_string(skipped));
    if (DS.size() < size_t(minSamples * 2)) {
        logE("too few samples");
        return 1;
    }

    // ─ Build features/labels/weights ─
    vector<array<float,NUM_FEATS>> X; X.reserve(DS.size());
    vector<int>                    Y; Y.reserve(DS.size());
    vector<float>                  W; W.reserve(DS.size());

    double P = 0, N = 0;
    for (auto& s: DS) s.label ? ++P : ++N;
    double S  = P + N;
    double w1 = S / (2 * P), w0 = S / (2 * N);

    for (auto& s: DS) {
        X.push_back(s.feat);
        Y.push_back(s.label);
        double gap   = fabs(s.row_t - s.col_t);
        double base  = s.label ? w1 : w0;
        double wgap  = 1 + GAP_EMPH * gap / (s.row_t + s.col_t + 1e-6);
        double wcost = 1 + min(s.qcost, 1e6) / 1e5;
        W.push_back(base * wgap * wcost);
    }

    // ─ Train or load model ─
    RandomForest rf(nTrees, maxDepth, minSamples, minGain, sampleRatio);
    mt19937 rng(42);
    if (!skip_train) {
        rf.fit(X, Y, W, rng);
        json outj; outj["forest"] = rf.to_json();
        ofstream out(modelPath);
        out << outj.dump(2);
        logI("model saved → " + modelPath);
    } else {
        logI("loading model from " + modelPath);
        ifstream in(modelPath);
        json inj; in >> inj;
        rf.from_json(inj["forest"]);
    }

    // ─ Evaluate ─
    size_t TP = 0, FP = 0, TN = 0, FN = 0;
    double rt_row  = 0,
           rt_col  = 0,
           rt_rule = 0,
           rt_rf   = 0,
           rt_hyb  = 0,
           rt_fann = 0,    // ← NEW: FANN-based average
           rt_opt  = 0;
    size_t n = 0;

    for (auto& s: DS) {
        ++n;
        rt_row  += (s.row_t  - rt_row)  / n;
        rt_col  += (s.col_t  - rt_col)  / n;
        double rule = s.qcost > costThreshold ? s.col_t : s.row_t;
        rt_rule += (rule       - rt_rule) / n;
        float p   = rf.predict(s.feat.data());
        bool dec  = (fabs(p - 0.5f) < delta
                       ? (s.qcost > costThreshold)
                       : (p >= 0.5f));
        rt_rf   += ((dec ? s.col_t : s.row_t) - rt_rf)   / n;
        rt_hyb  += ((s.hyb ? s.col_t : s.row_t) - rt_hyb)  / n;
        rt_fann += ((s.fann? s.col_t : s.row_t) - rt_fann) / n;  // ← NEW
        rt_opt  += (min(s.row_t, s.col_t) - rt_opt)      / n;

        if      (s.label &&  dec) ++TP;
        else if (!s.label &&  dec) ++FP;
        else if (!s.label && !dec) ++TN;
        else if ( s.label && !dec) ++FN;
    }

    double prec = TP ? double(TP)/(TP+FP) : 0;
    double rec  = TP ? double(TP)/(TP+FN) : 0;
    double f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : 0;

    cout << "\n=== CONFUSION (RF @0.5) ===\n"
         << "TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN<<"\n"
         << "Precision="<<prec<<" Recall="<<rec<<" F1="<<f1<<"\n\n";

    cout << "=== AVG RUNTIME (s) ===\n"
         << "Row only        : "<<rt_row<<"\n"
         << "Column only     : "<<rt_col<<"\n"
         << "Cost threshold  : "<<rt_rule<<"\n"
         << "Random-Forest   : "<<rt_rf<<"\n"
         << "Hybrid optimizer: "<<rt_hyb<<"\n"
         << "FANN-based      : "<<rt_fann<<"\n"    // ← NEW
         << "Optimal oracle  : "<<rt_opt<<"\n";

    return 0;
}