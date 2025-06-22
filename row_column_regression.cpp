/************************************************************************************
 *
 * Row/Column Time Dual‑Regression Router  (v2, log‑space, ReLU, safer training)
 *
 * ▸ Row regressor : 63‑d → hidden → 1  (MLP, ReLU, L2‑WD, log1p(time) target)
 * ▸ Col regressor : GIN encoder + linear head          (log1p(time) target)
 * ▸ Early stopping: monitor (valRowMSE + valColMSE)
 * ▸ Thread safety : single‑thread train loop (remove OpenMP on weight updates)
 * ▸ Runtime choose: pick engine with lower   expm1(predicted_log_time)
 *
 * Compile:
 *   g++ -std=c++17 -O2 -I/path/to/eigen3 -I/path/to/json/single_include \
 *       -o row_column_regression_v2 row_column_regression_v2.cpp
 *
 ************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <set>

#include <Eigen/Dense>
#include "json.hpp"

using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

/*────────────────────────────── utilities ──────────────────────────────*/
static void logInfo (const std::string &m){ std::cout  << "[INFO]  " << m << '\n'; }
static void logWarn (const std::string &m){ std::cerr  << "[WARN]  " << m << '\n'; }
static void logError(const std::string &m){ std::cerr  << "[ERROR] " << m << '\n'; }

inline double safe_q_error(double pred, double truth){
    const double eps = 1e-6;
    if(pred < eps || truth < eps) return std::numeric_limits<double>::infinity();
    double q = pred / truth;  return (q < 1.0) ? 1.0 / q : q;
}
inline double dhuber(double err, double delta=1.0){
    double a = std::abs(err);
    return (a <= delta) ? err : delta * ((err > 0)? 1.0 : -1.0);
}
bool fileExists(const std::string &p){ std::ifstream f(p); return f.good(); }

/*─────────────────────────── your feature/plan extraction … (保持不变) ──────────────*/
/*     为节省篇幅，此处省略 plan2feat / parseColPlan 等大段逻辑。                  */
/*     请直接把原文件同名函数/结构体粘进这里——它们无需改动，与本文档兼容。        */
/*──────────────────────────────────────────────────────────────────────────────────*/

/*───────────────────────────── progress ─────────────────────────────*/
void showProgressBar(size_t cur,size_t tot,size_t w=50){
    double frac = tot? double(cur)/tot : 1.0;
    size_t filled = size_t(frac*w);
    std::cout << "\r[";
    for(size_t i=0;i<filled;++i) std::cout << '=';
    for(size_t i=filled;i<w;++i) std::cout << ' ';
    std::cout << "] " << int(frac*100) << "% ("<<cur<<"/"<<tot<<")";
    if(cur>=tot) std::cout<<'\n';
    std::fflush(stdout);
}

/*──────────────────────── utilities ──────────────────────────*/
double parseNum(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string()) return std::stod(j[k].get<std::string>());
        if(j[k].is_number()) return j[k].get<double>();
    } catch(...) { /*ignore*/ }
    return 0.0;
}
double safeLog1p(double v){ return std::log1p(std::max(0.0,v)); }

/*────────────────── random‐forest’s plan2feat ──────────────────*/
static const int NUM_FEATS = 63;
static std::unordered_map<std::string,std::unordered_set<std::string>> indexCols;

static double strSizeToNum(std::string s){
    if(s.empty()) return 0;
    char suf=s.back(); double m=1;
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return std::stod(s)*m; }catch(...){ return 0; }
}
double convert_data_size_to_numeric(std::string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    char suff=s.back(); double f=1.0;
    if(suff=='G'){f=1e9; s.pop_back();}
    else if(suff=='M'){f=1e6; s.pop_back();}
    else if(suff=='K'){f=1e3; s.pop_back();}
    try{ return std::stod(s)*f; } catch(...){ return 0.0;}
}
double parsePossibleNumber(const json &j,const std::string &k){
    if(!j.contains(k)) return 0.0;
    try{
        if(j[k].is_string()) return std::stod(j[k].get<std::string>());
        if(j[k].is_number()) return j[k].get<double>();
    } catch(...){
        logWarn("Invalid number for key: "+k);
    }
    return 0.0;
}

/* ──────────────── parseRowPlanJSON (8‐D) ───────────────── */
bool parseRowPlanJSON(const std::string &path,float out[8],double &query_cost){
    std::fill(out,out+8,0.f);
    std::ifstream ifs(path); if(!ifs){ logWarn("Missing plan "+path); return false; }
    json j; try{ ifs>>j;}catch(...){ logWarn("JSON error "+path); return false; }
    if(!j.contains("query_block")){ logWarn("No query_block "+path); return false;}
    struct FC{ double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0; int n=0;} fc;
    std::function<void(const json&)> rec=[&](const json& b){
        if(b.is_object()){
            if(b.contains("table")){
                const auto&t=b["table"];
                double re=parsePossibleNumber(t,"rows_examined_per_scan");
                double rp=parsePossibleNumber(t,"rows_produced_per_join");
                double f =parsePossibleNumber(t,"filtered");
                double rc=0,ec=0,pc=0,dr=0;
                if(t.contains("cost_info")){
                    const auto&ci=t["cost_info"];
                    rc=parsePossibleNumber(ci,"read_cost");
                    ec=parsePossibleNumber(ci,"eval_cost");
                    pc=parsePossibleNumber(ci,"prefix_cost");
                    dr=convert_data_size_to_numeric(ci.value("data_read_per_join","0"));
                }
                if(re>=0&&rp>=0){
                    fc.re+=re; fc.rp+=rp; fc.f+=f;
                    fc.rc+=rc; fc.ec+=ec; fc.pc+=pc; fc.dr+=dr;
                    ++fc.n;
                }
            }
            for(const auto&kv:b.items()) rec(kv.value());
        } else if(b.is_array()){
            for(const auto&v:b) rec(v);
        }
    };
    rec(j["query_block"]);
    query_cost=0.0;
    if(j["query_block"].contains("cost_info"))
        query_cost=parsePossibleNumber(j["query_block"]["cost_info"],"query_cost");
    if(fc.n==0) return true;
    auto avg=[&](double x){ return safeLog1p(x/fc.n); };
    out[0]=float(avg(fc.re)); out[1]=float(avg(fc.rp)); out[2]=float(avg(fc.f));
    out[3]=float(avg(fc.rc)); out[4]=float(avg(fc.ec)); out[5]=float(avg(fc.pc));
    out[6]=float(avg(fc.dr)); out[7]=float(safeLog1p(query_cost));
    return true;
}

static bool getBool(const json&o,const char*k){
    if(!o.contains(k)) return false;
    auto &v=o[k];
    if(v.is_boolean()) return v.get<bool>();
    if(v.is_string()){
        std::string s=v.get<std::string>(); std::transform(s.begin(),s.end(),s.begin(),::tolower);
        return s=="true"||s=="yes"||s=="1";
    }
    return false;
}

struct Agg {
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0;
    int cnt=0, cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0;
    double selSum=0,selMin=1e30,selMax=0,fanoutMax=0;
    double ratioSum=0,ratioMax=0;
    bool grp=false, ord=false, tmp=false;
    int coverCount=0;
    int maxDepth=0;
};

static void walk(const json&n, Agg&a, int depth){
    if(n.is_object()){
        if(n.contains("table")){
            auto &t = n["table"];
            double re = parseNum(t,"rows_examined_per_scan");
            double rp = parseNum(t,"rows_produced_per_join");
            double fl = parseNum(t,"filtered");
            double rc=0,ec=0, pc=0, dr=0;
            if(t.contains("cost_info")){
                auto &ci = t["cost_info"];
                rc = parseNum(ci,"read_cost");
                ec = parseNum(ci,"eval_cost");
                pc = parseNum(ci,"prefix_cost");
                if(ci.contains("data_read_per_join")){
                    if(ci["data_read_per_join"].is_string())
                        dr = strSizeToNum(ci["data_read_per_join"].get<std::string>());
                    else dr = parseNum(ci,"data_read_per_join");
                }
            }
            a.re += re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr;
            a.cnt++;
            if(re>0){ double sel=rp/re; a.selSum+=sel; a.selMin=fmin(a.selMin,sel); a.selMax=fmax(a.selMax,sel); a.fanoutMax=fmax(a.fanoutMax,sel);}
            double ratio = ec>0?rc/ec:rc;
            a.ratioSum += ratio; a.ratioMax=fmax(a.ratioMax,ratio);
            std::string at = t.value("access_type","ALL");
            if      (at=="range")  a.cRange++;
            else if (at=="ref")    a.cRef++;
            else if (at=="eq_ref") a.cEq++;
            else if (at=="index")  a.cIdx++;
            else                   a.cFull++;
            if(getBool(t,"using_index")) a.idxUse++;
            if(t.contains("used_columns")&&t.contains("key")){
                std::string idx = t["key"].get<std::string>();
                auto it = indexCols.find(idx);
                if(it!=indexCols.end()){
                    bool full=true;
                    for(auto &u:t["used_columns"])
                        if(!it->second.count(u.get<std::string>())){ full=false; break;}
                    if(full) a.coverCount++;
                }
            }
        }
        if(n.contains("grouping_operation"))             a.grp=true;
        if(n.contains("ordering_operation")||getBool(n,"using_filesort")) a.ord=true;
        if(getBool(n,"using_temporary_table"))           a.tmp=true;
        for(auto&kv:n.items()) if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }
    else if(n.is_array()){
        for(auto &v:n) walk(v,a,depth);
    }
    a.maxDepth = std::max(a.maxDepth, depth);
}

static bool plan2feat(const json&plan, float f[NUM_FEATS]){
    if(!plan.contains("query_block")) return false;
    const json* qb = &plan["query_block"];
    if(qb->contains("union_result")){
        auto &specs = (*qb)["union_result"]["query_specifications"];
        if(specs.is_array()&&!specs.empty())
            qb = &specs[0]["query_block"];
    }
    Agg a{};
    walk(*qb,a,1);
    if(a.cnt==0) return false;
    double inv = 1.0 / a.cnt;
    int k=0;
    double qCost  = parseNum(qb->value("cost_info", json::object()),"query_cost");
    double rootRow= parseNum(*qb,"rows_produced_per_join");
    // 0–6
    f[k++] = safeLog1p(a.re*inv);
    f[k++] = safeLog1p(a.rp*inv);
    f[k++] = safeLog1p(a.f *inv);
    f[k++] = safeLog1p(a.rc*inv);
    f[k++] = safeLog1p(a.ec*inv);
    f[k++] = safeLog1p(a.pc*inv);
    f[k++] = safeLog1p(a.dr*inv);
    // 7–12
    f[k++] = a.cRange*inv; f[k++] = a.cRef*inv; f[k++] = a.cEq*inv;
    f[k++] = a.cIdx*inv;   f[k++] = a.cFull*inv;f[k++] = a.idxUse*inv;
    // 13–17
    f[k++] = a.selSum*inv; f[k++] = a.selMin;  f[k++] = a.selMax;
    f[k++] = a.maxDepth;   f[k++] = a.fanoutMax;
    // 18–20
    f[k++] = a.grp?1.f:0.f; f[k++] = a.ord?1.f:0.f; f[k++] = a.tmp?1.f:0.f;
    // 21–22
    f[k++] = a.ratioSum*inv; f[k++] = a.ratioMax;
    // 23–24
    f[k++] = safeLog1p(qCost);
    f[k++] = safeLog1p(rootRow);
    // 25–27
    f[k++] = safeLog1p((a.pc*inv)/std::max(1e-6, a.rc*inv));
    f[k++] = safeLog1p((a.rc*inv)/std::max(1e-6, a.re*inv));
    f[k++] = safeLog1p((a.ec*inv)/std::max(1e-6, a.re*inv));
    // 28–29
    f[k++] = (a.cnt==1)?1.f:0.f;
    f[k++] = (a.cnt>1)?1.f:0.f;
    // 30–31
    f[k++] = safeLog1p(a.maxDepth*(a.idxUse*inv));
    { double fullFrac = a.cFull? (a.cFull*inv) : 1e-3;
      f[k++] = safeLog1p((a.idxUse*inv)/fullFrac); }
    // 32–39
    f[k++] = float(a.cnt);
    f[k++] = a.cnt? float(a.selSum)/a.cnt : 0.f;  // reused slot
    f[k++] = safeLog1p(a.pc*inv);
    f[k++] = safeLog1p(a.rc*inv);
    f[k++] = (a.cnt>1)? float(a.cnt-1)/a.cnt : 0.f;
    f[k++] = rootRow>0? float(a.re)/rootRow : 0.f;
    f[k++] = a.selMax - a.selMin;
    { float denom = float(a.cRange+a.cRef+a.cEq+a.cIdx);
      f[k++] = denom? float(a.idxUse)/denom : 0.f; }
    // 40–41
    f[k++] = float(qCost);
    f[k++] = (qCost>5e4)?1.f:0.f;
    // 42–43
    f[k++] = a.cnt? float(a.coverCount)/a.cnt : 0.f;
    f[k++] = (a.coverCount==a.cnt)?1.f:0.f;
    // 44
    f[k++] = safeLog1p(a.re*inv) - safeLog1p(a.selSum*inv);
    // 45–46
    f[k++] = float(a.cnt);
    f[k++] = safeLog1p(a.cnt);
    // 47–48
    f[k++] = float(a.cnt);           // reused
    f[k++] = a.cnt? float(a.coverCount)/a.cnt : 0.f;
    // 49–50
    f[k++] = float(a.cnt);           // reused
    f[k++] = a.cnt? float(a.coverCount)/a.cnt : 0.f;
    // 51–56
    f[k++] = a.idxUse*inv;
    f[k++] = a.cRange*inv;
    f[k++] = a.cRef*inv;
    f[k++] = a.cEq*inv;
    f[k++] = a.cIdx*inv;
    f[k++] = a.cFull*inv;
    // 57–59
    f[k++] = safeLog1p(a.pc*inv);
    f[k++] = safeLog1p(a.rc*inv);
    f[k++] = a.selMax - a.selMin;
    // 60–62
    f[k++] = a.ratioMax;
    f[k++] = a.fanoutMax;
    f[k++] = a.selMin>0? (a.selMax/a.selMin) : 0.f;
    assert(k==NUM_FEATS);
    return true;
}


/*─────────────────── column stats & parseColPlan ───────────────────*/

/* ──────────────────── column planning stats ──────────────────── */
struct ColStats {
    double rows_c=0, rows_s=1, cost_c=0, cost_s=1;
    std::unordered_map<std::string,int> op2id;
};
static ColStats buildColStats(const std::string& dir) {
    ColStats st;
    std::string cache = dir+"/column_plan_statistics.json";
    if(fileExists(cache)){
        try{
            json j; std::ifstream(cache)>>j;
            if(j.contains("op2id") && j["op2id"].size()>0){
                st.rows_c=j["rows_c"];
                st.rows_s=j["rows_s"];
                st.cost_c=j["cost_c"];
                st.cost_s=j["cost_s"];
                for(auto&kv:j["op2id"].items())
                    st.op2id[kv.key()]=kv.value();
                return st;
            }
        }catch(...){
            logWarn("Bad stats cache, rebuilding: "+cache);
        }
    }
    logInfo("Scanning column plans in "+dir);
    std::vector<std::string> files;
    std::string cdir=dir+"/column_plans";
    DIR* dp = opendir(cdir.c_str());
    if(!dp){ logWarn("No directory "+cdir); return st; }
    struct dirent* de;
    while((de=readdir(dp))){
        std::string fn=de->d_name;
        if(fn.size()>5 && fn.substr(fn.size()-5)==".json")
            files.push_back(fn);
    }
    closedir(dp);
    size_t total=files.size();
    logInfo("Found "+std::to_string(total)+" column plan files");

    std::vector<double> rows,cost;
    std::set<std::string> ops;
    for(size_t i=0;i<total;++i){
        showProgressBar(i+1,total);
        std::ifstream f(cdir+"/"+files[i]);
        if(!f) continue;
        json j; try{ f>>j; }catch(...){ continue; }
        if(!j.contains("plan")||!j["plan"].is_array()) continue;
        std::vector<json> stack(j["plan"].begin(),j["plan"].end());
        while(!stack.empty()){
            json node=stack.back(); stack.pop_back();
            std::string op=node.value("operator","UNK");
            ops.insert(op);
            rows.push_back(safeLog1p(parsePossibleNumber(node,"esti_rows")));
            cost.push_back(safeLog1p(parsePossibleNumber(node,"esti_cost")));
            if(node.contains("children"))
                for(auto&c:node["children"]) stack.push_back(c);
        }
    }
    if(!rows.empty()){
        auto mid=[&](std::vector<double>&v){
            std::nth_element(v.begin(),v.begin()+v.size()/2,v.end());
            return v[v.size()/2];
        };
        auto iqr=[&](std::vector<double>&v){
            size_t n=v.size();
            std::nth_element(v.begin(),v.begin()+n/4,v.end());
            double q1=v[n/4];
            std::nth_element(v.begin()+n/4,v.begin()+3*n/4,v.end());
            double q3=v[3*n/4];
            return (q3-q1)/2.0;
        };
        st.rows_c=mid(rows);
        st.cost_c=mid(cost);
        st.rows_s=std::max(1e-6, iqr(rows));
        st.cost_s=std::max(1e-6, iqr(cost));
        int id=0;
        for(const auto&op:ops) st.op2id[op]=id++;
    }
    json dump{
        {"rows_c",st.rows_c},
        {"rows_s",st.rows_s},
        {"cost_c",st.cost_c},
        {"cost_s",st.cost_s},
        {"op2id",st.op2id}
    };
    std::ofstream(cache)<<dump.dump(2);
    return st;
}

/* ──────────────────── parseColPlan → Graph ──────────────────── */
struct Graph{
    int N=0;
    std::vector<Eigen::Vector2d> x;
    std::vector<std::vector<int>> adj;
    std::vector<int> op;
};
static Graph parseColPlan(const json&j,
                          const ColStats&st,
                          const std::unordered_map<std::string,int>&globalOp2id)
{
    Graph g;
    if(!j.contains("plan")||!j["plan"].is_array()) return g;
    std::vector<const json*> stack{&j["plan"][0]};
    std::vector<std::pair<int,int>> edges;
    while(!stack.empty()){
        auto cur=stack.back(); stack.pop_back();
        int idx=g.N++;
        double rawR=safeLog1p(parsePossibleNumber(*cur,"esti_rows"));
        double rawC=safeLog1p(parsePossibleNumber(*cur,"esti_cost"));
        double nr=(rawR-st.rows_c)/st.rows_s;
        double nc=(rawC-st.cost_c)/st.cost_s;
        g.x.emplace_back(nr,nc);
        std::string opName=cur->value("operator","UNK");
        int gid=globalOp2id.count(opName)?globalOp2id.at(opName):globalOp2id.at("UNK");
        g.op.push_back(gid);
        g.adj.emplace_back();
        if(cur->contains("children")&&cur->at("children").is_array()){
            for(auto&c:cur->at("children")){
                stack.push_back(&c);
                edges.emplace_back(idx,g.N);
                edges.emplace_back(g.N,idx);
            }
        }
    }
    for(auto&e:edges)
        if(e.first<g.N&&e.second<g.N)
            g.adj[e.first].push_back(e.second);
    return g;
}

/*────────────────── dataset loader (with 63‐dim row feats) ──────────────────*/
struct Sample {
    float  feats[NUM_FEATS];
    Graph  colGraph;
    int    label;        // 0=row,1=col,-1=skip
    double row_time, col_time, query_cost;
    int    hybrid_use_imci, fann_use_imci;
};


class ExecutionPlanDataset{
    std::vector<Sample> data_;
public:
    ExecutionPlanDataset(const std::vector<std::string>& dirs,double timeout){
        std::vector<ColStats> statsList;
        std::unordered_set<std::string> allOps;
        for(auto&d:dirs){
            std::string dir="/home/wuy/query_costs/"+d;
            auto cst=buildColStats(dir);
            statsList.push_back(cst);
            for(auto&kv:cst.op2id) allOps.insert(kv.first);
        }
        allOps.insert("UNK");
        std::unordered_map<std::string,int> globalOp2id;
        int oid=0; for(auto&op:allOps) globalOp2id[op]=oid++;
        size_t skip=0,val=0;
        for(size_t di=0;di<dirs.size();++di){
            auto&d=dirs[di];
            std::string dir="/home/wuy/query_costs/"+d;
            std::string csv=dir+"/query_costs.csv";
            if(!fileExists(csv)){ logWarn("Missing "+csv); continue; }
            size_t lines=0;
            { std::ifstream f(csv); std::string l; while(std::getline(f,l)) ++lines; }
            std::ifstream in(csv);
            std::string header; std::getline(in,header);
            size_t cur=0,tot=(lines?lines-1:0);
            const auto&cst=statsList[di];
            std::string rpdir=dir+"/row_plans", cpdir=dir+"/column_plans", line;
            while(std::getline(in,line)){
                ++cur; showProgressBar(cur,tot);
                if(line.empty()) continue;
                std::stringstream ss(line);
                std::string qid,lbl,rt,ct,hy,fy;
                std::getline(ss,qid,','); std::getline(ss,lbl,',');
                std::getline(ss,rt,',');  std::getline(ss,ct,',');
                std::getline(ss,hy,',');  std::getline(ss,fy,',');
                if(qid.empty()||lbl.empty()){ ++skip; continue;}
                Sample s{};
                s.label=std::stoi(lbl);
                s.row_time = rt.empty()?timeout:std::stod(rt);
                s.col_time = ct.empty()?timeout:std::stod(ct);
                if(std::fabs(s.row_time-s.col_time)<1e-3) s.label=-1;
                s.hybrid_use_imci = hy.empty()?0:std::stoi(hy);
                s.fann_use_imci   = fy.empty()?0:std::stoi(fy);
                if(!parseRowPlanJSON(rpdir+"/"+qid+".json",s.feats,s.query_cost)){
                    ++skip; continue;
                }
                if(fileExists(cpdir+"/"+qid+".json")){
                    std::ifstream cf(cpdir+"/"+qid+".json"); json cj; cf>>cj;
                    s.colGraph=parseColPlan(cj,cst,globalOp2id);
                }
                data_.push_back(s); ++val;
            }
        }
        logInfo("Loaded "+std::to_string(val)+" samples, skipped="+std::to_string(skip));
    }
    size_t size() const{ return data_.size(); }
    const Sample& operator[](size_t i) const{ return data_[i]; }

    auto begin()       -> decltype(data_.begin()) { return data_.begin(); }
    auto end()         -> decltype(data_.end())   { return data_.end();   }
    auto begin() const -> decltype(data_.cbegin()){ return data_.cbegin();}
    auto end()   const -> decltype(data_.cend())  { return data_.cend();  }
};

/*─────────────────────── GIN & MLPReg as defined above ─────────────────────*/

/* ──────────────────────── GIN encoder ──────────────────────── */
struct GIN {
    int in_dim, hidden, n_ops;

    MatrixXd W1, W2;
    VectorXd b1, b2;
    MatrixXd opEmb;

    RowVectorXd w_col;
    double      b_col;

    std::mt19937                      rng;
    std::uniform_real_distribution<> dist;  // uniform [-0.1,0.1]

    GIN(int n_ops_, int hidden_)
    : in_dim(2 + 8),
        hidden(hidden_),
        n_ops(n_ops_),
        W1(hidden, in_dim),
        b1(hidden),
        W2(hidden, hidden),
        b2(hidden),
        opEmb(8, n_ops_),
        w_col(hidden),
        b_col(0.0),
        rng(123),
        dist(-0.1, 0.1)
    {
        // Initialize W1
        for (int i = 0; i < W1.rows(); ++i)
            for (int j = 0; j < W1.cols(); ++j)
                W1(i, j) = dist(rng);

        // Initialize b1
        for (int i = 0; i < b1.size(); ++i)
            b1(i) = dist(rng);

        // Initialize W2
        for (int i = 0; i < W2.rows(); ++i)
            for (int j = 0; j < W2.cols(); ++j)
                W2(i, j) = dist(rng);

        // Initialize b2
        for (int i = 0; i < b2.size(); ++i)
            b2(i) = dist(rng);

        // Initialize opEmb
        for (int i = 0; i < opEmb.rows(); ++i)
            for (int j = 0; j < opEmb.cols(); ++j)
                opEmb(i, j) = dist(rng);

        // Initialize w_col
        for (int i = 0; i < w_col.size(); ++i)
            w_col(i) = dist(rng);

        // b_col remains zero
    }

    static VectorXd relu(const VectorXd&v){ return v.cwiseMax(0.0); }

    VectorXd encode(const Graph&g) const {
        if(g.N==0) return VectorXd::Zero(hidden);
        std::vector<VectorXd> z1(g.N), h(g.N);
        VectorXd out=VectorXd::Zero(hidden);
        for(int i=0;i<g.N;++i){
            VectorXd xin(in_dim);
            xin.head(2)=g.x[i];
            xin.tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*xin + b1.col(0));
        }
        for(int i=0;i<g.N;++i){
            VectorXd agg=VectorXd::Zero(hidden);
            for(int nb:g.adj[i]) agg+=z1[nb];
            h[i]=relu(W2*agg + b2.col(0));
            out+=h[i];
        }
        return out/double(g.N);
    }

    void backward_and_update(const Graph&g,
                             const VectorXd&d_out,
                             double lr)
    {
        if(g.N==0) return;
        // recompute
        std::vector<VectorXd> xin(g.N), z1(g.N), agg(g.N), pre2(g.N), h(g.N);
        for(int i=0;i<g.N;++i){
            xin[i].resize(in_dim);
            xin[i].head(2)=g.x[i];
            xin[i].tail(8)=opEmb.col(g.op[i]);
            z1[i]=relu(W1*xin[i]+b1.col(0));
        }
        for(int i=0;i<g.N;++i){
            agg[i]=VectorXd::Zero(hidden);
            for(int nb:g.adj[i]) agg[i]+=z1[nb];
            pre2[i]=W2*agg[i]+b2.col(0);
            h[i]=relu(pre2[i]);
        }
        // grads
        MatrixXd gW2=MatrixXd::Zero(W2.rows(),W2.cols()),
                 gb2=MatrixXd::Zero(b2.rows(),b2.cols()),
                 gW1=MatrixXd::Zero(W1.rows(),W1.cols()),
                 gb1=MatrixXd::Zero(b1.rows(),b1.cols()),
                 gOp=MatrixXd::Zero(opEmb.rows(),opEmb.cols());
        VectorXd dh=d_out/double(g.N);
        for(int i=0;i<g.N;++i){
            VectorXd mask2=(pre2[i].array()>0.0).cast<double>().matrix();
            VectorXd d_pre2=dh.cwiseProduct(mask2);
            gW2+=d_pre2*agg[i].transpose();
            gb2.col(0)+=d_pre2;
            VectorXd d_agg=W2.transpose()*d_pre2;
            for(int nb:g.adj[i]){
                VectorXd mask1=((W1*xin[nb]+b1.col(0)).array()>0.0).cast<double>().matrix();
                VectorXd d_pre1=d_agg.cwiseProduct(mask1);
                gW1+=d_pre1*xin[nb].transpose();
                gb1.col(0)+=d_pre1;
                VectorXd dxin=W1.transpose()*d_pre1;
                gOp.col(g.op[nb])+=dxin.tail(8);
            }
        }
        // sgd
        W2   -=lr*gW2;  b2   -=lr*gb2;
        W1   -=lr*gW1;  b1   -=lr*gb1;
        opEmb-=lr*gOp;
    }

    // --- column‐time head forward/back/backprop ---
    double predict_col_time(const Graph&g) const {
        VectorXd h=encode(g);
        return (w_col*h)(0)+b_col;
    }
    void backward_col(double d_out,const Graph&g,double lr){
        VectorXd h=encode(g);
        RowVectorXd gw=d_out*h.transpose();
        double gb=d_out;
        w_col-=lr*gw;
        b_col-=lr*gb;
        // propagate into GIN body
        VectorXd grad_h = d_out*w_col.transpose();
        backward_and_update(g,grad_h,lr);
    }

    json to_json() const {
    return json{
      {"in_dim",      in_dim},
      {"hidden",      hidden},
      {"n_ops",       n_ops},
      {"W1",          std::vector<double>(W1.data(),   W1.data()   + W1.size())},
      {"b1",          std::vector<double>(b1.data(),   b1.data()   + b1.size())},
      {"W2",          std::vector<double>(W2.data(),   W2.data()   + W2.size())},
      {"b2",          std::vector<double>(b2.data(),   b2.data()   + b2.size())},
      {"opEmb",       std::vector<double>(opEmb.data(),opEmb.data()+opEmb.size())},
      {"w_col",       std::vector<double>(w_col.data(),w_col.data()+w_col.size())},
      {"b_col",       b_col}
    };
  }

  // deserialize
  void from_json(const json& j) {
    // 1) resize opEmb to the trained size
    int new_n_ops = j.at("n_ops").get<int>();
    if (new_n_ops != n_ops) {
      n_ops = new_n_ops;
      opEmb.resize(8, n_ops);
    }

    // 2) copy all the weight arrays
    auto vecW1 = j.at("W1").get<std::vector<double>>();
    std::copy(vecW1.begin(), vecW1.end(), W1.data());

    auto vecb1 = j.at("b1").get<std::vector<double>>();
    std::copy(vecb1.begin(), vecb1.end(), b1.data());

    auto vecW2 = j.at("W2").get<std::vector<double>>();
    std::copy(vecW2.begin(), vecW2.end(), W2.data());

    auto vecb2 = j.at("b2").get<std::vector<double>>();
    std::copy(vecb2.begin(), vecb2.end(), b2.data());

    auto vecOp = j.at("opEmb").get<std::vector<double>>();
    std::copy(vecOp.begin(), vecOp.end(), opEmb.data());

    auto vecwc = j.at("w_col").get<std::vector<double>>();
    std::copy(vecwc.begin(), vecwc.end(), w_col.data());

    b_col = j.at("b_col").get<double>();
  }
};


/*──────────────────────── MLPReg (ReLU, log‑time, L2) ───────────────────────*/
class MLPReg{
public:
    MLPReg(int in,int hid,double lr,double wd=1e-5)
      : in_(in),hid_(hid),lr_(lr),wd_(wd),
        W1(hid,in),b1(hid),
        W2(hid),b2(0.0){
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> d(-0.1,0.1);
        for(int r=0;r<W1.rows();++r)for(int c=0;c<W1.cols();++c) W1(r,c)=d(gen);
        for(int i=0;i<b1.size();++i) b1(i)=d(gen);
        for(int i=0;i<W2.size();++i) W2(i)=d(gen);
    }
    /* forward returns log‑time prediction */
    double run(const std::vector<float>& xvec) const{
        Eigen::VectorXd x(in_);
        for(int i=0;i<in_;++i) x(i)=xvec[i];
        Eigen::VectorXd a1 = (W1*x + b1).unaryExpr([](double z){return z>0?z:0;});
        return W2.dot(a1) + b2;
    }
    /* SGD on one sample, y = log1p(time) */
    void train_sample(const std::vector<float>& xvec,double y){
        Eigen::VectorXd x(in_);
        for(int i=0;i<in_;++i) x(i)=xvec[i];

        Eigen::VectorXd z1 = W1*x + b1;
        Eigen::VectorXd a1 = z1.unaryExpr([](double z){return z>0?z:0;});
        double yhat = W2.dot(a1) + b2;
        double err  = yhat - y;           // MSE grad = 2*err
        double g2   = 2.0*err;

        // grad W2 / b2
        RowVectorXd gradW2 = g2 * a1.transpose();
        double      gradb2 = g2;

        // backprop through ReLU
        Eigen::VectorXd d1 = (W2.transpose()*g2);
        for(int i=0;i<d1.size();++i) d1(i) = (z1(i)>0?1.0:0.0) * d1(i);

        Eigen::MatrixXd gradW1 = d1 * x.transpose();
        Eigen::VectorXd gradb1 = d1;

        // update with L2 regularisation
        W2 -= lr_*(gradW2 + wd_*W2);
        b2 -= lr_*gradb2;
        W1 -= lr_*(gradW1 + wd_*W1);
        b1 -= lr_*gradb1;
    }
    json to_json() const{
        return {
          {"in",in_},{"hid",hid_},
          {"W1",std::vector<double>(W1.data(),W1.data()+W1.size())},
          {"b1",std::vector<double>(b1.data(),b1.data()+b1.size())},
          {"W2",std::vector<double>(W2.data(),W2.data()+W2.size())},
          {"b2",b2}
        };
    }
    void from_json(const json&j){
        auto w1=j.at("W1").get<std::vector<double>>();
        std::copy(w1.begin(),w1.end(),W1.data());
        auto bb1=j.at("b1").get<std::vector<double>>();
        std::copy(bb1.begin(),bb1.end(),b1.data());
        auto w2=j.at("W2").get<std::vector<double>>();
        std::copy(w2.begin(),w2.end(),W2.data());
        b2=j.at("b2").get<double>();
    }
private:
    int in_,hid_; double lr_,wd_;
    Eigen::MatrixXd  W1;  Eigen::VectorXd b1;
    Eigen::RowVectorXd W2; double b2;
};

/*──────────────────────────── GIN (仅列头改 log 目标，主体同旧版) ─────────────*/
/*   以下代码与原版几乎一致，只把训练误差改为 log 空间，并删除 OpenMP 并发。      */
/*   由于篇幅限制，若无 GNN 经验可直接沿用旧实现，唯一注意：                     */
/*      err_c = pred_log - std::log1p(col_time)                                 */
/*      gin.backward_col(dhuber(err_c), ...)                                    */
/*─────────────────────────────────────────────────────────────────────────────*/

/*──────────────── dataset split ─────────────────*/
std::pair<std::vector<size_t>,std::vector<size_t>>
splitDataset(size_t n,double ratio=0.8){
    std::vector<size_t> idx(n); std::iota(idx.begin(),idx.end(),0);
    std::shuffle(idx.begin(),idx.end(),std::mt19937(std::random_device{}()));
    size_t k = size_t(ratio * n);
    return {{idx.begin(),idx.begin()+k},{idx.begin()+k,idx.end()}};
}



/*──────────────────────────── updated main() ───────────────────────────*/
int main(int argc, char* argv[])
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    /* ---------- CLI defaults ---------- */
    std::vector<std::string> dataDirs;
    int    epochs        = 300;
    int    hiddenNeurons = 128;
    double lr            = 0.001;
    double costThreshold = 50000.0;    // baseline
    double margin        = 0.10;       // 10 % safety margin
    bool   skipTrain     = false;

    const std::string rowPath = "row_mlp_model.json";
    const std::string colPath = "col_gin_model.json";

    /* ---------- parse CLI ---------- */
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a.rfind("--data_dirs=",0)==0){
            std::stringstream ss(a.substr(12)); std::string d;
            while(std::getline(ss,d,',')) dataDirs.push_back(d);
        }
        else if (a.rfind("--epochs=",0)==0)          epochs        = std::stoi(a.substr(9));
        else if (a.rfind("--hidden_neurons=",0)==0)  hiddenNeurons = std::stoi(a.substr(17));
        else if (a.rfind("--lr=",0)==0)              lr            = std::stod(a.substr(5));
        else if (a.rfind("--cost_threshold=",0)==0)  costThreshold = std::stod(a.substr(17));
        else if (a.rfind("--margin=",0)==0)          margin        = std::stod(a.substr(9));
        else if (a == "--skip_train")                skipTrain     = true;
    }
    if (dataDirs.empty()) { logError("need --data_dirs=..."); return 1; }

    /* ---------- dataset ---------- */
    ExecutionPlanDataset ds(dataDirs, 60.0);
    if (ds.size() < 30) { logError("Too few samples"); return 1; }

    /* ---------- feature norm ---------- */
    std::vector<double> mu(NUM_FEATS,0), sigma(NUM_FEATS,0);
    for (const auto& s : ds)
        for (int j=0;j<NUM_FEATS;++j) mu[j]+=s.feats[j];
    for (double& m : mu) m/=double(ds.size());
    for (const auto& s : ds)
        for (int j=0;j<NUM_FEATS;++j){
            double d=s.feats[j]-mu[j]; sigma[j]+=d*d;
        }
    for (double& v : sigma){ v=std::sqrt(v/ds.size()); if(v<1e-6) v=1.0; }

    auto norm63=[&](const float* f){
        std::vector<float> x(NUM_FEATS);
        for(int j=0;j<NUM_FEATS;++j) x[j]=float((f[j]-mu[j])/sigma[j]);
        return x;
    };

    /* ---------- split train/val ---------- */
    auto [trainIdx, valIdx] = splitDataset(ds.size(),0.8);

    /* ---------- build models ---------- */
    int maxOp=0; for(const auto& s:ds) for(int op:s.colGraph.op) maxOp=std::max(maxOp,op);
    MLPReg rowReg(NUM_FEATS,hiddenNeurons,lr,1e-5);
    GIN    gin(maxOp+1,hiddenNeurons);
    const double lr_col = lr*0.5;

    auto dhuber=[](double e,double d=1.0){
        double a=std::abs(e); return (a<=d)?e: d*((e>0)?1:-1);
    };

    /* ---------- training ---------- */
    if(!skipTrain){
        double best=1e30; int wait=0, patience=15;
        for(int ep=1;ep<=epochs;++ep){
            std::shuffle(trainIdx.begin(),trainIdx.end(),std::mt19937(std::random_device{}()));
            double trR=0,trC=0; size_t done=0,tot=trainIdx.size();
            for(auto id:trainIdx){
                const auto& s=ds[id];
                auto x=norm63(s.feats);

                /* row */
                double yr=std::log1p(s.row_time);
                double pr=rowReg.run(x);
                rowReg.train_sample(x,yr);
                trR+=(pr-yr)*(pr-yr);

                /* col */
                double yc=std::log1p(s.col_time);
                double pc=gin.predict_col_time(s.colGraph);
                double err=pc-yc;
                trC+=err*err;
                gin.backward_col(dhuber(err),s.colGraph,lr_col);

                showProgressBar(++done,tot,30);
            }
            std::cout<<'\n'; trR/=trainIdx.size(); trC/=trainIdx.size();

            double vR=0,vC=0;
            for(auto id:valIdx){
                const auto& s=ds[id];
                auto x=norm63(s.feats);
                vR+=std::pow(rowReg.run(x)-std::log1p(s.row_time),2);
                vC+=std::pow(gin.predict_col_time(s.colGraph)-std::log1p(s.col_time),2);
            }
            vR/=valIdx.size(); vC/=valIdx.size(); double vSum=vR+vC;

            logInfo("Ep "+std::to_string(ep)+" trR="+std::to_string(trR)+
                    " trC="+std::to_string(trC)+" vR="+std::to_string(vR)+
                    " vC="+std::to_string(vC)+" vSum="+std::to_string(vSum));

            if(vSum<best){ best=vSum; wait=0;
                std::ofstream(rowPath)<<rowReg.to_json().dump(2);
                std::ofstream(colPath)<<gin.to_json().dump(2);
            }else if(++wait>=patience){ logInfo("Early stop"); break; }
        }
    }else{
        std::ifstream fr(rowPath),fc(colPath);
        if(!fr||!fc){ logError("skip_train but model missing"); return 1;}
        json jr,jc; fr>>jr; fc>>jc; rowReg.from_json(jr); gin.from_json(jc);
        logInfo("Loaded pretrained models.");
    }

    /* ---------- 1-step calibration on validation ---------- */
    auto fitAffine=[&](bool isRow){
        double sx=0, sy=0, sxx=0, sxy=0; size_t n=0;
        for(auto id:valIdx){
            const auto& s=ds[id];
            double y=std::log1p(isRow? s.row_time : s.col_time);
            double x = isRow ? rowReg.run(norm63(s.feats))
                             : gin.predict_col_time(s.colGraph);
            sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; ++n;
        }
        double denom = n*sxx - sx*sx;
        double a = (denom==0)?1:(n*sxy - sx*sy)/denom;
        double b = (sy - a*sx)/n;
        return std::pair<double,double>{a,b};
    };
    auto [ar,br] = fitAffine(true);
    auto [ac,bc] = fitAffine(false);
    logInfo("Calibration  row: a="+std::to_string(ar)+" b="+std::to_string(br));
    logInfo("Calibration  col: a="+std::to_string(ac)+" b="+std::to_string(bc));

    /* ---------- evaluation ---------- */
    double r_row=0,r_col=0,r_ct=0,r_hy=0,r_fann=0,r_reg=0;
    size_t n_row=0,n_col=0,n_ct=0,n_hy=0,n_f=0,n_rg=0;
    std::vector<double> qRow,qCol;

    for(const auto& s:ds){
        auto x=norm63(s.feats);
        double pr=std::expm1(ar*rowReg.run(x)+br);
        double pc=std::expm1(ac*gin.predict_col_time(s.colGraph)+bc);

        /* baselines */
        r_row=(r_row*n_row + s.row_time)/(n_row+1); ++n_row;
        r_col=(r_col*n_col + s.col_time)/(n_col+1); ++n_col;

        double chooseCT = (s.query_cost>costThreshold)?s.col_time:s.row_time;
        r_ct=(r_ct*n_ct+chooseCT)/(n_ct+1); ++n_ct;

        double chooseH = s.hybrid_use_imci? s.col_time:s.row_time;
        r_hy=(r_hy*n_hy+chooseH)/(n_hy+1); ++n_hy;

        double chooseF = s.fann_use_imci? s.col_time:s.row_time;
        r_fann=(r_fann*n_f+chooseF)/(n_f+1); ++n_f;

        /* our router with margin */
        bool pickCol = pc < (1.0-margin)*pr;
        double chooseR = pickCol ? s.col_time : s.row_time;
        r_reg=(r_reg*n_rg+chooseR)/(n_rg+1); ++n_rg;

        /* q-error */
        qRow.push_back(safe_q_error(pr,s.row_time));
        qCol.push_back(safe_q_error(pc,s.col_time));
    }

    auto pct=[&](std::vector<double>&v,double p){
        std::sort(v.begin(),v.end()); return v[size_t(p*(v.size()-1))];
    };
    std::cout<<"\n=== Q-ERROR ROW === median "<<pct(qRow,0.5)<<"  p95 "<<pct(qRow,0.95)<<'\n';
    std::cout<<"=== Q-ERROR COL === median "<<pct(qCol,0.5)<<"  p95 "<<pct(qCol,0.95)<<"\n\n";

    std::cout<<"=== AVG RUNTIME (s/query) ===\n";
    std::cout<<" • Pure Row Store     : "<<r_row <<'\n';
    std::cout<<" • Pure Column Store  : "<<r_col <<'\n';
    std::cout<<" • Cost-Threshold     : "<<r_ct  <<'\n';
    std::cout<<" • Hybrid Optimizer   : "<<r_hy  <<'\n';
    std::cout<<" • FANN Classifier    : "<<r_fann<<'\n';
    std::cout<<" • Row-Col Regressor  : "<<r_reg <<'\n';

    auto wall=std::chrono::duration_cast<std::chrono::seconds>(Clock::now()-t0).count();
    logInfo("Total wall-clock "+std::to_string(wall)+" s");
    return 0;
}
/*─────────────────────────────────────────────────────────────────────────*/
