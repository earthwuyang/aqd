/************************************************************************************
 * train_dtree_enhanced.cpp
 *
 *  ▸ Bootstrap-based Random-Forest (default 32 trees)
 *  ▸ Regularised CART (max-depth, min-samples, min-gain)
 *  ▸ Dynamic root split (no hard-coded 50 K guard)
 *  ▸ k-fold (k=5) cross-validation to pick hyper-params
 *  ▸ Tiny 3-feature logistic-regression stacked on top
 *
 *  CLI flags (all optional):
 *    --data_dirs=DIR1,DIR2,...      • required – training folders
 *    --base=/path/to/query_costs    • default /home/wuy/query_costs
 *    --trees=32                     • forest size
 *    --folds=5                      • k in k-fold CV
 *    --max_depth=15                 • CART regularisation
 *    --min_samples=40
 *    --min_gain=0.0005
 *    --model=rf_model.json
 *
 ************************************************************************************/

#include <bits/stdc++.h>
#include "json.hpp"
#include <chrono>                 // ⬅ for timing
#include <omp.h>

using json = nlohmann::json;
using namespace std;

/* ─────────── logging ─────────── */
static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }

/* ───────── additional progress helper ───────── */
static void progress(const std::string& tag,
                     std::size_t cur, std::size_t tot,
                     std::size_t w = 40)
{
    double f = tot ? double(cur) / tot : 1.0;
    std::size_t filled = std::size_t(f * w);
    std::cout << '\r' << tag << " ["
              << std::string(filled, '=')
              << std::string(w - filled, ' ')
              << "] " << std::setw(3) << int(f * 100) << "% ("
              << cur << '/' << tot << ')' << std::flush;
    if (cur == tot) std::cout << '\n';
}

/* ─────────── feature constants ─────────── */
constexpr int NUM_FEATS  = 44;        // ← 43 old + 1 all-covered flag
constexpr double GAP_EMPH=2.0;

/* ─────────── tolerant helpers ─────────── */
static double safe_f(const json&v){
    if(v.is_number())  return v.get<double>();
    if(v.is_string())  { try{ return stod(v.get<string>());}catch(...){ } }
    return 0.0;
}
static double safe_f(const json&obj,const char*key){
    if(!obj.contains(key)) return 0.0;
    return safe_f(obj[key]);
}
static double log1p_clip(double v){ return log1p(max(0.0,v)); }
static double str_size_to_num(string s){
    if(s.empty()) return 0.0;
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    double m=1; char suf=s.back();
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return stod(s)*m; }catch(...){ return 0.0; }
}

/* ─────────── per-plan DFS aggregation ─────────── */
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0;
    double selSum=0,selMin=1e30,selMax=0,ratioSum=0,ratioMax=0;
    int cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,maxDepth=0;
    double fanoutMax=0; bool grp=0,ord=0,tmp=0;
    int coverCount=0,sumPossibleKeys=0; double maxPrefix=0,minRead=1e30;
} ;
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

static void walk(const json&n,Agg&a,int depth)
{
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
            a.re+=re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr; a.cnt++;

            if(t.contains("possible_keys")&&t["possible_keys"].is_array())
                a.sumPossibleKeys+=int(t["possible_keys"].size());
            a.maxPrefix=max(a.maxPrefix,pc);
            a.minRead=min(a.minRead,rc<1e30?rc:1e30);

            if(re>0){
                double sel=rp/re;
                a.selSum+=sel; a.selMin=min(a.selMin,sel); a.selMax=max(a.selMax,sel);
                a.fanoutMax=max(a.fanoutMax,sel);
            }
            double ratio=(ec>0)?rc/ec:rc;
            a.ratioSum+=ratio; a.ratioMax=max(a.ratioMax,ratio);

            string at=t.value("access_type","ALL");
            if(at=="range")a.cRange++; else if(at=="ref")a.cRef++;
            else if(at=="eq_ref")a.cEq++; else if(at=="index")a.cIdx++; else a.cFull++;
            if(getBool(t,"using_index")) a.idxUse++;

            /* covering-index check */
            if(t.contains("used_columns")&&t["used_columns"].is_array()
               &&t.contains("possible_keys")&&t["possible_keys"].is_array()){
                unordered_set<string> pk;
                for(auto&x:t["possible_keys"]) if(x.is_string()) pk.insert(x.get<string>());
                bool full=true;
                for(auto&u:t["used_columns"])
                    if(!u.is_string()||!pk.count(u.get<string>())){full=false;break;}
                if(full) a.coverCount++;
            }
        }

        if(n.contains("grouping_operation")) a.grp=true;
        if(n.contains("ordering_operation")||getBool(n,"using_filesort")) a.ord=true;
        if(getBool(n,"using_temporary_table")) a.tmp=true;

        for(const auto&kv:n.items())
            if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }
    else if(n.is_array()) for(const auto&el:n) walk(el,a,depth);
    a.maxDepth=max(a.maxDepth,depth);
}

/* ─────────── JSON→feature vector (44D) ─────────── */
static bool plan2feat(const json&plan,float f[NUM_FEATS])
{
    if(!plan.contains("query_block")) return false;
    const json*qb=&plan["query_block"];
    if(qb->contains("union_result")){
        const auto&specs=(*qb)["union_result"]["query_specifications"];
        if(specs.is_array()&&!specs.empty()) qb=&specs[0]["query_block"];
    }
    Agg a; walk(*qb,a,1); if(a.cnt==0) return false;
    double inv=1.0/a.cnt;
    int k=0;
    double qCost=safe_f(qb->value("cost_info",json::object()),"query_cost");
    double rootRow=safe_f(*qb,"rows_produced_per_join");

    /* original 0-41 */
    f[k++]=log1p_clip(a.re*inv);
    f[k++]=log1p_clip(a.rp*inv);
    f[k++]=log1p_clip(a.f*inv);
    f[k++]=log1p_clip(a.rc*inv);
    f[k++]=log1p_clip(a.ec*inv);
    f[k++]=log1p_clip(a.pc*inv);
    f[k++]=log1p_clip(a.dr*inv);

    f[k++]=a.cRange*inv; f[k++]=a.cRef*inv; f[k++]=a.cEq*inv;
    f[k++]=a.cIdx*inv;  f[k++]=a.cFull*inv; f[k++]=a.idxUse*inv;

    f[k++]=a.selSum*inv; f[k++]=a.selMin; f[k++]=a.selMax;
    f[k++]=a.maxDepth;  f[k++]=a.fanoutMax;

    f[k++]=a.grp?1:0; f[k++]=a.ord?1:0; f[k++]=a.tmp?1:0;
    f[k++]=a.ratioSum*inv; f[k++]=a.ratioMax;
    f[k++]=log1p_clip(qCost); f[k++]=log1p_clip(rootRow);

    f[k++]=log1p_clip((a.pc*inv)/max(1e-6,a.rc*inv));
    f[k++]=log1p_clip((a.rc*inv)/max(1e-6,a.re*inv));
    f[k++]=log1p_clip((a.ec*inv)/max(1e-6,a.re*inv));

    f[k++]=(a.cnt==1); f[k++]=(a.cnt>1);
    f[k++]=log1p_clip(a.maxDepth*(a.idxUse*inv));
    double fullFrac=a.cFull? (a.cFull*inv):1e-3;
    f[k++]=log1p_clip((a.idxUse*inv)/fullFrac);

    f[k++]=a.cnt;
    f[k++]=a.cnt? float(a.sumPossibleKeys)/a.cnt : 0.f;
    f[k++]=log1p_clip(a.maxPrefix);
    f[k++]=log1p_clip(a.minRead<1e30?a.minRead:0.0);
    f[k++]=(a.cnt>1?float(a.cnt-1)/a.cnt:0.f);
    f[k++]=(rootRow>0?float(a.re)/rootRow:0.f);
    f[k++]=float(a.selMax-a.selMin);
    {
        float denom=float(a.cRange+a.cRef+a.cEq+a.cIdx);
        f[k++]=denom?float(a.idxUse)/denom:0.f;
    }
    f[k++]=float(qCost);                         // 40 raw
    f[k++]=(qCost>5e4?1.f:0.f);                  // 41
    f[k++]=a.cnt?float(a.coverCount)/a.cnt:0.f;  // 42
    /* NEW 43: all-tables covered flag */
    f[k++] = (a.coverCount==a.cnt ? 1.f : 0.f);  // 43
    return true;
}

/* ─────────── CART decision tree ─────────── */

struct DTNode {
    int feat = -1;
    int left = -1;
    int right = -1;
    float thr = 0;
    float prob = 0;
};

class DecisionTree {
    std::vector<DTNode> nodes_;
    double min_gain_;
    int max_depth_, min_samples_;

    // ─── Entropy impurity helper ───
    static inline double entropy(double pos, double tot) {
        if (tot <= 0) return 0.0;
        double p = pos / tot;
        double q = 1.0 - p;
        double h = 0.0;
        if (p > 0) h -= p * std::log2(p);
        if (q > 0) h -= q * std::log2(q);
        return h;
    }

    // ─── Recursive builder ───
    int build(const std::vector<int>& idx,
              const std::vector<std::array<float,NUM_FEATS>>& X,
              const std::vector<int>& y,
              const std::vector<float>& w,
              int depth)
    {
        // compute weighted positive / total
        double tot_w = 0.0, pos_w = 0.0;
        for (int i : idx) {
            tot_w += w[i];
            pos_w += w[i] * y[i];
        }
        DTNode node;
        node.prob = float(pos_w / std::max(1e-12, tot_w));

        // stopping criteria
        if (depth >= max_depth_
         || idx.size() <= size_t(min_samples_)
         || node.prob <= 0.02f
         || node.prob >= 0.98f)
        {
            nodes_.push_back(node);
            return int(nodes_.size()) - 1;
        }

        double base_ent = entropy(pos_w, tot_w);
        double best_gain = 0.0;
        int    best_f    = -1;
        float  best_thr  = 0.0f;

        // search for best split
        for (int f = 0; f < NUM_FEATS; ++f) {
            // collect unique feature values
            std::vector<float> vals;
            vals.reserve(idx.size());
            for (int i : idx) vals.push_back(X[i][f]);
            std::sort(vals.begin(), vals.end());
            vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
            if (vals.size() < 2) continue;

            int steps = std::min<int>(9, vals.size() - 1);
            for (int s = 1; s <= steps; ++s) {
                float thr = vals[ vals.size() * s / (steps + 1) ];
                double l_tot = 0.0, l_pos = 0.0;
                double r_tot = 0.0, r_pos = 0.0;
                int    l_cnt = 0,   r_cnt = 0;
                for (int i : idx) {
                    if (X[i][f] < thr) {
                        l_tot += w[i];
                        l_pos += w[i] * y[i];
                        ++l_cnt;
                    } else {
                        r_tot += w[i];
                        r_pos += w[i] * y[i];
                        ++r_cnt;
                    }
                }
                if (l_cnt < min_samples_ || r_cnt < min_samples_) continue;

                double entL = entropy(l_pos, l_tot);
                double entR = entropy(r_pos, r_tot);
                double weighted_ent = (l_tot/tot_w)*entL + (r_tot/tot_w)*entR;
                double gain = base_ent - weighted_ent;

                if (gain > best_gain && gain > min_gain_) {
                    best_gain = gain;
                    best_f    = f;
                    best_thr  = thr;
                }
            }
        }

        // if no valid split, make leaf
        if (best_f < 0) {
            nodes_.push_back(node);
            return int(nodes_.size()) - 1;
        }

        // otherwise split and recurse
        node.feat = best_f;
        node.thr  = best_thr;
        int self  = int(nodes_.size());
        nodes_.push_back(node);  // placeholder

        std::vector<int> left_idx, right_idx;
        for (int i : idx) {
            (X[i][best_f] < best_thr ? left_idx : right_idx).push_back(i);
        }

        int l = build(left_idx,  X, y, w, depth + 1);
        int r = build(right_idx, X, y, w, depth + 1);
        nodes_[self].left  = l;
        nodes_[self].right = r;
        return self;
    }

public:
    DecisionTree(int md, int ms, double mg)
      : min_gain_(mg), max_depth_(md), min_samples_(ms) {}

    void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
             const std::vector<int>& y,
             const std::vector<float>& w)
    {
        nodes_.clear();
        std::vector<int> idx(X.size());
        std::iota(idx.begin(), idx.end(), 0);
        build(idx, X, y, w, 0);
    }

    float predict(const float *f) const {
        int id = 0;
        while (nodes_[id].feat != -1) {
            id = (f[nodes_[id].feat] < nodes_[id].thr
                  ? nodes_[id].left
                  : nodes_[id].right);
        }
        return nodes_[id].prob;
    }

    json to_json() const {
        json arr = json::array();
        for (const auto &n : nodes_) {
            arr.push_back({
                {"feat",  n.feat},
                {"thr",   n.thr},
                {"left",  n.left},
                {"right", n.right},
                {"prob",  n.prob}
            });
        }
        return arr;
    }

    void from_json(const json &arr) {
        nodes_.clear();
        for (const auto &e : arr) {
            DTNode n;
            n.feat  = e.at("feat").get<int>();
            n.thr   = e.at("thr").get<float>();
            n.left  = e.at("left").get<int>();
            n.right = e.at("right").get<int>();
            n.prob  = e.at("prob").get<float>();
            nodes_.push_back(n);
        }
    }
};


/* ─────────── Random-Forest ─────────── */
class RandomForest {
  std::vector<DecisionTree> trees_;
  double sampleRatio_;
public:
  RandomForest(int n,int md,int ms,double mg,double sr)
    : sampleRatio_(sr)
  {
    trees_.reserve(n);
    for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
  }

  void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
           const std::vector<int>&  y,
           const std::vector<float>& w,
           std::mt19937& rng)
  {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    std::uniform_int_distribution<int> uni(0, int(X.size() - 1));
    std::size_t T = trees_.size();
    #pragma omp parallel for schedule(dynamic)
    for(int t = 0; t < int(T); ++t)
    {
      std::mt19937 rng_local(rng()+t);
      std::uniform_int_distribution<int> uni_local(0, int(X.size()-1));

      std::size_t m = std::size_t(sampleRatio_ * X.size());
      std::vector<std::array<float,NUM_FEATS>> BX; BX.reserve(m);
      std::vector<int>  By;  By.reserve(m);
      std::vector<float>Bw;  Bw.reserve(m);

      for(std::size_t i=0;i<m;++i){
        int j = uni_local(rng_local);
        BX.push_back(X[j]);
        By.push_back(y[j]);
        Bw.push_back(w[j]);
      }

      trees_[t].fit(BX,By,Bw);
      #pragma omp critical
      progress("RF-train", t+1, T);
    }

    auto t1 = clock::now();
    logI("Random-Forest training took " +
         std::to_string(std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()) +
         " s");
  }

  float predict(const float*f) const {
    double s=0;
    for(const auto& tr: trees_) s += tr.predict(f);
    return float(s/trees_.size());
  }

  json to_json() const {
    json arr = json::array();
    for(const auto& tr: trees_) arr.push_back(tr.to_json());
    return arr;
  }

  // ─────── NEW from_json ───────
  void from_json(const json& arr) {
    trees_.clear();
    for(const auto& jt : arr) {
      DecisionTree tr(0,0,0);
      tr.from_json(jt);
      trees_.push_back(std::move(tr));
    }
  }
};


/* ─────────── sample struct ─────────── */
struct Sample{
    array<float,NUM_FEATS> feat{};
    int label=0; double row_t=0,col_t=0,qcost=0; bool hyb=0;
};

/* ─────────── CSV / plan loader ─────────── */
static bool load_plan(const string&path,array<float,NUM_FEATS>&f,double&qcost){
    ifstream in(path); if(!in) return false;
    json j; try{ in>>j; }catch(...){ return false; }
    if(!plan2feat(j,f.data())) return false;
    qcost=0; if(j.contains("query_block"))
        qcost=safe_f(j["query_block"].value("cost_info",json::object()),"query_cost");
    return true;
}



int main(int argc, char* argv[]) {
    // CLI defaults
    string baseDir = "/home/wuy/query_costs";
    vector<string> dataDirs;
    int nTrees = 32;
    int kFolds = 5;             // unused now
    int maxDepth = 15;
    int minSamples = 40;
    double minGain = 0.0005;
    double sampleRatio = 0.6;
    bool skip_train = false;
    string modelPath = "rf_entropy.json";

    // Parse flags
    for(int i=1;i<argc;++i){
        string a(argv[i]);
        if(a.rfind("--data_dirs=",0)==0){
            string s=a.substr(12),tmp;
            stringstream ss(s);
            while(getline(ss,tmp,',')) dataDirs.push_back(tmp);
        } else if(a.rfind("--base=",0)==0)     baseDir = a.substr(7);
        else if(a.rfind("--trees=",0)==0)     nTrees = stoi(a.substr(8));
        else if(a.rfind("--max_depth=",0)==0) maxDepth = stoi(a.substr(12));
        else if(a.rfind("--min_samples=",0)==0)minSamples= stoi(a.substr(14));
        else if(a.rfind("--min_gain=",0)==0)  minGain = stod(a.substr(11));
        else if(a.rfind("--sample_ratio=",0)==0)sampleRatio = stod(a.substr(14));
        else if(a.rfind("--model=",0)==0)     modelPath = a.substr(8);
        else if(a=="--skip_train")            skip_train = true;
    }

    if(dataDirs.empty()){
        logE("need --data_dirs=...");
        return 1;
    }
    if(sampleRatio<=0 || sampleRatio>1){ logE("sample_ratio must be in (0,1]"); return 1; }

    // Load samples
    struct Sample { array<float,NUM_FEATS> feat; int label; double row_t,col_t,qcost; bool hyb; };
    vector<Sample> DS;
    size_t skipped = 0;

    for(const auto& d: dataDirs){
        ifstream fin(baseDir+"/"+d+"/query_costs.csv");
        if(!fin){ logW("missing CSV: "+d); continue; }
        string line;
        getline(fin,line); // header
        vector<string> lines;
        while(getline(fin,line)) lines.push_back(line);

        size_t cur=0, tot=lines.size();
        for(auto& l: lines){
            progress("load", ++cur, tot);
            stringstream ss(l);
            string qid, lab, rt, ct, hyb;
            getline(ss,qid,','); getline(ss,lab,',');
            getline(ss,rt,',');  getline(ss,ct,',');
            getline(ss,hyb,',');
            if(qid.empty()){ skipped++; continue; }
            Sample s;
            try{ s.row_t = stod(rt); } catch(...){ s.row_t = 60.0; }
            try{ s.col_t = stod(ct); } catch(...){ skipped++; continue; }
            s.label = (lab=="1");
            s.hyb = (hyb=="1");
            string planPath = baseDir + "/" + d + "/row_plans/" + qid + ".json";
            double qc;
            if(!load_plan(planPath, s.feat, qc)){ skipped++; continue; }
            s.qcost = qc;
            DS.push_back(s);
        }
    }
    logI("loaded " + to_string(DS.size()) + " samples, skipped=" + to_string(skipped));
    if(DS.size() < size_t(minSamples*2)){ logE("too few samples"); return 1; }

    // Build X, Y, W
    vector<array<float,NUM_FEATS>> X;
    vector<int> Y;
    vector<float> W;
    X.reserve(DS.size()); Y.reserve(DS.size()); W.reserve(DS.size());

    double P=0,N=0;
    for(auto& s: DS) s.label? ++P : ++N;
    double S = P+N;
    double w1 = S/(2*P), w0 = S/(2*N);
    for(auto& s: DS){
        X.push_back(s.feat);
        Y.push_back(s.label);
        double gap = fabs(s.row_t - s.col_t);
        double base = s.label? w1 : w0;
        double wgap = 1 + GAP_EMPH * gap / (s.row_t + s.col_t + 1e-6);
        double wcost = 1 + min(s.qcost,1e6)/1e5;
        W.push_back(base * wgap * wcost);
    }

    // Train or load
    RandomForest rf(nTrees, maxDepth, minSamples, minGain, sampleRatio);
    mt19937 rng(42);
    if(!skip_train){
        rf.fit(X,Y,W,rng);
        json outj;
        outj["forest"] = rf.to_json();
        ofstream out(modelPath);
        out << outj.dump(2);
        logI("model saved → " + modelPath);
    } else {
        logI("loading model from " + modelPath);
        ifstream in(modelPath);
        json inj; in>>inj;
        rf.from_json(inj["forest"]);
    }

    // Evaluation
    size_t TP=0,FP=0,TN=0,FN=0;
    double rt_row=0, rt_col=0, rt_rule=0, rt_rf=0, rt_hyb=0, rt_opt=0;
    size_t n=0;
    for(auto& s: DS){
        ++n;
        rt_row += (s.row_t - rt_row)/n;
        rt_col += (s.col_t - rt_col)/n;
        double rule = s.qcost>5e4? s.col_t : s.row_t;
        rt_rule += (rule - rt_rule)/n;
        float p = rf.predict(s.feat.data());
        bool dec = p >= 0.5;
        rt_rf += ((dec? s.col_t : s.row_t) - rt_rf)/n;
        rt_hyb += ((s.hyb? s.col_t : s.row_t) - rt_hyb)/n;
        rt_opt += (min(s.row_t,s.col_t) - rt_opt)/n;
        if(s.label && dec) ++TP;
        else if(!s.label && dec) ++FP;
        else if(!s.label && !dec) ++TN;
        else if(s.label && !dec) ++FN;
    }
    double prec = TP? double(TP)/(TP+FP):0;
    double rec  = TP? double(TP)/(TP+FN):0;
    double f1   = (prec+rec)? 2*prec*rec/(prec+rec):0;

    cout << "\n=== CONFUSION (RF @0.5) ===\n"
         << "TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN<<"\n"
         << "Precision="<<prec<<" Recall="<<rec<<" F1="<<f1<<"\n\n";
    cout << "=== AVG RUNTIME (s) ===\n"
         << "Row only        : "<<rt_row<<"\n"
         << "Column only     : "<<rt_col<<"\n"
         << "Cost threshold  : "<<rt_rule<<"\n"
         << "Random-Forest   : "<<rt_rf<<"\n"
         << "Hybrid optimizer: "<<rt_hyb<<"\n"
         << "Optimal oracle  : "<<rt_opt<<"\n";

    return 0;
}
