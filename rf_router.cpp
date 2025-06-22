/************************************************************************************
 * rf_router.cpp
 *
 *   – A slimmed-down RandomForest router that trains on only a few key features
 *   – Improves training speed and reduces model size
 *
 * CLI flags:
 *   --data_dirs=DIR1,DIR2,...      (required)
 *   --base=/path/to/query_costs    (default /home/wuy/query_costs)
 *   --trees=12                     (number of trees)
 *   --folds=5                      (k in k-fold CV)
 *   --max_depth=11                 (max depth per tree)
 *   --min_samples=40               (min samples per leaf)
 *   --min_gain=0.0005              (min impurity decrease to split)
 *   --sample_ratio=0.6             (bootstrap ratio)
 *   --model=rf_router_model.json   (where to save/load)
 *   --skip_train                   (skip training, just load & evaluate)
 ************************************************************************************/

#include <bits/stdc++.h>
#include "json.hpp"
#include <chrono>
#include <omp.h>

using json = nlohmann::json;
using namespace std;

/* ─────── logging ─────── */
static void logI(const string&s){ cerr<<"[INFO]  "<<s<<'\n'; }
static void logW(const string&s){ cerr<<"[WARN]  "<<s<<'\n'; }
static void logE(const string&s){ cerr<<"[ERR]   "<<s<<'\n'; }
static void progress(const string& tag, size_t cur, size_t tot, size_t w=40){
  double p = tot ? double(cur)/tot : 1.0;
  size_t filled = size_t(p*w);
  cout << '\r' << tag << " [" << string(filled,'=') << string(w-filled,' ')
       << "] " << setw(3) << int(p*100) << "% ("<<cur<<"/"<<tot<<")"<<flush;
  if(cur==tot) cout<<"\n";
}



static constexpr int FEAT_DIM = 7;

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


static bool plan2feat(const json& plan, float out[FEAT_DIM]) {
  if (!plan.contains("query_block")) return false;
  const json* qb = &plan["query_block"];
  if (qb->contains("union_result")) {
    const auto& specs = (*qb)["union_result"]["query_specifications"];
    if (specs.is_array() && !specs.empty())
      qb = &specs[0]["query_block"];
  }

  Agg a;
  walk(*qb, a, 1);
  if (a.cnt == 0) return false;

  double inv    = 1.0 / a.cnt;
  double qCost  = safe_f(qb->value("cost_info", json::object()), "query_cost");

  // feature 0: avg rows_examined_per_scan (log)
  out[0] = float(log1p_clip(a.re * inv));
  // feature 1: avg read_cost (log)
  out[1] = float(log1p_clip(a.rc * inv));
  // feature 2: avg data_read_per_join (log)
  out[2] = float(log1p_clip(a.dr * inv));
  // feature 3: fraction using index
  out[3] = float(a.idxUse * inv);
  // feature 4: max selectivity
  out[4] = float(a.selMax);
  // feature 5: total query_cost (log)
  out[5] = float(log1p_clip(qCost));
  // feature 6: all-tables covered flag
  out[6] = float(a.coverCount == a.cnt ? 1.0f : 0.0f);

  return true;
}

/* ─────── simple DecisionTree ─────── */
struct DTNode { short feat, left, right; float thr, prob; };
class DecisionTree {
  vector<DTNode> nodes_;
  double min_gain_;
  int max_depth_, min_samples_;
  static double gini(double pos,double tot){
    if(tot<=0) return 1.0;
    double p=pos/tot; return 2*p*(1-p);
  }
  int build(const vector<int>& idx,
            const vector<array<float,FEAT_DIM>>& X,
            const vector<int>& y,
            const vector<float>& w,
            int depth)
  {
    double pos=0, tot=0;
    for(int i:idx){ tot+=w[i]; pos+=w[i]*y[i]; }
    DTNode node{ -1,-1,-1, 0.f, float(pos/max(1e-12,tot)) };
    // stop criteria
    if(depth>=max_depth_ || idx.size()<=min_samples_ ||
       node.prob<=0.02f || node.prob>=0.98f)
    {
      nodes_.push_back(node);
      return nodes_.size()-1;
    }
    double base_imp = gini(pos,tot),
           best_imp = base_imp;
    int best_f=-1; float best_t=0;
    // search splits
    for(int f=0;f<FEAT_DIM;++f){
      vector<float> vals;
      vals.reserve(idx.size());
      for(int i:idx) vals.push_back(X[i][f]);
      sort(vals.begin(), vals.end());
      vals.erase(unique(vals.begin(),vals.end()), vals.end());
      if(vals.size()<2) continue;
      int steps = min<int>(7, vals.size()-1);
      for(int s=1;s<=steps;++s){
        float thr = vals[ vals.size()*s/(steps+1) ];
        double lpos=0, ltot=0, rpos=0, rtot=0;
        for(int i:idx){
          if(X[i][f]<thr){ ltot+=w[i]; lpos+=w[i]*y[i]; }
          else            { rtot+=w[i]; rpos+=w[i]*y[i]; }
        }
        if(ltot<1e-6||rtot<1e-6) continue;
        double imp = gini(lpos,ltot)*(ltot/tot)
                   + gini(rpos,rtot)*(rtot/tot);
        if(base_imp-imp < min_gain_) continue;
        if(imp < best_imp){
          best_imp = imp;
          best_f   = f;
          best_t   = thr;
        }
      }
    }
    if(best_f<0){
      nodes_.push_back(node);
      return nodes_.size()-1;
    }
    // split and recurse
    node.feat = best_f;
    node.thr  = best_t;
    int me = nodes_.size();
    nodes_.push_back(node);
    vector<int> L,R;
    for(int i:idx){
      if(X[i][best_f]<best_t) L.push_back(i);
      else                    R.push_back(i);
    }
    int li = build(L,X,y,w,depth+1),
        ri = build(R,X,y,w,depth+1);
    nodes_[me].left  = li;
    nodes_[me].right = ri;
    return me;
  }
public:
  DecisionTree(int md,int ms,double mg)
    : min_gain_(mg), max_depth_(md), min_samples_(ms) {}
  void fit(const vector<array<float,FEAT_DIM>>& X,
           const vector<int>& y,
           const vector<float>& w)
  {
    nodes_.clear();
    vector<int> idx(X.size());
    iota(idx.begin(), idx.end(), 0);
    build(idx,X,y,w,0);
  }
  float predict(const float *fv) const {
    int id=0;
    while(nodes_[id].feat>=0){
      id = (fv[nodes_[id].feat] < nodes_[id].thr
            ? nodes_[id].left
            : nodes_[id].right);
    }
    return nodes_[id].prob;
  }
  json to_json() const {
    json a = json::array();
    for(auto &n: nodes_){
      a.push_back({
        {"feat", n.feat},
        {"thr",  n.thr},
        {"left", n.left},
        {"right",n.right},
        {"prob", n.prob}
      });
    }
    return a;
  }
  void from_json(const json &a){
    nodes_.clear();
    for(auto &e: a){
      DTNode n;
      n.feat  = e["feat"];
      n.thr   = e["thr"];
      n.left  = e["left"];
      n.right = e["right"];
      n.prob  = e["prob"];
      nodes_.push_back(n);
    }
  }
};





/* ─────── RandomForest on FEAT_DIM dims ─────── */
class RandomForest {
  vector<DecisionTree> trees_;
  double sampleRatio_;
public:
  RandomForest(int n,int md,int ms,double mg,double sr)
    : sampleRatio_(sr)
  {
    trees_.reserve(n);
    for(int i=0;i<n;++i) trees_.emplace_back(md,ms,mg);
  }
  void fit(const vector<array<float,FEAT_DIM>>& X,
           const vector<int>& y,
           const vector<float>& w,
           mt19937 &rng)
  {
    uniform_int_distribution<int> uni(0, X.size()-1);
    #pragma omp parallel for schedule(dynamic)
    for(int t=0; t<int(trees_.size()); ++t){
      mt19937 loc(rng()+t);
      uniform_int_distribution<int> u2(0,X.size()-1);
      size_t m = size_t(sampleRatio_ * X.size());
      vector<array<float,FEAT_DIM>> BX; BX.reserve(m);
      vector<int> By; By.reserve(m);
      vector<float> Bw; Bw.reserve(m);
      for(size_t i=0;i<m;++i){
        int j = u2(loc);
        BX.push_back(X[j]);
        By.push_back(y[j]);
        Bw.push_back(w[j]);
      }
      trees_[t].fit(BX,By,Bw);
      #pragma omp critical
      progress(" RF-train", t+1, trees_.size());
    }
  }
  float predict(const float *fv) const {
    double sum=0;
    for(auto &tr: trees_) sum += tr.predict(fv);
    return float(sum/trees_.size());
  }
  json to_json() const {
    json arr = json::array();
    for(auto &tr: trees_) arr.push_back(tr.to_json());
    return arr;
  }
  void from_json(const json &arr){
    trees_.clear();
    for(auto &a: arr){
      DecisionTree tr(0,0,0);
      tr.from_json(a);
      trees_.push_back(tr);
    }
  }
};

/* ─────── load/save model ─────── */
static const char *MODEL_JSON = "rf_router_model_simple.json";

bool save_model(const RandomForest &rf){
  json j;
  j["forest"] = rf.to_json();
  ofstream out(MODEL_JSON);
  if(!out) return false;
  out << j.dump(2);
  return true;
}
bool load_model(RandomForest &rf){
  ifstream in(MODEL_JSON);
  if(!in) return false;
  json j; in>>j;
  rf.from_json(j["forest"]);
  return true;
}

// int main(int argc, char** argv) {
//     // ─────────── CLI parsing ───────────
//     std::string base = "/home/wuy/query_costs";
//     std::string modelPath = MODEL_JSON;  // defined above
//     std::vector<std::string> dataDirs;
//     int nTrees = 12, kFolds = 5, maxDepth = 11, minSamples = 40;
//     double minGain = 0.0005, sampleRatio = 0.6;
//     bool skip_train = false;

//     for (int i = 1; i < argc; ++i) {
//         std::string a(argv[i]);
//         if      (a.rfind("--data_dirs=", 0) == 0) {
//             std::string s = a.substr(12), tmp;
//             std::stringstream ss(s);
//             while (std::getline(ss, tmp, ',')) dataDirs.push_back(tmp);
//         }
//         else if (a == "--skip_train")             skip_train = true;
//         else if (a.rfind("--trees=", 0) == 0)     nTrees     = std::stoi(a.substr(8));
//         else if (a.rfind("--folds=", 0) == 0)     kFolds     = std::stoi(a.substr(8));
//         else if (a.rfind("--max_depth=", 0) == 0) maxDepth   = std::stoi(a.substr(12));
//         else if (a.rfind("--min_samples=", 0) == 0) minSamples= std::stoi(a.substr(14));
//         else if (a.rfind("--min_gain=", 0) == 0)  minGain    = std::stod(a.substr(11));
//         else if (a.rfind("--sample_ratio=", 0)==0) sampleRatio= std::stod(a.substr(14));
//         else if (a.rfind("--model=", 0) == 0)     modelPath  = a.substr(8);
//         else if (a.rfind("--base=", 0) == 0)      base       = a.substr(7);
//     }
//     if (dataDirs.empty()) {
//         logE("need --data_dirs=...");
//         return 1;
//     }

//     // ─────────── load dataset ───────────
//     struct Sample {
//         std::array<float, FEAT_DIM> f;
//         int    label;
//         double row_t, col_t, qc;
//         bool   hyb;
//     };
//     std::vector<Sample> DS;
//     size_t skipped = 0;
//     constexpr double DEF_T = 60.0;

//     for (auto& d : dataDirs) {
//         std::string csv = base + "/" + d + "/query_costs.csv";
//         std::ifstream fin(csv);
//         if (!fin) { logW("missing CSV: " + csv); continue; }
//         std::string head; std::getline(fin, head);
//         std::vector<std::string> lines;
//         for (std::string l; std::getline(fin, l); ) lines.push_back(l);

//         size_t tot = lines.size();
//         for (size_t i = 0; i < tot; ++i) {
//             progress("load", i+1, tot);
//             auto& line = lines[i];
//             std::stringstream ss(line);
//             std::string V[5];
//             for (int k = 0; k < 5; ++k) std::getline(ss, V[k], ',');

//             if (V[0].empty()) { ++skipped; continue; }

//             double rt = 0.0, ct = 0.0;
//             bool rowOk = false, colOk = false;
//             try { rt = std::stod(V[2]); rowOk = true; } catch(...) {}
//             try { ct = std::stod(V[3]); colOk = true; } catch(...) {}
//             if (!colOk) { ++skipped; continue; }
//             if (!rowOk) rt = DEF_T;

//             // parse JSON plan → raw 44D → reduced FEAT_DIM
//             float feats[FEAT_DIM];
//             ifstream pjson(base + "/" + d + "/row_plans/" + V[0] + ".json");
//             if (!pjson) { ++skipped; continue; }
//             json plan; pjson >> plan;
//             if (!plan2feat(plan, feats)) {
//                 ++skipped;
//                 continue;
//             }
//             Sample s;
//             for (int j=0; j<FEAT_DIM;++j) {
//                 s.f[j] = feats[j];
//             }
//             s.label = (V[1] == "1");
//             s.row_t = rt; s.col_t = ct;
//             s.qc    = safe_f(plan["query_block"]
//                              .value("cost_info", json::object()), "query_cost");
//             s.hyb   = (V[4] == "1");
//             DS.push_back(s);
//         }
//     }
//     logI("loaded samples = " + std::to_string(DS.size()) +
//          ", skipped = " + std::to_string(skipped));
//     if (DS.size() < size_t(minSamples*2)) { logE("too few samples"); return 1; }

//     // ─────────── assemble ML arrays ───────────
//     std::vector<std::array<float, FEAT_DIM>> X;
//     std::vector<int> Y;
//     std::vector<float> W;
//     X.reserve(DS.size()); Y.reserve(DS.size()); W.reserve(DS.size());

//     double P=0,N=0;
//     for (auto& s : DS) (s.label ? P : N)++;
//     double Ssum = P+N, w1 = Ssum/(2*P), w0 = Ssum/(2*N);

//     for (auto& s : DS) {
//         X.push_back(s.f);
//         Y.push_back(s.label);
//         double gap  = std::fabs(s.row_t - s.col_t);
//         double base = s.label ? w1 : w0;
//         double wgap = 1 + GAP_EMPH * gap / (s.row_t + s.col_t + 1e-6);
//         double wcost= 1 + std::min(s.qc,1e6)/1e5;
//         W.push_back(float(base * wgap * wcost));
//     }

//     // ─────────── hyper-parameter CV ───────────
//     int bestD = maxDepth, bestM = minSamples;
//     if (!skip_train) {
//         logI("performing CV on reduced feature set...");
//         std::mt19937 rng(42);
//         std::vector<int> idx(DS.size());
//         std::iota(idx.begin(), idx.end(), 0);
//         std::shuffle(idx.begin(), idx.end(), rng);
//         std::vector<std::vector<int>> folds(kFolds);
//         for (size_t i = 0; i < idx.size(); ++i)
//             folds[i % kFolds].push_back(idx[i]);

//         double bestF1 = -1;
//         for (int d : { maxDepth-2, maxDepth, maxDepth+2 })
//         for (int m : { minSamples/2, minSamples, minSamples*2 }) {
//             double f1sum = 0;
//             for (int k = 0; k < kFolds; ++k) {
//                 std::vector<std::array<float,FEAT_DIM>> trX, teX;
//                 std::vector<int> trY, teY;
//                 std::vector<float> trW;
//                 for (int i = 0; i < kFolds; ++i) {
//                     for (int id : folds[i]) {
//                         if (i == k) { teX.push_back(X[id]); teY.push_back(Y[id]); }
//                         else        { trX.push_back(X[id]); trY.push_back(Y[id]); trW.push_back(W[id]); }
//                     }
//                 }
//                 RandomForest rf_cv(8, d, m, minGain, sampleRatio);
//                 rf_cv.fit(trX, trY, trW, rng);
//                 size_t TP=0,FP=0,FN=0;
//                 for (size_t i = 0; i < teX.size(); ++i) {
//                     bool p = rf_cv.predict(teX[i].data()) >= 0.5f;
//                     if      (teY[i] && p) TP++;
//                     else if (!teY[i] && p) FP++;
//                     else if (teY[i] && !p) FN++;
//                 }
//                 double prec = TP? double(TP)/(TP+FP) : 0;
//                 double rec  = TP? double(TP)/(TP+FN) : 0;
//                 double f1   = (prec+rec)? 2*prec*rec/(prec+rec) : 0;
//                 f1sum += f1;
//             }
//             double avgF1 = f1sum / kFolds;
//             if (avgF1 > bestF1) { bestF1 = avgF1; bestD = d; bestM = m; }
//         }
//         logI("CV chose depth=" + std::to_string(bestD) +
//              ", min_samples=" + std::to_string(bestM));
//     }

//     // ─────────── train or load final RF ───────────
//     RandomForest rf(nTrees, bestD, bestM, minGain, sampleRatio);
//     std::mt19937 rng2(123);
//     if (skip_train) {
//         logI("loading model from disk...");
//         if (!load_model(rf)) { logE("failed to load model"); return 1; }
//     } else {
//         logI("training final RF on " + std::to_string(DS.size()) + " samples...");
//         rf.fit(X, Y, W, rng2);
//         logI("saving model...");
//         save_model(rf);
//     }

//     // ─────────── evaluation ───────────
//     double rt_rule=0, rt_rf=0, rt_hyb=0, rt_opt=0, rt_row=0, rt_col=0;
//     size_t n=0, TP=0,FP=0,TN=0,FN=0;
//     auto cost_thresh = [&](const Sample& s){ return s.qc > 5e4 ? s.col_t : s.row_t; };

//     for (auto& s : DS) {
//         ++n;
//         // Random-Forest decision
//         float p_rf = rf.predict(s.f.data());
//         bool  dec  = (p_rf >= 0.5f);

//         // confusion
//         if      (s.label && dec)       ++TP;
//         else if (!s.label && dec)      ++FP;
//         else if (!s.label && !dec)     ++TN;
//         else if (s.label && !dec)      ++FN;

//         // incremental averages
//         rt_row += (s.row_t - rt_row) / n;
//         rt_col += (s.col_t - rt_col) / n;
//         double rt1 = cost_thresh(s);
//         rt_rule += (rt1 - rt_rule)/n;
//         double rt2 = dec ? s.col_t : s.row_t;
//         rt_rf   += (rt2 - rt_rf)/n;
//         double rt3 = s.hyb ? s.col_t : s.row_t;
//         rt_hyb  += (rt3 - rt_hyb)/n;
//         double rt4 = std::min(s.row_t, s.col_t);
//         rt_opt  += (rt4 - rt_opt)/n;
//     }

//     double prec = TP? double(TP)/(TP+FP) : 0;
//     double rec  = TP? double(TP)/(TP+FN) : 0;
//     double f1   = (prec+rec)? 2*prec*rec/(prec+rec) : 0;

//     std::cout << "\n=== CONFUSION MATRIX (RF @0.5) ===\n"
//               << "TP="<<TP<<"  FP="<<FP<<"  TN="<<TN<<"  FN="<<FN<<"\n"
//               << "Precision="<<prec<<"  Recall="<<rec<<"  F1="<<f1<<"\n\n";

//     std::cout << "=== AVG RUNTIME (seconds) ===\n"
//               << "Row only           : " << rt_row  << "\n"
//               << "Column only        : " << rt_col  << "\n"
//               << "Cost threshold     : " << rt_rule << "\n"
//               << "Random-Forest      : " << rt_rf   << "\n"
//               << "Hybrid optimizer   : " << rt_hyb  << "\n"
//               << "Optimal oracle     : " << rt_opt  << "\n";

//     return 0;
// }



int main(int argc, char** argv) {
    // ─────────── CLI parsing ───────────
    std::string base = "/home/wuy/query_costs";
    std::string modelPath = MODEL_JSON;  // defined above
    std::vector<std::string> dataDirs;
    int nTrees = 12, kFolds = 5, maxDepth = 11, minSamples = 40;
    double minGain = 0.0005, sampleRatio = 0.6;
    bool skip_train = false;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a.rfind("--data_dirs=", 0) == 0) {
            std::string s = a.substr(12), tmp;
            std::stringstream ss(s);
            while (std::getline(ss, tmp, ',')) dataDirs.push_back(tmp);
        }
        else if (a == "--skip_train")               skip_train = true;
        else if (a.rfind("--trees=", 0) == 0)       nTrees     = std::stoi(a.substr(8));
        else if (a.rfind("--folds=", 0) == 0)       kFolds     = std::stoi(a.substr(8));
        else if (a.rfind("--max_depth=", 0) == 0)   maxDepth   = std::stoi(a.substr(12));
        else if (a.rfind("--min_samples=", 0) == 0) minSamples = std::stoi(a.substr(14));
        else if (a.rfind("--min_gain=", 0) == 0)    minGain    = std::stod(a.substr(11));
        else if (a.rfind("--sample_ratio=", 0) == 0)sampleRatio= std::stod(a.substr(14));
        else if (a.rfind("--model=", 0) == 0)       modelPath  = a.substr(8);
        else if (a.rfind("--base=", 0) == 0)        base       = a.substr(7);
    }
    if (dataDirs.empty()) {
        logE("need --data_dirs=...");
        return 1;
    }

    // ─────────── load dataset ───────────
    struct Sample {
        std::array<float, FEAT_DIM> f;
        int    label;
        double row_t, col_t, qc;
        bool   hyb;
    };
    std::vector<Sample> DS;
    size_t skipped = 0;
    constexpr double DEF_T = 60.0;

    for (auto& d : dataDirs) {
        std::string csv = base + "/" + d + "/query_costs.csv";
        std::ifstream fin(csv);
        if (!fin) { logW("missing CSV: " + csv); continue; }
        std::string head; std::getline(fin, head);
        std::vector<std::string> lines;
        for (std::string l; std::getline(fin, l); ) lines.push_back(l);

        size_t tot = lines.size();
        for (size_t i = 0; i < tot; ++i) {
            progress("load", i+1, tot);
            auto& line = lines[i];
            std::stringstream ss(line);
            std::string V[5];
            for (int k = 0; k < 5; ++k) std::getline(ss, V[k], ',');

            if (V[0].empty()) { ++skipped; continue; }

            double rt = 0.0, ct = 0.0;
            bool rowOk = false, colOk = false;
            try { rt = std::stod(V[2]); rowOk = true; } catch(...) {}
            try { ct = std::stod(V[3]); colOk = true; } catch(...) {}
            if (!colOk) { ++skipped; continue; }
            if (!rowOk) rt = DEF_T;

            float feats[FEAT_DIM];
            std::ifstream pjson(base + "/" + d + "/row_plans/" + V[0] + ".json");
            if (!pjson) { ++skipped; continue; }
            json plan; pjson >> plan;
            if (!plan2feat(plan, feats)) {
                ++skipped;
                continue;
            }

            Sample s;
            for (int j = 0; j < FEAT_DIM; ++j) s.f[j] = feats[j];
            s.label = (V[1] == "1");
            s.row_t = rt; s.col_t = ct;
            s.qc    = safe_f(plan["query_block"]
                             .value("cost_info", json::object()), "query_cost");
            s.hyb   = (V[4] == "1");
            DS.push_back(s);
        }
    }
    logI("loaded samples = " + std::to_string(DS.size()) +
         ", skipped = " + std::to_string(skipped));
    if (DS.size() < size_t(minSamples*2)) { logE("too few samples"); return 1; }

    // ─────────── assemble ML arrays ───────────
    std::vector<std::array<float, FEAT_DIM>> X;
    std::vector<int> Y;
    std::vector<float> W;
    X.reserve(DS.size()); Y.reserve(DS.size()); W.reserve(DS.size());

    double P=0, N=0;
    for (auto& s : DS) (s.label ? P : N)++;
    double Ssum = P+N, w1 = Ssum/(2*P), w0 = Ssum/(2*N);

    for (auto& s : DS) {
        X.push_back(s.f);
        Y.push_back(s.label);
        double gap  = std::fabs(s.row_t - s.col_t);
        double base = s.label ? w1 : w0;
        double wgap = 1 + GAP_EMPH * gap / (s.row_t + s.col_t + 1e-6);
        double wcost= 1 + std::min(s.qc,1e6)/1e5;
        W.push_back(float(base * wgap * wcost));
    }

    // ─────────── no CV — use provided maxDepth/minSamples ───────────
    int bestD = maxDepth, bestM = minSamples;

    // ─────────── train or load final RF ───────────
    RandomForest rf(nTrees, bestD, bestM, minGain, sampleRatio);
    std::mt19937 rng2(123);
    if (skip_train) {
        logI("loading model from disk...");
        if (!load_model(rf)) { logE("failed to load model"); return 1; }
    } else {
        logI("training final RF on " + std::to_string(DS.size()) + " samples...");
        rf.fit(X, Y, W, rng2);
        logI("saving model...");
        save_model(rf);
    }

    // ─────────── evaluation ───────────
    double rt_rule=0, rt_rf=0, rt_hyb=0, rt_opt=0, rt_row=0, rt_col=0;
    size_t n=0, TP=0, FP=0, TN=0, FN=0;
    auto cost_thresh = [&](const Sample& s){ return s.qc > 5e4 ? s.col_t : s.row_t; };

    for (auto& s : DS) {
        ++n;
        float p_rf = rf.predict(s.f.data());
        bool dec = (p_rf >= 0.5f);

        if      (s.label && dec)  ++TP;
        else if (!s.label && dec) ++FP;
        else if (!s.label && !dec)++TN;
        else if (s.label && !dec) ++FN;

        rt_row  += (s.row_t - rt_row) / n;
        rt_col  += (s.col_t - rt_col) / n;
        double rt1 = cost_thresh(s);
        rt_rule += (rt1 - rt_rule) / n;
        double rt2 = dec ? s.col_t : s.row_t;
        rt_rf   += (rt2 - rt_rf) / n;
        double rt3 = s.hyb ? s.col_t : s.row_t;
        rt_hyb  += (rt3 - rt_hyb) / n;
        double rt4 = std::min(s.row_t, s.col_t);
        rt_opt  += (rt4 - rt_opt) / n;
    }

    double prec = TP ? double(TP)/(TP+FP) : 0;
    double rec  = TP ? double(TP)/(TP+FN) : 0;
    double f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : 0;

    std::cout << "\n=== CONFUSION MATRIX (RF @0.5) ===\n"
              << "TP="<<TP<<"  FP="<<FP<<"  TN="<<TN<<"  FN="<<FN<<"\n"
              << "Precision="<<prec<<"  Recall="<<rec<<"  F1="<<f1<<"\n\n";

    std::cout << "=== AVG RUNTIME (seconds) ===\n"
              << "Row only           : " << rt_row  << "\n"
              << "Column only        : " << rt_col  << "\n"
              << "Cost threshold     : " << rt_rule << "\n"
              << "Random-Forest      : " << rt_rf   << "\n"
              << "Hybrid optimizer   : " << rt_hyb  << "\n"
              << "Optimal oracle     : " << rt_opt  << "\n";

    return 0;
}
