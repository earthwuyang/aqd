#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>

#include <fann.h>
#include "json.hpp"
using json = nlohmann::json;


/* ───────────── small helpers ───────────── */
static void logInfo (const std::string& s){ std::cout  << "[INFO]  "<<s<<'\n'; }
static void logWarn (const std::string& s){ std::cerr  << "[WARN]  "<<s<<'\n'; }
static void logError(const std::string& s){ std::cerr  << "[ERROR] "<<s<<'\n'; }
static bool fileExists(const std::string& p){ std::ifstream f(p); return f.good(); }
double safeLog1p(double v){ return std::log1p(std::max(0.0,v)); }

/* progress bar */
void showProgressBar(size_t cur,size_t tot,size_t w=50){
    double f = tot? double(cur)/tot : 1.0;  int filled=int(f*w);
    std::cout<<"\r["<<std::string(filled,'=')<<std::string(w-filled,' ')
             <<"] "<<int(f*100)<<"% ("<<cur<<'/'<<tot<<')';
    if(cur>=tot) std::cout<<'\n';  std::fflush(stdout);
}

/* parse helpers */
double parseNum(const json& j,const std::string& k){
    if(!j.contains(k)) return 0.0;
    try{ if(j[k].is_string()) return std::stod(j[k].get<std::string>());
         if(j[k].is_number()) return j[k].get<double>(); }
    catch(...){}
    return 0.0;
}
double bytesFrom(const std::string& s){
    if(s.empty()) return 0.0;  std::string t=s; while(t.size()&&isspace(t.back())) t.pop_back();
    double f=1;  switch(t.back()){ case 'G':case 'g':f=1e9; t.pop_back(); break;
                                    case 'M':case 'm':f=1e6; t.pop_back(); break;
                                    case 'K':case 'k':f=1e3; t.pop_back(); break; }
    try{ return std::stod(t)*f; }catch(...){ return 0.0; }
}

/* ───────────── scaling config ───────────── */
constexpr int NUM_FEATS = 32;
static double fMin[NUM_FEATS]{}, fMax[NUM_FEATS]{};
static bool   scaleReady=false;

static const char* KEY[NUM_FEATS] = {
/* 0-6  */ "rows_examined_per_scan","rows_produced_per_join","filtered",
           "read_cost","eval_cost","prefix_cost","data_read_per_join",
/* 7-11 */ nullptr,nullptr,nullptr,nullptr,nullptr,
/* 12   */ nullptr,                 // using-idx
/* 13-15*/ "filtered","filtered","filtered",
/* 16-17*/ nullptr,nullptr,         // depth, fanout
/* 18-20*/ nullptr,nullptr,nullptr, // flags grp/ord/tmp
/* 21-22*/ "read_cost","read_cost",
/* 23   */ "query_cost",
/* 24   */ "rows_produced_per_join", // root cardinality
/* 25-27*/ nullptr,nullptr,nullptr, // hasConst, hasLimit, hasDistinct
/* 28-29*/ nullptr,nullptr,         // hasUnion, numUnion
/* 30-31*/ nullptr,nullptr          // cross1, cross2
};
static const double DEF_MAX[NUM_FEATS] = {
/* 0-6 */0,0,0,0,0,0,0,
/*7-11*/1,1,1,1,1,  /* histogram */
 1,                /* using-idx */
 5,0,5,            /* sel stats */
 10,10,            /* depth/fanout */
 1,1,1,            /* grp/ord/tmp */
 10,10,            /* read/eval ratio */
 30,               /* qcost */
 20,               /* root rows log space max */
 1,1,1,            /* const, limit, distinct */
 1,10,             /* union flag, numUnion */
 10,10             /* cross features (rough caps) */
};

template<typename T>
inline const T& clamp11(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

static void loadScaling(){
    if(scaleReady) return;
    for(int i=0;i<NUM_FEATS;i++){ fMin[i]=0; fMax[i]=DEF_MAX[i]; }

    std::ifstream ifs("/home/wuy/query_costs/hybench_sf1/row_plan_statistics.json");
    if(ifs.good()){
        json st; try{ ifs>>st;
            for(int i=0;i<NUM_FEATS;i++)
                if(KEY[i] && st.contains(KEY[i]))
                    fMax[i]=std::max(st[KEY[i]].value("max",DEF_MAX[i]),1e-6);
        }catch(...){ logWarn("stats JSON parse error – defaults for scaling"); }
    }else logWarn("stats file missing – defaults for scaling");
    scaleReady=true;
}
static float scl(int i,double v){
    if(!scaleReady) loadScaling();
    v=clamp11(v,fMin[i],fMax[i]);
    return float((v-fMin[i])/(fMax[i]-fMin[i]+1e-12));
}

struct RowSampleStruct{
    float feats[NUM_FEATS];
    int   label;       /* 0/1 , -1=ignore */
    double row_time,column_time,original_query_cost;
    int hybrid_use_imci,fann_use_imci;
};


/* ───────────── feature aggregation ───────────── */
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0;
    double selSum=0,selMin=1e30,selMax=0,ratioSum=0,ratioMax=0;
    int cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,maxDepth=0;
    double fanoutMax=0; bool grp=0,ord=0,tmp=0,hasLimit=0,hasDistinct=0,hasUnion=0,hasConst=0;
    int numUnion=0;
};

/* ------------------------------------------------------------------ *
 *  getBool(obj,key) – safe bool accessor that also understands
 *                     "YES"/"NO" or "true"/"false" strings.
 * ------------------------------------------------------------------ */
static bool getBool(const json &o, const char *k, bool d=false)
{
    if(!o.contains(k)) return d;
    const json &v = o[k];
    if(v.is_boolean())            return v.get<bool>();
    if(v.is_string())
    {
        std::string s = v.get<std::string>();
        std::transform(s.begin(), s.end(), s.begin(),
                       static_cast<int(*)(int)>(std::tolower));
        return (s=="yes"||s=="true"||s=="1");
    }
    return d;
}

void walk(const json& n,Agg& a,int d){
    if(n.is_object()){
        if(n.contains("table")){
            const auto& t=n["table"]; const auto ci=t.value("cost_info",json::object());
            double re=parseNum(t,"rows_examined_per_scan");
            double rp=parseNum(t,"rows_produced_per_join");
            double fl=parseNum(t,"filtered");
            double rc=parseNum(ci,"read_cost"), ec=parseNum(ci,"eval_cost"),
                   pc=parseNum(ci,"prefix_cost");
            double dr=ci.contains("data_read_per_join")?
                      (ci["data_read_per_join"].is_string()? bytesFrom(ci["data_read_per_join"].get<std::string>())
                                                           : parseNum(ci,"data_read_per_join"))
                      :0.0;
            a.re+=re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr;
            if(re>0){ double sel=rp/re; a.selSum+=sel; a.selMin=std::min(a.selMin,sel);
                      a.selMax=std::max(a.selMax,sel); a.fanoutMax=std::max(a.fanoutMax,sel);}
            double ratio=(ec>0? rc/ec: rc); a.ratioSum+=ratio; a.ratioMax=std::max(a.ratioMax,ratio);
            a.cnt++;

            std::string at=t.value("access_type","ALL");
            if(at=="const"){ a.hasConst=true; }
            if(at=="range")      a.cRange++;
            else if(at=="ref")   a.cRef++;
            else if(at=="eq_ref")a.cEq++;
            else if(at=="index") a.cIdx++;
            else                 a.cFull++;
            // if(t.value("using_index",false)) a.idxUse++;
            if(getBool(t,"using_index")) a.idxUse++;
        }
        if(n.contains("limit"))     a.hasLimit=true;
        if(n.contains("distinct"))  a.hasDistinct=true;
        if(n.contains("union_result")){ a.hasUnion=true; a.numUnion++; }

        // if(n.contains("grouping_operation"))            a.grp=true;
        // if(n.contains("ordering_operation")||n.value("using_filesort",false)) a.ord=true;
        // if(n.value("using_temporary_table",false))      a.tmp=true;
        if(n.contains("grouping_operation")) a.grp = true;

        if(n.contains("ordering_operation") ||
           getBool(n,"using_filesort"))                 a.ord = true;

        if(getBool(n,"using_temporary_table"))          a.tmp = true;


        for(const auto& kv:n.items()) if(kv.key()!="table") walk(kv.value(),a,d+1);
    } else if(n.is_array()) for(const auto& e:n) walk(e,a,d);
    a.maxDepth=std::max(a.maxDepth,d);
}

/* ───────── plan → 32-feature vector ───────── */
bool plan2feat(const json& j,float feat[NUM_FEATS]){
    std::fill(feat,feat+NUM_FEATS,0.f);
    if(!j.contains("query_block")){ logWarn("missing query_block"); return false; }

    Agg a; walk(j["query_block"],a,1);

    double rootRows=parseNum(j["query_block"],"rows_produced_per_join");
    // double qCost=parseNum(j["query_block"]["cost_info"],"query_cost");

    const json &ci = j["query_block"].value("cost_info", json::object());
    double qCost   = parseNum(ci, "query_cost");



    if(a.cnt==0) return true;
    double inv=1.0/a.cnt; int k=0;

    feat[k++]=safeLog1p(a.re*inv);   feat[k++]=safeLog1p(a.rp*inv);   feat[k++]=safeLog1p(a.f*inv);
    feat[k++]=safeLog1p(a.rc*inv);   feat[k++]=safeLog1p(a.ec*inv);   feat[k++]=safeLog1p(a.pc*inv);
    feat[k++]=safeLog1p(a.dr*inv);

    feat[k++]=a.cRange*inv; feat[k++]=a.cRef*inv;  feat[k++]=a.cEq*inv;
    feat[k++]=a.cIdx*inv;   feat[k++]=a.cFull*inv;

    feat[k++]=a.idxUse*inv;

    feat[k++]=a.selSum*inv; feat[k++]=a.selMin; feat[k++]=a.selMax;

    feat[k++]=a.maxDepth;   feat[k++]=a.fanoutMax;

    feat[k++]=a.grp;  feat[k++]=a.ord; feat[k++]=a.tmp;

    feat[k++]=a.ratioSum*inv; feat[k++]=a.ratioMax;

    feat[k++]=safeLog1p(qCost);

    /* new ones */
    feat[k++]=safeLog1p(rootRows);     // 24

    feat[k++]=a.hasConst;              // 25
    feat[k++]=a.hasLimit;              // 26
    feat[k++]=a.hasDistinct;           // 27

    feat[k++]=a.hasUnion;              // 28
    feat[k++]=std::min(a.numUnion,10); // 29 (cap)

    double selMean=a.selSum*inv;
    feat[k++]=selMean * a.maxDepth;                       // 30   interaction
    double fullFrac = a.cFull? a.cFull*inv : 1e-3;
    feat[k++]=(a.idxUse*inv) / fullFrac;                  // 31   idxFrac / fullFrac

    /* scale 0-1 */
    for(int i=0;i<NUM_FEATS;i++) feat[i]=scl(i,feat[i]);
    return true;
}


/* ------------------------------------------------------------------ *
 *  loadPlanFeatures()  –  just open+parse JSON, then call plan2feat
 *  Returns false if the file can’t be opened, can’t be parsed, or
 *  lacks a "query_block" section.
 * ------------------------------------------------------------------ */
static bool loadPlanFeatures(const std::string &path,
                             float out[NUM_FEATS],
                             double &qryCost /* ignored by caller */)
{
    std::ifstream in(path.c_str());
    if(!in.good())
    {
        logWarn("cannot open plan JSON: " + path);
        return false;
    }

    json j;
    try { in >> j; }
    catch(...)
    {
        logWarn("parse error in: " + path);
        return false;
    }

    // the new extractor already checks for "query_block"
    bool ok = plan2feat(j, out);
    // qryCost = j.value("query_block", json::object())
    //          .value("cost_info",    json::object())
    //          .value("query_cost",   0.0);
    /* cost_info can be an object, a string, or even missing – use
    parseNum() to be tolerant and avoid type_error.302               */
    qryCost = 0.0;
    if(j.contains("query_block"))
    {
        const json &qb = j["query_block"];
        if(qb.contains("cost_info"))
            qryCost = parseNum(qb["cost_info"], "query_cost");
    }

    return ok;
}


// -----------------------------------------------------------------------------
/*
 * ExecutionPlanDataset Class
 */
// -----------------------------------------------------------------------------
class ExecutionPlanDatasetClass {
private:
    std::vector<RowSampleStruct> data_;
public:
    ExecutionPlanDatasetClass(const std::vector<std::string> &dataDirList, double timeout){
        if(dataDirList.empty()){
            logError("No dataDirList provided.");
            return;
        }

        int skipCount = 0, validCount = 0;

        for(const auto &dDir_name : dataDirList){
            std::string dDir = "/home/wuy/query_costs/" + dDir_name;
            logInfo("=== Loading dataset from directory: " + dDir + " ===");
            std::string csvPath    = dDir + "/query_costs.csv";
            std::string rowPlanDir = dDir + "/row_plans";

            if(!fileExists(csvPath)){
                logWarn("CSV file does not exist: " + csvPath);
                continue;
            }

            DIR *dir = opendir(rowPlanDir.c_str());
            if(dir == nullptr){
                logWarn("Cannot open row_plans directory: " + rowPlanDir);
                continue;
            }
            closedir(dir); // We'll handle individual file checks later

            // count lines
            std::ifstream counter(csvPath.c_str());
            if(!counter.is_open()){
                logError("Cannot open CSV => "+ csvPath);
                return;
            }
            size_t totalLines=0;
            {
                std::string tmp;
                while(std::getline(counter, tmp)) totalLines++;
            }

            // load data
            std::ifstream ifs(csvPath.c_str());
            if(!ifs.is_open()){
                logWarn("Cannot open CSV => " + csvPath);
                continue;
            }
            // skip header
            std::string line;
            std::getline(ifs, line);
            size_t currentLine=0;
            size_t dataLines= (totalLines>0? totalLines-1:0);
            // Process each line
            while(std::getline(ifs, line)){
                currentLine++;
                showProgressBar(currentLine, dataLines);
                if(line.empty()) continue;
                std::stringstream ss(line);
                std::string qidStr, labelStr, rowTimeStr, columnTimeStr, hybridUseImciStr, fannUseImciStr;
                std::getline(ss, qidStr, ',');
                std::getline(ss, labelStr, ',');
                std::getline(ss, rowTimeStr, ',');
                std::getline(ss, columnTimeStr, ',');
                std::getline(ss, hybridUseImciStr, ',');
                std::getline(ss, fannUseImciStr, ',');

                if(qidStr.empty() || labelStr.empty()){
                    skipCount++;
                    continue;
                }

                int lab = 0;
                double rowTime = 0.0, columnTime = 0.0;
                lab = std::stoi(labelStr);
                try{
                    rowTime = std::stod(rowTimeStr);
                }
                catch(...){
                    continue;
                    rowTime = timeout;
                }
                try{
                    columnTime = std::stod(columnTimeStr);
                }
                catch(...){
                    continue;
                    columnTime = timeout;
                }
                if(std::fabs(rowTime - columnTime) < 1e-3){
                    lab = -1; // Ignore queries with no difference
                }
              
                int hybridUseImci = std::stoi(hybridUseImciStr);

                int fannUseImci   = std::stoi(fannUseImciStr);

                std::string planPath = rowPlanDir + "/" + qidStr + ".json";
                if(!fileExists(planPath)){
                    skipCount++;
                    continue;
                }

                float feats[9];
                double original_query_cost = 0.0;
                if(!loadPlanFeatures(planPath, feats, original_query_cost)){
                    skipCount++;
                    continue;
                }

                RowSampleStruct rs;
                for(int i=0; i<NUM_FEATS; i++) rs.feats[i] = feats[i];
                rs.label = lab;
                rs.row_time = rowTime;
                rs.column_time = columnTime;
                rs.original_query_cost = original_query_cost;
                rs.hybrid_use_imci = hybridUseImci;
                rs.fann_use_imci   = fannUseImci;
                data_.push_back(rs);
                validCount++;
            }
        }

        logInfo("Total valid samples => " + std::to_string(validCount) +
                ", skipped => " + std::to_string(skipCount));
    }

    size_t size() const { return data_.size(); }
    const RowSampleStruct& operator[](size_t idx) const { return data_[idx]; }
    const std::vector<RowSampleStruct>& getData() const { return data_; }
};

// -----------------------------------------------------------------------------
/*
 * WeightedRandomSampler Class
 */
// -----------------------------------------------------------------------------
class WeightedRandomSamplerClass {
private:
    std::vector<double> weights_;
    std::mt19937 rng_;
    std::discrete_distribution<> dist_;
public:
    WeightedRandomSamplerClass(const std::vector<int> &labels){
        int count0 = 0, count1 = 0;
        for(auto lb : labels){
            if(lb == 0) count0++;
            else if(lb == 1) count1++;
        }

        double w0 = (count0 > 0) ? 1.0 / count0 : 1.0;
        double w1 = (count1 > 0) ? 1.0 / count1 : 1.0;

        for(auto lb : labels){
            if(lb == 0){
                weights_.push_back(w0);
            }
            else if(lb == 1){
                weights_.push_back(w1);
            }
            // Ignore labels == -1
        }

        dist_ = std::discrete_distribution<>(weights_.begin(), weights_.end());
        rng_.seed(std::random_device{}());
    }

    size_t sample(){
        return dist_(rng_);
    }
};

// ---------------------------------------------------------------
// Adjusted TrainSampleStruct to hold continuous target (gap)
struct TrainSampleStruct{
    std::vector<float> input; // NUM_FEATS features
    float targetGap;          // (row_time - column_time)
};


// Prepare regression samples: now target is gap
std::vector<TrainSampleStruct>
prepareRegressionSamples(const ExecutionPlanDatasetClass &ds,
                         const std::vector<size_t> &trainIdx) {
    std::vector<TrainSampleStruct> samples;
    for(auto idx : trainIdx){
        const RowSampleStruct &rs = ds[idx];
        if(rs.label == -1) continue; // ignore
        TrainSampleStruct ts;
        ts.input.assign(rs.feats, rs.feats + NUM_FEATS);
        ts.targetGap = float(rs.row_time - rs.column_time);
        samples.push_back(ts);
    }
    return samples;
}

// ---------------------------------------------------------------
// Updated FannModel: use linear output activation for regression
struct FannModel{
    struct fann *ann;
    FannModel(int num_input, int num_hidden, int num_output, float learning_rate){
        // 3-layer: input -> hidden -> output (linear)
        ann = fann_create_standard(3, num_input, num_hidden, num_output);
        fann_set_activation_function_hidden(ann, FANN_SIGMOID);
        fann_set_activation_function_output(ann, FANN_LINEAR);
        fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
        fann_set_learning_rate(ann, learning_rate);
    }
    ~FannModel(){ if(ann) fann_destroy(ann); }

    void train(const std::vector<TrainSampleStruct> &samples) {
        size_t total = samples.size();
        for(size_t i = 0; i < total; ++i){
            const auto &s = samples[i];
            fann_type *input = const_cast<fann_type*>(s.input.data());
            fann_type desired = s.targetGap;
            fann_train(ann, input, &desired);
            showProgressBar(i+1, total);
        }
    }

    float predict(const std::vector<float> &input_vec) {
        fann_type *output = fann_run(ann, const_cast<fann_type*>(input_vec.data()));
        return float(output[0]);
    }

    void save(const std::string &path){ fann_save(ann, path.c_str()); }
    void load(const std::string &path){
        if(ann) fann_destroy(ann);
        ann = fann_create_from_file(path.c_str());
    }
};


// -----------------------------------------------------------------------------
/*
 * Define the Dataset Splitting Function
 */
// -----------------------------------------------------------------------------
std::pair<std::vector<size_t>, std::vector<size_t>> splitDataset(size_t total, double train_ratio=0.8){
    std::vector<size_t> indices(total);
    for(size_t i=0; i<total; i++) indices[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t train_size = static_cast<size_t>(train_ratio * total);
    std::vector<size_t> trainIdx(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> valIdx(indices.begin() + train_size, indices.end());

    return {trainIdx, valIdx};
}



// ---------------------------------------------------------------
// Entry point: regression training loop with epoch-level logging
// ---------------------------------------------------------------
int main(int argc, char *argv[]){
    // Default parameters
    std::vector<std::string> dataDirs;
    int epochs = 10000;
    int hiddenNeurons = 128;
    float learning_rate = 0.001f;
    std::string bestModelPath = "checkpoints/rowmlp_gap.net";
    bool skipTrain = false;

    // Parse command-line arguments
    for(int i=1; i<argc; ++i){
        if(std::strncmp(argv[i], "--data_dirs=", 12) == 0){
            dataDirs.push_back(std::string(argv[i] + 12));
        } else if(std::strncmp(argv[i], "--epochs=", 9) == 0){
            epochs = std::stoi(std::string(argv[i] + 9));
        } else if(std::strncmp(argv[i], "--hidden_neurons=", 17) == 0){
            hiddenNeurons = std::stoi(std::string(argv[i] + 17));
        } else if(std::strncmp(argv[i], "--lr=", 5) == 0){
            learning_rate = std::stof(std::string(argv[i] + 5));
        } else if(std::strncmp(argv[i], "--best_model_path=", 18) == 0){
            bestModelPath = std::string(argv[i] + 18);
        } else if(std::strncmp(argv[i], "--skip_train", 11) == 0){
            skipTrain = true;
            logInfo("** Skipping training; only evaluating using regression sign. **");
        }
    }
    logInfo("Best model path => " + bestModelPath);
    if(dataDirs.empty()){
        logWarn("No data_dirs provided. Using defaults.");
        dataDirs = {"tpch_sf1_templates","tpch_sf1_templates_index",
                    "tpch_sf1_zsce_index_TP","tpch_sf100_zsce"};
    }

    double timeout = 60.0;
    // 1) Load dataset
    ExecutionPlanDatasetClass dataset(dataDirs, timeout);
    size_t dsSize = dataset.size();
    logInfo("Combined dataset size => " + std::to_string(dsSize));
    if(dsSize < 30){ logError("Not enough data. Exiting."); return 1; }

    // 2) Split into train/val
    auto splits = splitDataset(dsSize, 0.8);
    std::vector<size_t> trainIdx = splits.first;
    std::vector<size_t> valIdx   = splits.second;
    logInfo("Train samples = " + std::to_string(trainIdx.size()) +
            ", Val samples = " + std::to_string(valIdx.size()));

    // 3) Prepare regression samples
    auto trainingSamples = prepareRegressionSamples(dataset, trainIdx);

    // 4) Init or load model
    FannModel *model = new FannModel(NUM_FEATS, hiddenNeurons, 1, learning_rate);
    if(!skipTrain){
        double bestValMSE = 1e9;
        for(int epoch=1; epoch<=epochs; ++epoch){
            // Train epoch
            model->train(trainingSamples);

            // Compute training MSE
            double trainMSE = 0;
            for(const auto &s: trainingSamples){
                std::vector<float> in = s.input;
                float pred = model->predict(in);
                double err = pred - s.targetGap;
                trainMSE += err*err;
            }
            trainMSE /= trainingSamples.size();

            // Compute validation MSE
            double valMSE = 0; int cnt=0;
            for(auto idx: valIdx){
                const auto &rs = dataset[idx];
                if(rs.label==-1) continue;
                std::vector<float> in(rs.feats, rs.feats+NUM_FEATS);
                float pred = model->predict(in);
                double trueGap = rs.row_time - rs.column_time;
                double err = pred - trueGap;
                valMSE += err*err; ++cnt;
            }
            valMSE = cnt? valMSE / cnt : 0;

            // Log epoch and losses
            logInfo("Epoch " + std::to_string(epoch) + "/" + std::to_string(epochs) +
                    " => Train MSE=" + std::to_string(trainMSE) +
                    ", Val MSE=" + std::to_string(valMSE));

            // Save best
            if(valMSE < bestValMSE){ bestValMSE = valMSE; model->save(bestModelPath); }
        }
        logInfo("Training complete. Best Val MSE=" + std::to_string(bestValMSE));
    } else {
        model->load(bestModelPath);
        logInfo("Loaded regression model from " + bestModelPath);
    }

    // 5) Final evaluation: confusion & runtimes
    size_t TP=0,FP=0,TN=0,FN=0;
    double rt_row=0,rt_col=0,rt_cost=0,rt_hybrid=0,rt_reg=0,rt_opt=0;
    for(size_t i=0;i<dsSize;++i){
        const auto &rs = dataset[i];
        rt_row = rt_row*i/(i+1) + rs.row_time/(i+1);
        rt_col = rt_col*i/(i+1) + rs.column_time/(i+1);
        double cost_rt = (rs.original_query_cost>5e4)? rs.column_time:rs.row_time;
        rt_cost = rt_cost*i/(i+1) + cost_rt/(i+1);
        double hy_rt = (rs.hybrid_use_imci==1)? rs.column_time:rs.row_time;
        rt_hybrid = rt_hybrid*i/(i+1) + hy_rt/(i+1);
        double opt_rt = std::min(rs.row_time, rs.column_time);
        rt_opt = rt_opt*i/(i+1) + opt_rt/(i+1);
        // regression
        std::vector<float> in(rs.feats, rs.feats+NUM_FEATS);
        float pred = model->predict(in);
        bool col = (pred>0);
        double reg_rt = col? rs.column_time:rs.row_time;
        rt_reg = rt_reg*i/(i+1) + reg_rt/(i+1);
        // confusion
        if(rs.label>=0){
            if(rs.label==1 && col) TP++;
            else if(rs.label==0 && col) FP++;
            else if(rs.label==0 && !col) TN++;
            else if(rs.label==1 && !col) FN++;
        }
    }
    double prec = TP? double(TP)/(TP+FP):0;
    double rec  = TP? double(TP)/(TP+FN):0;
    double f1   = (prec+rec)? 2*prec*rec/(prec+rec):0;
    double acc  = double(TP+TN)/(TP+TN+FP+FN);

    logInfo("=== CONFUSION (gap>0⇒col) ===");
    logInfo("TP="+std::to_string(TP)+" FP="+std::to_string(FP)+
            " TN="+std::to_string(TN)+" FN="+std::to_string(FN));
    logInfo("Prec="+std::to_string(prec)+" Rec="+std::to_string(rec)+
            " F1="+std::to_string(f1)+" Acc="+std::to_string(acc));

    logInfo("=== AVG RUNTIME ===");
    logInfo("Row only: " + std::to_string(rt_row));
    logInfo("Col only: " + std::to_string(rt_col));
    logInfo("Cost thr: " + std::to_string(rt_cost));
    logInfo("Hybrid : " + std::to_string(rt_hybrid));
    logInfo("Reg sign: " + std::to_string(rt_reg));
    logInfo("Optimal: " + std::to_string(rt_opt));

    delete model;
    return 0;
}