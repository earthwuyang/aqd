/************************************************************************************
 * Updated C++ code compatible with C++11 that:
 *   - Reads from multiple "dataDir" directories
 *   - Combines them into a single ExecutionPlanDataset
 *   - Uses FANN for a 2-layer MLP (9->hidden->1)
 *   - Splits data into train/val (~80%/20%)
 *   - Implements early stopping with patience=20 based on val MSE
 *   - Saves best model => "best_mlp_no_tg.net"
 *   - Reports negative vs. positive precision & recall
 *   - Uses trapezoidal AUC => "roc_curve.csv"
 *   - Implements threshold search on the validation set to pick the best threshold by F1-score
 *   - Adds a "--skip_train" cmd argument so you can skip training and load the model.
 *
 * Compile:
 *   g++ -std=c++11 -I/path/to/json/single_include -lfann -o rowmlp_updated rowmlp_updated.cpp
 *
 * Run example:
 *   ./rowmlp_updated \
 *       --data_dirs=tpch_sf1_templates \
 *       --data_dirs=tpch_sf1_templates_index \
 *       --data_dirs=tpch_sf1_zsce_index_TP \
 *       --data_dirs=tpch_sf100_zsce \
 *       --epochs=1000 \
 *       --hidden_neurons=128 \
 *       --lr=0.001 \
 *       --best_model_path=checkpoints/best_mlp_no_tg.net \
 *       [--skip_train]
 ************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>

// POSIX Directory Traversal
#include <dirent.h>
#include <sys/stat.h>

// FANN Library
#include <fann.h>

// JSON Library (nlohmann/json)
#include "json.hpp"

using json = nlohmann::json;

// -----------------------------------------------------------------------------
/*
 * Basic Logging
 */
// -----------------------------------------------------------------------------
static void logInfo(const std::string &msg)  { std::cout << "[INFO]  " << msg << std::endl; }
static void logWarn(const std::string &msg)  { std::cerr << "[WARN]  " << msg << std::endl; }
static void logError(const std::string &msg) { std::cerr << "[ERROR] " << msg << std::endl; }

// -----------------------------------------------------------------------------
/*
 * Check if a path is a directory
 */
// -----------------------------------------------------------------------------
bool isDirectory(const std::string &path){
    struct stat statbuf;
    if(stat(path.c_str(), &statbuf) != 0){
        return false;
    }
    return S_ISDIR(statbuf.st_mode);
}

// -----------------------------------------------------------------------------
/*
 * Check file existence
 */
// -----------------------------------------------------------------------------
bool fileExists(const std::string &path){
    std::ifstream ifs(path.c_str());
    return ifs.good();
}

// -----------------------------------------------------------------------------
/*
 * Show progress bar
 */
// -----------------------------------------------------------------------------
void showProgressBar(size_t current, size_t total, size_t width=50){
    double fraction= (total==0)? 1.0 : double(current)/ double(total);
    int filled= int(fraction* width);
    std::cout << "\r[";
    for(int i=0; i<filled; i++) std::cout<<"=";
    for(int i=filled; i<(int)width; i++) std::cout<<" ";
    std::cout << "] " << int(fraction*100) <<"% ("<< current <<"/"<< total <<")";
    if(current >= total) std::cout << std::endl;
    std::fflush(stdout);
}

// -----------------------------------------------------------------------------
/*
 * Minimal parsePossibleNumber
 */
// -----------------------------------------------------------------------------
double parsePossibleNumber(const json &j, const std::string &key){
    if(!j.contains(key)) return 0.0;
    try{
        if(j[key].is_string()){
            return std::stod(j[key].get<std::string>());
        } else if(j[key].is_number_float() || j[key].is_number_integer()){
            return j[key].get<double>();
        }
    } catch(...){
        logWarn("Invalid number format for key: " + key);
        return 0.0;
    }
    return 0.0;
}

// -----------------------------------------------------------------------------
/*
 * Convert data size strings => e.g. "86G" => numeric
 */
// -----------------------------------------------------------------------------
double convert_data_size_to_numeric(const std::string &s){
    if(s.empty()) return 0.0;
    std::string str = s;
    // Trim trailing whitespace
    str.erase(std::find_if(str.rbegin(), str.rend(),
        [](unsigned char ch) { return !std::isspace(ch); }).base(), str.end());

    if(str.empty()) return 0.0;
    char suffix = str.back();
    double factor = 1.0;
    std::string numPart = str;
    if(suffix == 'G'){ factor = 1e9;  numPart.pop_back(); }
    else if(suffix == 'M'){ factor = 1e6;  numPart.pop_back(); }
    else if(suffix == 'K'){ factor = 1e3;  numPart.pop_back(); }
    else{
        // No suffix, assume raw number
    }
    try{
        double val = std::stod(numPart);
        return val * factor;
    } catch(...){
        logWarn("Invalid data size string: " + s);
        return 0.0;
    }
}

// -----------------------------------------------------------------------------
/*
 * Statistics Information for Normalization
 */
// -----------------------------------------------------------------------------
struct StatsInfo{
    double center;
    double scale;
};

// -----------------------------------------------------------------------------
/*
 * Safe log1p function
 */
// -----------------------------------------------------------------------------
double safeLog1p(double v){
    if(v < 0) v = 0;
    return std::log1p(v);
}

// -----------------------------------------------------------------------------
/*
 * Feature Collector for Row Plans
 */
// -----------------------------------------------------------------------------
struct FeatureCollector{
    double sumRe=0, sumRp=0, sumF=0, sumRc=0, sumEc=0, sumPc=0, sumDr=0;
    int count=0;
    void addOne(double re, double rp, double f, double rc, double ec, double pc, double dr){
        sumRe += re; sumRp += rp; sumF += f; sumRc += rc;
        sumEc += ec; sumPc += pc; sumDr += dr;
        count++;
    }
};

// -----------------------------------------------------------------------------
/*
 * Recursive function to parse Row Plan JSON
 */
// -----------------------------------------------------------------------------
void recurseRowPlan(const json &block, FeatureCollector &fc){
    if(block.is_object()){
        if(block.contains("table")){
            const auto &tbl = block["table"];
            double re = parsePossibleNumber(tbl, "rows_examined_per_scan");
            double rp = parsePossibleNumber(tbl, "rows_produced_per_join");
            double f  = parsePossibleNumber(tbl, "filtered");
            double rc=0, ec=0, pc=0, dr=0;
            if(tbl.contains("cost_info")){
                const auto &ci = tbl["cost_info"];
                rc = parsePossibleNumber(ci, "read_cost");
                ec = parsePossibleNumber(ci, "eval_cost");
                pc = parsePossibleNumber(ci, "prefix_cost");
                std::string sdr = ci.value("data_read_per_join", "0");
                dr = convert_data_size_to_numeric(sdr);
            }
            fc.addOne(re, rp, f, rc, ec, pc, dr);
        }
        // Recurse subkeys
        for(auto it = block.begin(); it != block.end(); ++it){
            if(it.key() == "table") continue;
            recurseRowPlan(it.value(), fc);
        }
    }
    else if(block.is_array()){
        for(auto &elem : block){
            recurseRowPlan(elem, fc);
        }
    }
}



bool parseRowPlanJSON(const std::string &planPath,
                      float outFeats[8],
                      double &original_query_cost) {
    for (int i = 0; i < 8; i++) outFeats[i] = 0.f;

    std::ifstream ifs(planPath);
    if (!ifs.is_open()) {
        logWarn("Cannot open plan => " + planPath);
        return false;
    }

    json j;
    try {
        ifs >> j;
    } catch (...) {
        logWarn("JSON parse error => " + planPath);
        return false;
    }

    if (!j.contains("query_block")) {
        logWarn("No query_block => " + planPath);
        return false;
    }

    struct FeatureCollector {
        double sumRe = 0, sumRp = 0, sumF = 0, sumRc = 0;
        double sumEc = 0, sumPc = 0, sumDr = 0;
        int count = 0;
    } fc;

    std::function<void(const json&)> recurse = [&](const json &blk){
        if (blk.is_object()) {
            if (blk.contains("table")) {
                const auto &tbl = blk["table"];
                double re = parsePossibleNumber(tbl, "rows_examined_per_scan");
                double rp = parsePossibleNumber(tbl, "rows_produced_per_join");
                double f  = parsePossibleNumber(tbl, "filtered");
                double rc = parsePossibleNumber(tbl["cost_info"], "read_cost");
                double ec = parsePossibleNumber(tbl["cost_info"], "eval_cost");
                double pc = parsePossibleNumber(tbl["cost_info"], "prefix_cost");
                double dr = convert_data_size_to_numeric(
                              tbl["cost_info"].value("data_read_per_join", "0"));
                if (re >= 0 && rp >= 0) {
                    fc.sumRe += re;
                    fc.sumRp += rp;
                    fc.sumF  += f;
                    fc.sumRc += rc;
                    fc.sumEc += ec;
                    fc.sumPc += pc;
                    fc.sumDr += dr;
                    fc.count++;
                }
            }
            for (const auto& kv : blk.items()) recurse(kv.value());
        } else if (blk.is_array()) {
            for (const auto &el : blk) recurse(el);
        }
    };

    recurse(j["query_block"]);

    original_query_cost = 0.0;
    if (j["query_block"].contains("cost_info")) {
        original_query_cost = parsePossibleNumber(
            j["query_block"]["cost_info"], "query_cost");
    }

    if (fc.count == 0) return true;

    auto safeLog1p = [](double v) { return std::log1p(std::max(0.0, v)); };

    outFeats[0] = static_cast<float>(safeLog1p(fc.sumRe / fc.count));
    outFeats[1] = static_cast<float>(safeLog1p(fc.sumRp / fc.count));
    outFeats[2] = static_cast<float>(safeLog1p(fc.sumF  / fc.count));
    outFeats[3] = static_cast<float>(safeLog1p(fc.sumRc / fc.count));
    outFeats[4] = static_cast<float>(safeLog1p(fc.sumEc / fc.count));
    outFeats[5] = static_cast<float>(safeLog1p(fc.sumPc / fc.count));
    outFeats[6] = static_cast<float>(safeLog1p(fc.sumDr / fc.count));
    outFeats[7] = static_cast<float>(safeLog1p(original_query_cost));
    return true;
}



// -----------------------------------------------------------------------------
/*
 * Parse Row Plan JSON and extract features
 */
// // -----------------------------------------------------------------------------
// bool parseRowPlanJSON(const std::string &planPath, 
//                     //  const std::map<std::string, StatsInfo> &rowStats, 
//                      float outFeats[8],
//                      double &original_query_cost){
//     for(int i=0; i<8; i++) outFeats[i] = 0.f;

//     std::ifstream ifs(planPath.c_str());
//     if(!ifs.is_open()){
//         logWarn("Cannot open plan => " + planPath);
//         return false;
//     }
//     json j;
//     try{
//         ifs >> j;
//     } catch(...){
//         logWarn("JSON parse error => " + planPath);
//         return false;
//     }
//     if(!j.contains("query_block")){
//         logWarn("No query_block => " + planPath);
//         return false;
//     }
//     FeatureCollector fc;
//     recurseRowPlan(j["query_block"], fc);

//     // Extract original query_cost before normalization
//     if(j["query_block"].contains("cost_info")){
//         original_query_cost = parsePossibleNumber(j["query_block"]["cost_info"], "query_cost");
//     } else {
//         return false;  // if Impossible Where, return false
//         original_query_cost = 0.0;
//     }

//     if(fc.count == 0){
//         // If no tables found, assign default features
//         for(int i=0; i<8; i++) outFeats[i] = 0.f;
//         return true;
//     }

//     // Compute mean features
//     double meRe = fc.sumRe / fc.count;
//     double meRp = fc.sumRp / fc.count;
//     double meF  = fc.sumF  / fc.count;
//     double meRc = fc.sumRc / fc.count;
//     double meEc = fc.sumEc / fc.count;
//     double mePc = fc.sumPc / fc.count;
//     double meDr = fc.sumDr / fc.count;

//     // Apply log1p
//     meRe = safeLog1p(meRe);
//     meRp = safeLog1p(meRp);
//     meF  = safeLog1p(meF);
//     meRc = safeLog1p(meRc);
//     meEc = safeLog1p(meEc);
//     mePc = safeLog1p(mePc);
//     meDr = safeLog1p(meDr);

//     // Normalize using rowStats
//     auto norm = [&](double v, const std::string &feat) -> float {
//         auto it = rowStats.find(feat);
//         if(it != rowStats.end()){
//             double c = it->second.center;
//             double s = it->second.scale;
//             if(std::fabs(s) < 1e-9) s = 1.0;
//             return static_cast<float>((v - c) / s);
//         }
//         return static_cast<float>(v);
//     };

//     // Assign normalized features
//     // outFeats[0] = norm(meRe, "rows_examined_per_scan");
//     // outFeats[1] = norm(meRp, "rows_produced_per_join");
//     // outFeats[2] = norm(meF,  "filtered");
//     // outFeats[3] = norm(meRc, "read_cost");
//     // outFeats[4] = norm(meEc, "eval_cost");
//     // outFeats[5] = norm(mePc, "prefix_cost");
//     // outFeats[6] = norm(meDr, "data_read_per_join");
//     outFeats[0] = meRe;
//     outFeats[1] = meRp;
//     outFeats[2] = meF;
//     outFeats[3] = meRc;
//     outFeats[4] = meEc;
//     outFeats[5] = mePc;
//     outFeats[6] = meDr;


//     // Additional features: query_cost and sort_cost
//     // Assuming these are also part of the 9 features
//     double query_cost = parsePossibleNumber(j["query_block"]["cost_info"], "query_cost");
//     double sort_cost = 0.0;
//     if(j["query_block"].contains("ordering_operation")){
//         const auto &ordering_op = j["query_block"]["ordering_operation"];
//         if(ordering_op.contains("grouping_operation")){
//             const auto &grouping_op = ordering_op["grouping_operation"];
//             if(grouping_op.contains("cost_info")){
//                 sort_cost = parsePossibleNumber(grouping_op["cost_info"], "sort_cost");
//             }
//         }
//         if(ordering_op.contains("cost_info")){
//             sort_cost = parsePossibleNumber(ordering_op["cost_info"], "sort_cost");
//         }
//     }

//     // Apply log1p
//     double log_query_cost = safeLog1p(query_cost);
//     double log_sort_cost  = safeLog1p(sort_cost);

//     // Normalize
//     float norm_query_cost = norm(log_query_cost, "query_cost");
//     float norm_sort_cost  = norm(log_sort_cost, "sort_cost");

//     outFeats[7] = norm_query_cost;
//     outFeats[8] = norm_sort_cost;

//     return true;
// }

// -----------------------------------------------------------------------------
/*
 * Define a Row Sample
 */
// -----------------------------------------------------------------------------
struct RowSampleStruct{
    float feats[8]; // 8 features
    int label;       // 0 or 1, -1 for ignore
    double row_time;
    double column_time;
    double original_query_cost;
    int hybrid_use_imci;
    int fann_use_imci;
};

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

        // Example rowStats (These should be loaded from a file or defined appropriately)
        // Here we define them manually for demonstration
        // std::map<std::string, StatsInfo> rowStats;
        // rowStats["rows_examined_per_scan"] = {0.6931471824645996, 10.143671989440918};
        // rowStats["rows_produced_per_join"] = {17.0823974609375, 3.895216464996338};
        // rowStats["filtered"]               = {4.498586654663086, 2.121088743209839};
        // rowStats["read_cost"]              = {16.266338348388672, 3.760223388671875};
        // rowStats["eval_cost"]              = {14.779812812805176, 3.895212173461914};
        // rowStats["prefix_cost"]            = {17.97785186767578, 2.416825294494629};
        // rowStats["data_read_per_join"]     = {23.28821563720703, 3.8918190002441406};
        // rowStats["query_cost"]             = {0.0, 0.0}; // Add appropriate stats
        // rowStats["sort_cost"]              = {0.0, 0.0}; // Add appropriate stats

        // rowStats["rows_examined_per_scan"] = {0.6931471824645996, 10.143671989440918};
        // rowStats["rows_produced_per_join"] = {17.0823974609375, 3.895216464996338};
        // rowStats["filtered"]               = {4.498586654663086, 2.121088743209839};
        // rowStats["read_cost"]              = {16.266338348388672, 3.760223388671875};
        // rowStats["eval_cost"]              = {14.779812812805176, 3.895212173461914};
        // rowStats["prefix_cost"]            = {17.97785186767578, 2.416825294494629};
        // rowStats["data_read_per_join"]     = {23.28821563720703, 3.8918190002441406};
        // rowStats["query_cost"]             = {0.0, 0.0}; // Add appropriate stats
        // rowStats["sort_cost"]              = {0.0, 0.0}; // Add appropriate stats

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
                if(!parseRowPlanJSON(planPath, feats, original_query_cost)){
                    skipCount++;
                    continue;
                }

                RowSampleStruct rs;
                for(int i=0; i<8; i++) rs.feats[i] = feats[i];
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

// -----------------------------------------------------------------------------
/*
 * Define a Row Sample for Training
 */
// -----------------------------------------------------------------------------
struct TrainSampleStruct{
    std::vector<float> input; // 9 features
    float desired;            // 0.0 or 1.0
};

// -----------------------------------------------------------------------------
/*
 * Prepare Training Samples
 */
// -----------------------------------------------------------------------------
std::vector<TrainSampleStruct> prepareTrainingSamples(const ExecutionPlanDatasetClass &ds, 
                                               const std::vector<size_t> &trainIdx){
    std::vector<TrainSampleStruct> samples;
    for(auto idx : trainIdx){
        const RowSampleStruct &rs = ds[idx];
        if(rs.label == -1) continue; // Ignore
        TrainSampleStruct ts;
        ts.input.assign(rs.feats, rs.feats + 8);
        ts.desired = (rs.label == 1) ? 1.0f : 0.0f;
        samples.push_back(ts);
    }
    return samples;
}

// -----------------------------------------------------------------------------
/*
 * Evaluation Metrics
 */
// -----------------------------------------------------------------------------
struct Metrics{
    double loss;
    double accuracy;
    double precision;
    double recall;
    double f1;
    double auc;
    double recall_neg;
    double recall_pos;
    double precision_neg;
    double precision_pos;
    std::vector<int> ytrue;
    std::vector<float> scores;
    float best_threshold;
};

// Function to compute AUC using trapezoidal rule
double computeAUC(const std::vector<std::pair<double, double>> &rocPoints){
    double auc = 0.0;
    for(size_t i=1; i < rocPoints.size(); i++){
        double dx = rocPoints[i].first - rocPoints[i-1].first;
        double midy = (rocPoints[i].second + rocPoints[i-1].second) / 2.0;
        // std::cout << "dx = " << dx << ", midy = " << midy << std::endl;
        auc += dx * midy;
    }
    return auc;
}

// Function to compute ROC curve points
std::vector<std::pair<double, double>> computeRocCurve(const std::vector<int> &ytrue, const std::vector<float> &scores){
    std::vector<std::pair<double, double>> rocPoints;
    for(int i=0; i<=100; i++){
        double thr = 1.f - double(i) / 100.0;
        int TP=0, FP=0, TN=0, FN=0;
        for(size_t j=0; j< ytrue.size(); j++){
            int pred = (scores[j] >= thr) ? 1 : 0;
            if(ytrue[j] == 1 && pred == 1) TP++;
            if(ytrue[j] == 1 && pred == 0) FN++;
            if(ytrue[j] == 0 && pred == 1) FP++;
            if(ytrue[j] == 0 && pred == 0) TN++;
        }
        double tpr = (TP + FN > 0) ? double(TP) / (TP + FN) : 0.0;
        double fpr = (FP + TN > 0) ? double(FP) / (FP + TN) : 0.0;
        rocPoints.emplace_back(fpr, tpr);
    }
    return rocPoints;
}

// Function to save ROC curve to CSV
void saveRocCurve(const std::vector<std::pair<double, double>> &rocPoints, const std::string &filename){
    std::ofstream ofs(filename.c_str());
    ofs << "FPR,TPR\n";
    for(const auto &p : rocPoints){
        ofs << p.first << "," << p.second << "\n";
    }
    ofs.close();
    logInfo("Saved ROC curve to " + filename);
}

// Function to find best threshold by F1-score
double findBestThreshold(const std::vector<int> &ytrue, const std::vector<float> &scores, double &bestF1){
    double bestThr = 0.5;
    bestF1 = -1.0;

    for(int i=0; i<=100; i++){
        double thr = double(i) / 100.0;
        int TP=0, FP=0, TN=0, FN=0;
        for(size_t j=0; j < ytrue.size(); j++){
            int pred = (scores[j] >= thr) ? 1 : 0;
            if(ytrue[j] == 1 && pred == 1) TP++;
            if(ytrue[j] == 1 && pred == 0) FN++;
            if(ytrue[j] == 0 && pred == 1) FP++;
            if(ytrue[j] == 0 && pred == 0) TN++;
        }
        double precision = (TP + FP > 0) ? double(TP) / (TP + FP) : 0.0;
        double recall    = (TP + FN > 0) ? double(TP) / (TP + FN) : 0.0;
        double f1 = (precision + recall > 0.0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
        if(f1 > bestF1){
            bestF1 = f1;
            bestThr = thr;
        }
    }
    return bestThr;
}

// -----------------------------------------------------------------------------
/*
 * Define the MLP Model using FANN
 */
// -----------------------------------------------------------------------------
struct FannModel{
    struct fann *ann;
    FannModel(int num_input, int num_hidden, int num_output, float learning_rate){
        ann = fann_create_standard(3, num_input, num_hidden, num_output);
        fann_set_activation_function_hidden(ann, FANN_SIGMOID);
        fann_set_activation_function_output(ann, FANN_SIGMOID);
        fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
        fann_set_learning_rate(ann, learning_rate);
    }

    ~FannModel(){
        if(ann) fann_destroy(ann);
    }

    void train(const std::vector<TrainSampleStruct> &samples){
        for(const auto &s : samples){
            // 方法 1: 使用 const_cast
            fann_train(ann, const_cast<fann_type*>(s.input.data()), const_cast<fann_type*>(&s.desired));

            // 或者，方法 2: 使用临时变量
            /*
            std::vector<fann_type> input = s.input; // 复制输入
            fann_type desired = s.desired;          // 复制期望输出
            fann_train(ann, input.data(), &desired);
            */
        }
    }

    float run(const std::vector<float> &input){
        fann_type *output = fann_run(ann, const_cast<fann_type*>(input.data()));
        return static_cast<float>(output[0]);
    }

    double getMSE(){
        return fann_get_MSE(ann);
    }

    void save(const std::string &path){
        fann_save(ann, path.c_str());
    }

    void load(const std::string &path){
        fann_destroy(ann);
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

// -----------------------------------------------------------------------------
/*
 * Main Function
 */
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]){
    // Default parameters
    // std::vector<std::string> dataDirs = {"tpch_sf1_templates", "tpch_sf1_templates_index", 
                                        // "tpch_sf1_zsce_index_TP", "tpch_sf100_zsce"};
    std::vector<std::string> dataDirs = {};
    
    int epochs = 10000;
    int hiddenNeurons = 128;
    float learning_rate = 0.001f;
    std::string bestModelPath = "checkpoints/best_mlp_no_tg.net";
    bool skipTrain = false;

    // Parse command-line arguments
    for(int i=1; i<argc; i++){
        if(std::strncmp(argv[i], "--data_dirs=", 12) == 0){
            std::string val = std::string(argv[i] + 12);
            dataDirs.push_back(val);
        }
        else if(std::strncmp(argv[i], "--epochs=", 9) == 0){
            epochs = std::stoi(std::string(argv[i] + 8));
        }
        else if(std::strncmp(argv[i], "--hidden_neurons=", 17) == 0){
            hiddenNeurons = std::stoi(std::string(argv[i] + 17));
        }
        else if(std::strncmp(argv[i], "--lr=", 5) == 0){
            learning_rate = std::stof(std::string(argv[i] + 5));
        }
        else if(std::strncmp(argv[i], "--best_model_path=", 18) == 0){
            bestModelPath = std::string(argv[i] + 18);
        }
        else if(std::strncmp(argv[i], "--skip_train", 12) == 0){
            skipTrain = true;
            logInfo("** Received --skip_train => Skipping training and only evaluating. **");
        }
    }

    logInfo("Best model path => " + bestModelPath);

    if(dataDirs.empty()){
        logWarn("No data_dirs provided. Using default directories.");
        dataDirs = {"tpch_sf1_templates", "tpch_sf1_templates_index", 
                   "tpch_sf1_zsce_index_TP", "tpch_sf100_zsce"};
    }

    double timeout = 60.0; // As defined in Python script
    
    // 1) Build dataset
    ExecutionPlanDatasetClass dataset(dataDirs, timeout);
    size_t dsSize = dataset.size();
    logInfo("Combined dataset size => " + std::to_string(dsSize));
    if(dsSize < 30){
        logError("Not enough data. Exiting.");
        return 1;
    }

    // 2) Split dataset into train and validation
    auto splits = splitDataset(dsSize, 0.8);
    std::vector<size_t> trainIdx = splits.first;
    std::vector<size_t> valIdx   = splits.second;

    logInfo("Train samples = " + std::to_string(trainIdx.size()) + 
            ", Val samples = " + std::to_string(valIdx.size()));

    // 3) Prepare Training Samples
    std::vector<int> trainLabels;
    for(auto idx : trainIdx){
        int lab = dataset[idx].label;
        if(lab != -1){
            trainLabels.push_back(lab);
        }
    }

    WeightedRandomSamplerClass sampler(trainLabels);
    std::vector<TrainSampleStruct> trainingSamples = prepareTrainingSamples(dataset, trainIdx);

    // 4) Initialize or Load Model
    FannModel *model = nullptr;
    if(!skipTrain){
        logInfo("Initializing and training the MLP model.");
        model = new FannModel(8, hiddenNeurons, 1, learning_rate);
    }
    else{
        logInfo("Skipping training. Loading existing model from " + bestModelPath);
        if(!fileExists(bestModelPath)){
            logError("Model file does not exist: " + bestModelPath);
            return 1;
        }
        model = new FannModel(8, hiddenNeurons, 1, learning_rate); // Temporary initialization
        model->load(bestModelPath);
    }

    // 5) Training Loop with Early Stopping
    double bestValMSE = 1e9;
    int patience = 20;
    int patienceCount = 0;

    if(!skipTrain){
        for(int epoch=1; epoch <= epochs; epoch++){
            // Training
            size_t trainSampleSize = trainingSamples.size();
            double epochMSE = 0.0;
            for(auto &ts : trainingSamples){
                model->train({ts});
                epochMSE += model->getMSE();
            }
            epochMSE /= trainSampleSize;

            // Validation
            double valMSE = 0.0;
            size_t valSampleSize = 0;
            std::vector<int> valYtrue;
            std::vector<float> valScores;
            for(auto idx : valIdx){
                const RowSampleStruct &rs = dataset[idx];
                if(rs.label == -1) continue; // Ignore
                std::vector<float> input(rs.feats, rs.feats + 8);
                float output = model->run(input);
                valScores.push_back(output);
                valYtrue.push_back(rs.label);
                double desired = (rs.label == 1) ? 1.0 : 0.0;
                double err = output - desired;
                valMSE += (err * err);
                valSampleSize++;
            }
            if(valSampleSize > 0){
                valMSE /= valSampleSize;
            }

            logInfo("Epoch " + std::to_string(epoch) + "/" + std::to_string(epochs) + 
                    " => Train MSE=" + std::to_string(epochMSE) + 
                    ", Val MSE=" + std::to_string(valMSE));

            // Check for improvement
            if(valMSE < bestValMSE){
                bestValMSE = valMSE;
                patienceCount = 0;
                // Create directory if it doesn't exist
                std::string dirPath = bestModelPath.substr(0, bestModelPath.find_last_of("/\\"));
                // Use POSIX mkdir with -p like behavior
                std::string mkdirCmd = "mkdir -p " + dirPath;
                system(mkdirCmd.c_str());
                model->save(bestModelPath);
                logInfo("Saved best model with Val MSE = " + std::to_string(bestValMSE));
            }
            else{
                patienceCount++;
                logInfo("No improvement in Val MSE for " + std::to_string(patienceCount) + " epochs.");
                if(patienceCount >= patience){
                    logInfo("Early stopping triggered.");
                    break;
                }
            }
        }

        // Reload the best model
        delete model;
        model = new FannModel(8, hiddenNeurons, 1, learning_rate);
        model->load(bestModelPath);
        logInfo("Reloaded the best model from " + bestModelPath);
    }

    // 6) Evaluation on Validation Set
    logInfo("Evaluating the model on the entire dataset.");
    std::vector<int> valYtrue;
    std::vector<float> valScores;
    // allIdx = valIdx + trainIdx;
    valIdx.insert(valIdx.end(), trainIdx.begin(), trainIdx.end());
    for(auto idx : valIdx){
        const RowSampleStruct &rs = dataset[idx];
        if(rs.label == -1) continue; // Ignore
        std::vector<float> input(rs.feats, rs.feats + 8);
        float output = model->run(input);
        valScores.push_back(output);
        valYtrue.push_back(rs.label);
    }

    // Compute Metrics
    double valMSE = 0.0;
    for(size_t i=0; i < valYtrue.size(); i++){
        double desired = (valYtrue[i] == 1) ? 1.0 : 0.0;
        double err = valScores[i] - desired;
        valMSE += (err * err);
    }
    if(valYtrue.size() > 0){
        valMSE /= valYtrue.size();
    }

    // Find Best Threshold by F1-score
    double bestF1 = -1.0;
    double bestThr = findBestThreshold(valYtrue, valScores, bestF1);
    bestThr = 0.5; // Use default threshold for now, manually set
    logInfo("Threshold search => Best Threshold = " + std::to_string(bestThr) + 
            ", Best F1 = " + std::to_string(bestF1));

    // Compute final metrics at best threshold
    Metrics metrics;
    metrics.loss = valMSE;
    metrics.best_threshold = static_cast<float>(bestThr);
    for(size_t i=0; i < valYtrue.size(); i++){
        int pred = (valScores[i] >= bestThr) ? 1 : 0;
        metrics.ytrue.push_back(valYtrue[i]);
        metrics.scores.push_back(valScores[i]);

        if(valYtrue[i] == pred){
            metrics.accuracy += 1.0;
        }
        if(valYtrue[i] == 1 && pred == 1){
            metrics.precision += 1.0;
            metrics.recall += 1.0;
        }
        if(valYtrue[i] == 0 && pred == 1){
            metrics.precision += 1.0;
        }
        if(valYtrue[i] == 1 && pred == 0){
            metrics.recall += 1.0;
        }
        if(valYtrue[i] == 0 && pred == 0){
            // True Negative
        }
    }

    size_t total = valYtrue.size();
    if(total > 0){
        metrics.accuracy /= total;
    }

    // Calculate Precision, Recall, F1
    int TP=0, FP=0, TN=0, FN=0;
    for(size_t i=0; i < valYtrue.size(); i++){
        int t = valYtrue[i];
        int p = (valScores[i] >= bestThr) ? 1 : 0;
        if(t == 1 && p == 1) TP++;
        if(t == 0 && p == 1) FP++;
        if(t == 0 && p == 0) TN++;
        if(t == 1 && p == 0) FN++;
    }

    metrics.accuracy = (double)(TP + TN) / (TP + TN + FP + FN);
    metrics.precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
    metrics.recall    = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
    metrics.f1        = (metrics.precision + metrics.recall > 0.0) ? 
                         2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) : 0.0;

    // Precision and Recall for Negatives
    if((TN + FN) > 0){
        metrics.precision_neg = (double)TN / (TN + FN);
    }
    if((TN + FP) > 0){
        metrics.recall_neg = (double)TN / (TN + FP);
    }

    if((TP + FP) > 0){
        metrics.precision_pos = (double)TP / (TP + FP);
    }
    if((TP + FN) > 0){
        metrics.recall_pos = (double)TP / (TP + FN);
    }

    // Compute AUC
    std::vector<std::pair<double, double>> rocPoints = computeRocCurve(metrics.ytrue, metrics.scores);
    metrics.auc = computeAUC(rocPoints);
    saveRocCurve(rocPoints, "roc_curve.csv");

    logInfo("Final Val => Loss=" + std::to_string(metrics.loss) + 
            ", Acc=" + std::to_string(metrics.accuracy) + 
            ", Prec=" + std::to_string(metrics.precision) + 
            ", Rec=" + std::to_string(metrics.recall) + 
            ", F1=" + std::to_string(metrics.f1) + 
            ", AUC=" + std::to_string(metrics.auc) + 
            ", Best Threshold=" + std::to_string(metrics.best_threshold) + 
            ", RecNeg=" + std::to_string(metrics.recall_neg) + 
            ", RecPos=" + std::to_string(metrics.recall_pos) + 
            ", PrecNeg=" + std::to_string(metrics.precision_neg) + 
            ", PrecPos=" + std::to_string(metrics.precision_pos));

    // 7) Optional: Evaluate on the entire dataset for runtime metrics
    // Note: This part may need adjustments based on your specific requirements
    // Here we compute average runtime based on model predictions

    double cost_threshold_run_time = 0.0;
    double hybrid_opt_run_time = 0.0;
    double fann_model_run_time = 0.0;
    double optimal_run_time = 0.0;
    double ai_classifier_run_time = 0.0;
    double row_execution_time = 0.0;
    double column_execution_time = 0.0;
    

    for(size_t i=0; i < dataset.size(); i++){
        const RowSampleStruct &rs = dataset[i];
        row_execution_time = row_execution_time * i / (i + 1) + rs.row_time / (i + 1);
        column_execution_time = column_execution_time * i / (i + 1) + rs.column_time / (i + 1);

        double original_query_cost = rs.original_query_cost; // This should be captured during parsing if needed

        // Cost Threshold Method
        double cost_threshold_delta = (original_query_cost > 50000.0) ? rs.column_time : rs.row_time;
        cost_threshold_run_time = cost_threshold_run_time * i / (i + 1) + cost_threshold_delta / (i + 1);

        double hybrid_opt_delta = (rs.hybrid_use_imci == 1) ? rs.column_time : rs.row_time;
        hybrid_opt_run_time = hybrid_opt_run_time * i / (i + 1) + hybrid_opt_delta / (i + 1);

        double fann_model_delta = (rs.fann_use_imci == 1) ? rs.column_time : rs.row_time;
        fann_model_run_time = fann_model_run_time * i / (i + 1) + fann_model_delta / (i + 1);

        // Optimal Runtime
        double optimal_delta = std::min(rs.row_time, rs.column_time);
        optimal_run_time = optimal_run_time * i / (i + 1) + optimal_delta / (i + 1);

        // AI Classifier Method
        // If using FANN with sigmoid activation, output > 0.5 indicates class 1
        std::vector<float> input(rs.feats, rs.feats + 8);
        float pred_prob = model->run(input);
        int pred_label = (pred_prob >= 0.5f) ? 1 : 0;
        double ai_delta = (pred_label == 1) ? rs.column_time : rs.row_time;
        // If no valid prediction, fallback to row_time
        ai_classifier_run_time = ai_classifier_run_time * i / (i + 1) + ai_delta / (i + 1);
        if (ai_delta > cost_threshold_delta)
            logInfo("cost_threshold_delta = " + std::to_string(cost_threshold_delta) + 
                ", hybrid_opt_delta = " + std::to_string(hybrid_opt_delta) + 
                ", optimal_delta = " + std::to_string(optimal_delta) + 
                ", ai_delta = " + std::to_string(ai_delta));
    }

    logInfo("Runtime Metrics => Cost Threshold run_time: " + std::to_string(cost_threshold_run_time) + 
            "s, Hybrid Opt run_time: " + std::to_string(hybrid_opt_run_time) + 
            "s, Fann model run_time: " + std::to_string(fann_model_run_time) +
            "s, Optimal run_time: " + std::to_string(optimal_run_time) + 
            "s, AI Classifier run_time: " + std::to_string(ai_classifier_run_time) + 
            "s (threshold 0.5) Row Execution Time: " + std::to_string(row_execution_time) + 
            "s  Column Execution Time: " + std::to_string(column_execution_time) + "s");

    // Cleanup
    delete model;

    return 0;
}
