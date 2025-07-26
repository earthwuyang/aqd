/* -----------------------------------------------------------
 *  main.cpp – driver for row-vs-column routers
 * ----------------------------------------------------------- */
#include "common.hpp"
#include <iostream>
#include <sys/stat.h>  // 声明 ::mkdir
#include <cerrno>      // errno 与 EEXIST
#include <cmath>
#include <unordered_map>

#include "model_iface.hpp"
#include "global_stats.hpp"

bool g_need_col_plans = true;
bool g_use_col_feat = false;
bool g_use_vib = false;

/* factory fns declared somewhere in your project ---------------- */
std::unique_ptr<IModel> make_lightgbm(const std::string& booster);
std::unique_ptr<IModel> make_fann();
std::unique_ptr<IModel> make_dtree();
std::unique_ptr<IModel> make_gin();   // new

static std::string join(const std::vector<std::string>& v,
                        const std::string& sep = ", ")
{
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        out += v[i];
        if (i + 1 < v.size()) out += sep;
    }
    return out;
}

inline void split_train_test(const std::vector<Sample>& all,
                             double              test_ratio,
                             uint32_t            seed,
                             std::vector<Sample>& train_out,
                             std::vector<Sample>& test_out)
{
    if (all.empty()) return;

    std::vector<Sample> tmp = all;           // 复制再打乱
    std::mt19937 rng(seed);
    std::shuffle(tmp.begin(), tmp.end(), rng);

    size_t n_test = std::max<size_t>(1, std::lround(tmp.size() * test_ratio));
    size_t split  = tmp.size() - n_test;

    train_out.assign(tmp.begin(),           tmp.begin() + split);
    test_out .assign(tmp.begin() + split,   tmp.end());
}

static std::vector<std::vector<std::string>>
make_5x3_partition(std::vector<std::string> dirs, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::shuffle(dirs.begin(), dirs.end(), rng);

    std::vector<std::vector<std::string>> groups(5);
    for (size_t i = 0; i < dirs.size(); ++i)
        groups[i / 3].push_back(dirs[i]);   // 0-based
    return groups;   // 5 groups, each of size 3
}



/* ----------------------------------------------------------- */
/*  main.cpp – driver for row-vs-column routers (ONE-VAL split) */
/* ----------------------------------------------------------- */
int main(int argc, char* argv[])
{
    /* ========== 1. CLI and defaults ========================= */
    std::string query_sql, query_file, database;
    std::string sql_host="127.0.0.1", sql_user="root", sql_pass="";
    int         sql_port=44444;

    std::string model_type="lightgbm";
    std::string base="/home/wuy/query_costs_trace";
    std::vector<std::string> data_dirs{
        "tpch_sf1","tpch_sf10","tpch_sf100",
        "tpcds_sf1","tpcds_sf10","tpcds_sf100",
        "hybench_sf1","hybench_sf10",
        "airline","credit","carcinogenesis","employee",
        "financial","geneea","hepatitis"
    };

    TrainOpt hp; hp.trees=400; hp.max_depth=4; hp.lr=0.06;
    hp.subsample=0.7; hp.colsample=0.8; hp.skip_train=false; hp.vib=false;

    bool g_mix_folds=false, run_all=false;
    int  fold_id=0;            /* 1…5 ; 0 = default first split   */
    uint32_t seed=42;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--all")         run_all = true;
        else if (a.rfind("--fold=",0)==0) fold_id = std::stoi(a.substr(7)); // 1-5
        else if      (a.rfind("--data_dirs=",0)==0) {
            data_dirs.clear();
            std::string s = a.substr(12), t;
            std::stringstream ss(s);
            while (std::getline(ss, t, ',')) if(!t.empty()) data_dirs.push_back(t);
        }
        else if (a.rfind("--base=",0)==0)          base         = a.substr(7);
        else if (a.rfind("--seed=",0)==0)          seed         = std::stoul(a.substr(7));
        else if (a.rfind("--model=",0)==0)         model_type   = a.substr(8);
        else if (a.rfind("--trees=",0)==0)         hp.trees     = std::stoi(a.substr(8));
        else if (a.rfind("--max_depth=",0)==0)     hp.max_depth = std::stoi(a.substr(12));
        else if (a.rfind("--lr=",0)==0)            hp.lr        = std::stod(a.substr(5));
        else if (a.rfind("--subsample=",0)==0)     hp.subsample = std::stod(a.substr(12));
        else if (a.rfind("--colsample=",0)==0)     hp.colsample = std::stod(a.substr(12));
        else if (a == "--skip_train")              hp.skip_train = true;
        else if (a.rfind("--query=",0)==0)         query_sql = a.substr(8);
        else if (a.rfind("--query_file=",0)==0)     query_file = a.substr(13);
        else if (a.rfind("--database=",0)==0)      database = a.substr(11);
        else if (a == "--fold_id")                  fold_id = std::stoi(a.substr(9));
        else if (a == "--use_col")               g_use_col_feat = true;
        else if (a == "--vib")                 { hp.vib = true; g_use_vib = true; }
        else if (a == "--shap")             hp.shap = true;
        else                                       logW("ignored arg: "+a);
    }
    if (run_all && fold_id){ logE("use --all OR --fold, not both"); return 1; }
    if (fold_id<0||fold_id>5){ logE("--fold must be 1…5"); return 1; }

    /* ========== 2. Instantiate learner ====================== */
    std::unique_ptr<IModel> learner;
    if      (model_type=="lightgbm") learner=make_lightgbm("goss");
    else if (model_type=="rowmlp")   learner=make_fann();
    else if (model_type=="dtree")    learner=make_dtree();
    else if (model_type=="forest")   learner=make_lightgbm("rf");
    else if (model_type=="gin")      learner=make_gin();
    else { logE("unknown --model="+model_type); return 1; }

    g_need_col_plans = (model_type=="gin");

    /* ========== 3. One-query / CSV fast paths (unchanged) ==== */
    /* ╔═══════════════╗
       ║  ONE-TIME DB  ║  — column / table / index meta
       ╚═══════════════╝ */
    {
        /* data_dirs 里的目录名就是 MySQL schema 名；若不是，请在这里替换 */
        std::vector<std::string> dbs = data_dirs;

        // if (!populate_col_stats(sql_host, sql_port, sql_user, sql_pass, dbs))
        //     logW("populate_col_stats() failed – column-level features will degrade");

        // if (!populate_tbl_stats(sql_host, sql_port, sql_user, sql_pass, dbs))
        //     logW("populate_tbl_stats() failed – table-level features will degrade");

        // /* (可选) 预加载所有索引列全集 — 只影响 covering-index 相关特征 */
        // load_all_index_defs(sql_host, sql_port, sql_user, sql_pass, dbs);
        collect_db_sizes(sql_host, sql_port, sql_user, sql_pass, data_dirs);
    }

    if (!query_sql.empty() && database.empty()) {
        logE("When providing a single query, must also provide its database.");
    }

    /* ---------- single-query fast path ---------- */
    if (!query_sql.empty() && !database.empty()) {
        /* 1) 解析 SQL → Sample（你已有的工具；这里只用示例函数名） */
        Sample s = build_sample_from_sql(query_sql, database, g_need_col_plans);

        /* 2) 模型文件路径：默认 checkpoints/<model_type>_best.txt  */
        std::string model_path = "checkpoints/" + model_type + "_best.txt";
        // 允许 --model_path=... 覆盖
        for (int i = 1; i < argc; ++i) {
            std::string a(argv[i]);
            if (a.rfind("--model_path=",0)==0)  model_path = a.substr(13);
        }
        if (!file_exists(model_path)) {
            logE("model file not found: " + model_path);
            return 1;
        }

        /* 3) 预测并输出结果 */
        std::vector<Sample> one{ s };
        for (int j=0;j<NUM_FEATS;j++)   
            printf("feat[%d]=%f\n",j,s.feat[j]);
        int pr = learner->predict(model_path, one, /*τ=*/0.0f).front();
        std::cout << "\n[Result] " << (pr ? "Use COLUMN store" : "Use ROW store")
                << "  (model=" << model_type << ")\n";
        return 0;
    }


    /* ╔════════════════════════════════════════════════════════════╗
       ║  Fast-path ② : --query_file=...  (CSV 批量)                ║
       ╚════════════════════════════════════════════════════════════╝ */
    if (!query_file.empty()) {
        std::vector<Sample> DS = build_samples_from_csv(query_file, g_need_col_plans);
        if (DS.empty()) { logE("CSV has no valid rows");  return 1; }

        std::string model_path = "checkpoints/" + model_type + "_best.txt";
        if (!file_exists(model_path)) {
            logE("model file not found: " + model_path);  return 1;
        }

        auto pred = learner->predict(model_path, DS);
        report_metrics(pred, DS);
        return 0;
    }

    /* ========== 4. Load all datasets ======================== */
    DirSamples ALL = load_all_datasets(base, data_dirs);
    if (ALL.empty()){ logE("no samples found"); return 1; }

    /* dataset weight √(N_total / N_i) */
    std::unordered_map<std::string,double> DIR_W;
    double tot=0; for(auto&kv:ALL) tot+=kv.second.size();
    for(auto&kv:ALL) DIR_W[kv.first]=std::pow(tot/kv.second.size(),0.2);
    global_stats().freeze();

    /* ========== 5. Build fixed 5×3 test groups ============== */
    static auto groups = make_5x3_partition(data_dirs, seed);

    /* ========== 6. Helper to run ONE split ================== */
    auto run_one_split =
        [&](size_t idx, const std::vector<std::string>& test_dirs)->int
    {
        std::cout<<"\n========== Split "<<idx+1<<" ==========\n";
        std::cout<<"TEST dirs : "<<join(test_dirs)<<"\n";

        /* ---- pick 1 validation dir randomly from the 12 left ---- */
        std::vector<std::string> pool;
        for(auto&d:data_dirs)
            if(!std::count(test_dirs.begin(),test_dirs.end(),d))
                pool.push_back(d);

        std::shuffle(pool.begin(), pool.end(), std::mt19937(seed+idx));
        std::string              val_dir   = pool.front();
        std::vector<std::string> train_dirs(pool.begin()+1,pool.end());

        std::cout<<"VAL  dir  : "<<val_dir<<"\n";
        std::cout<<"TRAIN dirs: "<<join(train_dirs)<<"\n";

        /* ---- build sample sets ---------------------------------- */
        auto DS_tr = build_subset(train_dirs, ALL);
        auto DS_va = build_subset({val_dir}, ALL);
        auto DS_te = build_subset(test_dirs,  ALL);
        if(DS_tr.empty()||DS_va.empty()||DS_te.empty()){
            logE("empty split"); return 1;
        }

        /* ---- checkpoint path ------------------------------------ */
        const std::string ckpt_dir="checkpoints";
        if(!is_directory(ckpt_dir) &&
           ::mkdir(ckpt_dir.c_str(),0755)!=0 && errno!=EEXIST){
            logE("mkdir failed"); return 1;
        }
        std::string mp=ckpt_dir+'/'+model_type+"_split"+std::to_string(idx+1)+".txt";

        /* ---- train / validate ----------------------------------- */
        if(!hp.skip_train){
            learner->train(DS_tr, DS_va, mp, hp, DIR_W);
            auto pv=learner->predict(mp, DS_va);
            report_metrics(pv, DS_va);
        }

        /* ---- test ----------------------------------------------- */
        auto pt=learner->predict(mp, DS_te);
        report_metrics(pt, DS_te);
        return 0;
    };

    /* ========== 7. Run requested split(s) =================== */
    if(run_all){
        for(size_t i=0;i<groups.size();++i)
            if(run_one_split(i, groups[i])) return 1;
    }else{
        size_t idx=fold_id?fold_id-1:0;
        if(run_one_split(idx, groups[idx])) return 1;
    }
    return 0;
}
