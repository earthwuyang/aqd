/* -----------------------------------------------------------
 *  main.cpp – driver for row-vs-column routers
 * ----------------------------------------------------------- */
#include "common.hpp"
#include <iostream>
#include <sys/stat.h>  // 声明 ::mkdir
#include <cerrno>      // errno 与 EEXIST

#include "model_iface.hpp"
#include "global_stats.hpp"

bool g_need_col_plans = true;
bool g_use_col_feat = false;


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

/* ----------------------------------------------------------- */
int main(int argc, char* argv[])
{
    std::string query_sql;
    std::string query_file;
    std::string database;

    /* ───── MySQL connection params (optional CLI overrides) ───── */
    std::string sql_host = "127.0.0.1";
    int         sql_port = 44444;
    std::string sql_user = "root";
    std::string sql_pass = "";

    /* ───── CLI / hyper-params ─────────────────────────────── */
    std::string model_type  = "forest";
    std::string base        = "/home/wuy/query_costs";
    std::vector<std::string> data_dirs{
        "tpch_sf100",
        "tpcds_sf10",
        "tpch_sf1",
        "tpcds_sf1",
        "hybench_sf1",
        "hybench_sf10",
        "airline",
        "credit",
        "carcinogenesis",
        "employee",
        "financial",
        "geneea",
        "hepatitis"
    };

    TrainOpt hp;                 // generic hyper-param struct you defined
    uint32_t seed        = 42;
    hp.trees             = 800;  // sensible defaults
    hp.max_depth         = 12;
    hp.lr                = 0.06;
    hp.subsample         = 0.7;
    hp.colsample         = 0.8;
    hp.skip_train        = false;
    bool mix_folds = false;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);

        if      (a.rfind("--data_dirs=",0)==0) {
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
        else if (a == "--mix")                  mix_folds = true;
        else if (a == "--use_col")               g_use_col_feat - true;
        else                                       logW("ignored arg: "+a);
    }

    if (g_use_col_feat)
        g_need_col_plans - true;
    
    if (data_dirs.empty()) {
        logE("No --data_dirs given");  return 1;
    }

    /* ───── instantiate learner ────────────────────────────── */
    std::unique_ptr<IModel> learner;
    if      (model_type == "lightgbm") learner = make_lightgbm("goss");
    else if (model_type == "rowmlp")   learner = make_fann();
    else if (model_type == "dtree")    learner = make_dtree();
    else if (model_type == "forest")   learner = make_lightgbm("rf");
    else if (model_type == "gin")      learner = make_gin();
    else { logE("unknown --model="+model_type); return 1; }

    g_need_col_plans = (model_type == "gin");   // 只有 GIN 需要列计划

    /* ╔═══════════════╗
       ║  ONE-TIME DB  ║  — column / table / index meta
       ╚═══════════════╝ */
    {
        /* data_dirs 里的目录名就是 MySQL schema 名；若不是，请在这里替换 */
        std::vector<std::string> dbs = data_dirs;

        if (!populate_col_stats(sql_host, sql_port, sql_user, sql_pass, dbs))
            logW("populate_col_stats() failed – column-level features will degrade");

        if (!populate_tbl_stats(sql_host, sql_port, sql_user, sql_pass, dbs))
            logW("populate_tbl_stats() failed – table-level features will degrade");

        /* (可选) 预加载所有索引列全集 — 只影响 covering-index 相关特征 */
        load_all_index_defs(sql_host, sql_port, sql_user, sql_pass, dbs);
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


    std::unordered_map<std::string,double> DIR_W;
    /* ---------- fast-path: only inference when --skip_train ---------- */
    if (hp.skip_train) {
        /* 1) 加载指定数据集 */
        DirSamples ALL = load_all_datasets(base, data_dirs);
        if (ALL.empty()) { logE("no samples found"); return 1; }
        

        auto DS_test = build_subset(data_dirs, ALL);
        if (DS_test.empty()) { logE("no test samples"); return 1; }

        /* 2) 决定要加载的模型文件 */
        std::string model_path;                       // ← 新局部变量

        /* 2-a) 解析 CLI 中的 --model_path=… （可选） */
        for (int i = 1; i < argc; ++i) {
            std::string a(argv[i]);
            if (a.rfind("--model_path=", 0) == 0) {
                model_path = a.substr(13);            // 13 = strlen("--model_path=")
                break;
            }
        }

        /* 2-b) 若 CLI 未指定，则回退到 checkpoints/<model_type>_best.txt */
        if (model_path.empty())
            model_path = "checkpoints/" + model_type + "_best.txt";

        if (!file_exists(model_path)) {
            logE("model file not found: " + model_path);
            return 1;
        }

        /* 3) 预测并报告指标 */
        auto pred = learner->predict(model_path, DS_test, /*τ=*/0.0f);
        report_metrics(pred, DS_test);
        return 0;                  // 直接结束，不再进入训练/交叉验证
    }


    /* ───── one-shot disk scan (reuse across folds) ────────── */
    DirSamples ALL = load_all_datasets(base, data_dirs);
    if (ALL.empty()) { logE("no samples found"); return 1; }

    /* ---------- after ALL is built ---------- */
        
    double total_samples = 0.0;
    for (auto& kv : ALL) total_samples += kv.second.size();
    for (auto& kv : ALL) {
        double n = kv.second.size();
        DIR_W[kv.first] = std::sqrt(total_samples / n);   // √反比
    }
    global_stats().freeze();

    /* choose 3 random dirs as hold-out test (oracle  ≈ 20 %)  */
    std::vector<std::string> test_dirs = pick_test3(data_dirs, seed);

    std::cout << "\n[Split] TEST dirs : " << join(test_dirs) << '\n';

    std::vector<std::string> cv_pool;
    for (const auto& d : data_dirs)
        if (!std::count(test_dirs.begin(), test_dirs.end(), d))
            cv_pool.push_back(d);

    std::cout << "[Split] CV pool   : " << join(cv_pool) << '\n';

    /* ───── 5-fold split on remaining pool ─────────────────── */
    struct SampFold {                  // 新结构：直接存样本指针
        std::vector<const Sample*> tr;
        std::vector<const Sample*> va;
    };

    std::vector<SampFold> folds;
    std::vector<Fold>  dirFolds;

    if (mix_folds) {
        /* ----------  混合打乱 ---------- */
        std::vector<const Sample*> all_ptrs;
        for (auto& kv : ALL)                // ALL = DirSamples
            for (auto& s : kv.second) all_ptrs.push_back(&s);

        std::mt19937 g(seed);
        std::shuffle(all_ptrs.begin(), all_ptrs.end(), g);

        const int K = 5;
        const size_t fold_sz = all_ptrs.size() / K;
        folds.resize(K);

        for (int k = 0; k < K; ++k) {
            size_t beg = k * fold_sz;
            size_t end = (k == K - 1) ? all_ptrs.size() : beg + fold_sz;
            auto& f = folds[k];
            for (size_t i = 0; i < all_ptrs.size(); ++i)
                (i >= beg && i < end ? f.va : f.tr).push_back(all_ptrs[i]);
        }
    } else {
        /* ----------  旧的 LODO  ---------- */
        dirFolds = make_lodo(cv_pool);          // 还是按目录
        // dirFolds = make_cv3(cv_pool);          // 还是按目录
        for (auto& df : dirFolds) {
            SampFold sf;
            for (const auto& d : df.tr_dirs) {
                auto it = ALL.find(d);
                if (it != ALL.end())
                    for (auto& s : it->second) sf.tr.push_back(&s);
            }
            for (const auto& d : df.val_dirs) {
                auto it = ALL.find(d);
                if (it != ALL.end())
                    for (auto& s : it->second) sf.va.push_back(&s);
            }
            folds.push_back(std::move(sf));
        }
    }
    std::vector<std::pair<double,std::string>> fold_scores;   // (balAcc, modelPath)

    const std::string ckpt_dir = "checkpoints";

    if (!is_directory(ckpt_dir)) {              // ← 用你的 helper
        // 若已存在同名文件而非目录，给出友好错误
        if (file_exists(ckpt_dir)) {
            logE("path '" + ckpt_dir + "' exists but is not a directory");
            return 1;
        }
        if (::mkdir(ckpt_dir.c_str(), 0755) != 0 && errno != EEXIST) {
            logE("cannot create '" + ckpt_dir + "': " + std::strerror(errno));
            return 1;
        }
        logI("created directory '" + ckpt_dir + "'");
    }

    for (size_t idx=0; idx<folds.size(); ++idx) {
        const auto&f = folds[idx];
        const int fid = static_cast<int>(idx) + 1;

        if (mix_folds) {
            std::cout << "\n[Fold " << fid << "]"
                        << "  Train samples = " << f.tr.size()
                        << "  Val samples = "   << f.va.size() << '\n';
        } else {
            const auto& df = dirFolds[idx];        // 与 folds 下标一致
            std::cout << "\n[Fold " << fid << "] --------------------------------------------------------\n"
                    << "Train dirs = {" << join(df.tr_dirs) << "}"
                    << "  Val dirs = {"   << join(df.val_dirs) << "}\n";
        }


        std::string mp = ckpt_dir + '/' + model_type + "_fold" + std::to_string(fid) + ".txt";

        std::vector<Sample> DS_tr, DS_va;
        for (auto* p : f.tr) DS_tr.push_back(*p);
        for (auto* p : f.va) DS_va.push_back(*p);
        if (DS_tr.empty() || DS_va.empty()) { logW("fold" + std::to_string(fid) + "skipped"); continue; }

        /* ---- train or just load ---- */
        if (!hp.skip_train) {
            learner->train(DS_tr, DS_va, mp, hp, DIR_W);
            logI("Validate on validation set:\n");
            /* ─── 1. 在验证集上输出完整评估 ─── */
            {
                auto pred_va = learner->predict(mp, DS_va, /*τ=*/0.0f);
                report_metrics(pred_va, DS_va);           // 打印全部指标
            }

            /* ─── 2. 仍用 BalAcc 做折内排名 ─── */
            double bal = learner->bal_acc(mp, DS_va);
            std::cout << "[Fold " << fid << "] BalAcc(on validation set)="
                    << bal << '\n';
            fold_scores.push_back({bal, mp});
        }
    }

    if (fold_scores.size() < 3) { logE("need ≥3 good folds"); return 1; }

    /* ───── keep top-50 % folds for ensemble ───────────────── */
    std::sort(fold_scores.rbegin(), fold_scores.rend());
    std::vector<std::string> ens_models;
    for (size_t i = 0; i < (fold_scores.size()+1)/2; ++i)
        ens_models.push_back(fold_scores[i].second);

    std::cerr << "\n[INFO] ensemble uses "<<ens_models.size()<<" model(s):\n";
    for (auto& m: ens_models) std::cerr<<"  "<<m<<'\n';

    /* ───── predict on test dirs ───────────────────────────── */
    auto DS_test = build_subset(test_dirs, ALL);
    if (DS_test.empty()) { logE("no test samples"); return 1; }

    std::vector<int> vote(DS_test.size(), 0);
    for (const auto& m : ens_models) {
        auto p = learner->predict(m, DS_test, /*τ=*/0.0f);
        for (size_t i = 0; i < p.size(); ++i) vote[i] += p[i];
    }
    const int maj = int(ens_models.size())/2 + 1;
    std::vector<int> final(DS_test.size());
    for (size_t i = 0; i < final.size(); ++i) final[i] = vote[i] >= maj;

    /* ---------- 复制最佳折模型为全局 best ---------- */
    if (!hp.skip_train && !fold_scores.empty()) {
        const std::string best_src = fold_scores.front().second;          // 已按 balAcc 降序排好
        const std::string best_dst = "checkpoints/" + model_type + "_best.txt";

        try {
            #if __cpp_lib_filesystem >= 201703
            std::filesystem::copy_file(   // C++17
                best_src, best_dst,
                std::filesystem::copy_options::overwrite_existing);
            #else
            std::string cmd = "cp -f " + best_src + " " + best_dst;
            if (std::system(cmd.c_str()) != 0)
                throw std::runtime_error("cp failed");
            #endif
            logI("saved best model → " + best_dst);
        } catch (const std::exception& e) {
            logW(std::string("cannot save best model: ") + e.what());
        }
    }

    /* ───── metrics ────────────────────────────────────────── */
    report_metrics(final, DS_test);
    return 0;
}
