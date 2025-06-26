/* -----------------------------------------------------------
 *  main.cpp – driver for row-vs-column routers
 * ----------------------------------------------------------- */
#include "common.hpp"
#include "model_iface.hpp"

bool g_need_col_plans = true;


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
    /* ───── CLI / hyper-params ─────────────────────────────── */
    std::string model_type  = "lightgbm";
    std::string base        = "/home/wuy/query_costs";
    std::vector<std::string> data_dirs{
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
    hp.max_depth         = 20;
    hp.lr                = 0.06;
    hp.subsample         = 0.7;
    hp.colsample         = 0.8;
    hp.skip_train        = false;

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
        else if (a.rfind("--query_file",0)==0)     query_file = a.substr(12);
        else if (a.rfind("--database=",0)==0)      database = a.substr(11);
        else                                       logW("ignored arg: "+a);
    }

    if (!query_sql.empty() && database.empty()) {
        logE("When providing a single query, must also provide its database.");
    }

    /* ---------- single-query fast path ---------- */
    if (!query_sql.empty() && !database.empty()) {
        /* 1) 解析 SQL → Sample（你已有的工具；这里只用示例函数名） */
        Sample s = build_sample_from_sql(query_sql, g_need_col_plans);

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
        int pr = learner->predict(model_path, one, /*τ=*/0.0f).front();
        std::cout << "\n[Result] " << (pr ? "Use COLUMN store" : "Use ROW store")
                << "  (model=" << model_type << ")\n";
        return 0;
    }


    /* ╔════════════════════════════════════════════════════════════╗
       ║  Fast-path ② : --query_file=...  (CSV 批量)                ║
       ╚════════════════════════════════════════════════════════════╝ */
    if (!query_file.empty()) {
        std::vector<Sample> DS = build_samples_from_csv(query_csv, g_need_col_plans);
        if (DS.empty()) { logE("CSV has no valid rows");  return 1; }

        std::string model_path = model_path_cli.empty()
                               ? "checkpoints/" + model_type + "_best.txt"
                               : model_path_cli;
        if (!file_exists(model_path)) {
            logE("model file not found: " + model_path);  return 1;
        }

        auto pred = learner->predict(model_path, DS);
        report_metrics(pred, DS);
        return 0;
    }

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

    /* choose 3 random dirs as hold-out test (oracle  ≈ 20 %)  */
    std::vector<std::string> test_dirs = pick_test3(data_dirs, seed);

    std::cout << "\n[Split] TEST dirs : " << join(test_dirs) << '\n';

    std::vector<std::string> cv_pool;
    for (const auto& d : data_dirs)
        if (!std::count(test_dirs.begin(), test_dirs.end(), d))
            cv_pool.push_back(d);

    std::cout << "[Split] CV pool   : " << join(cv_pool) << '\n';

    /* ───── 5-fold split on remaining pool ─────────────────── */
    auto folds   = make_cv5(cv_pool, seed);
    std::vector<std::pair<double,std::string>> fold_scores;   // (balAcc, modelPath)

    int fid = 0;

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

    for (auto& f : folds) {
        ++fid;

        std::cout << "\n[Fold " << fid << "]"
            << "  Train dirs = {" << join(f.tr_dirs)  << "}"
            << "  Val dirs = {"   << join(f.val_dirs) << "}\n";


        std::string mp = ckpt_dir + '/' + model_type + "_fold" + std::to_string(fid) + ".txt";

        auto DS_tr = build_subset(f.tr_dirs , ALL);
        auto DS_va = build_subset(f.val_dirs, ALL);
        if (DS_tr.empty() || DS_va.empty()) { logW("fold" + std::to_string(fid) + "skipped"); continue; }

        /* ---- train or just load ---- */
        if (!hp.skip_train)
            learner->train(DS_tr, DS_va, mp, hp);

        double bal = learner->bal_acc(mp, DS_va);   // your convenience helper
        std::cout << "[Fold "<<fid<<"] BalAcc="<<bal<<'\n';
        fold_scores.push_back({bal, mp});
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
