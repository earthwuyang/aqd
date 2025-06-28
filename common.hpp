/*───────────────────────────────────────────────────────────
 *  common.hpp   –  declarations-only
 *───────────────────────────────────────────────────────────*/
#pragma once

/* ---------- 基础 & STL ---------- */
#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>      // pair
#include <cstdint>      // uint8_t
#include <random>     // ← 新增
#include <unordered_set>
#include <fstream>

/* ---------- 依赖库 ---------- */
#include <Eigen/Dense>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

/* ---------- 全局常量 ---------- */
constexpr int    ORIG_FEATS   = 115;      // feature dim
constexpr double EPS_RUNTIME = 1e-2;
constexpr double COST_THR    = 5e4;

constexpr int EMB_DIM = 0;
constexpr int NUM_FEATS = ORIG_FEATS + EMB_DIM;

/* ---------- 全局开关 ---------- */
extern bool g_need_col_plans;            // 默认 = true
extern bool g_use_col_feat;          // 默认为 false，由 CLI 设置

/* ────────────────── Column / Table 元数据 ────────────────── */
enum ColDType : uint8_t { COL_INT=0, COL_FLOAT, COL_STRING,
                          COL_DATETIME, COL_BOOL, COL_DTYPE_N };



/* ────────────────── 基础数据结构 ────────────────── */
struct Graph {
    int N = 0;
    std::vector<Eigen::Vector2d>       x;      // (rows , cost) – z-scored
    std::vector<std::vector<int>>      adj;    // adjacency list
    std::vector<int>                   op;     // global op-id / node
};

// std::vector<float> gnn_encoder(const Graph& g);    // 长度 = EMB_DIM

struct Sample {
    std::array<float, NUM_FEATS> feat {};
    Graph    colGraph;
    int      label       = 0;
    double   row_t       = 0, col_t = 0, qcost = 0;
    int      fann_pred   = -1;
    int      hybrid_pred = -1;
    std::string dir_tag;                       // 来源数据集目录
};

struct TblStats {
    double rows        = 0;    // INFORMATION_SCHEMA.TABLES.TABLE_ROWS
    double data_bytes  = 0;    // DATA_LENGTH
    double idx_bytes   = 0;    // INDEX_LENGTH
    double frag_ratio  = 0;    // DATA_FREE / (DATA_LENGTH+1)
    double upd_pct     = 0;    // rows_updated / (rows_read+1)   (PS)
    int    partitions  = 1;    // # partitions (≥1)
    int    idx_cnt     = 0;    // total indexes
    int    uniq_cnt    = 0;    // UNIQUE indexes
    int    pk_len      = 0;    // ⅀ PK column byte width
    int    cover_cols  = 0;    // # columns covered by any index
    int    total_cols  = 0;    // total columns
    bool   compressed  = false;
};

struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0,selSum=0,selMin=1e30,selMax=0,
           ratioSum=0,ratioMax=0,maxPrefix=0,minRead=1e30,fanoutMax=0;
    int    cnt=0,cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,sumPK=0,
           coverCount=0,maxDepth=0;
    bool   grp=false,ord=false,tmp=false;
    double outerRows      = 0;   // first non-ALL driver cardinality
    int    eqChainDepth   = 0;   // longest consecutive eq_ref chain
    int    _curEqChain    = 0;   // internal: running counter
    double lateFanMax  = 0;   // ❶ NEW  – max fan-out seen at depth ≥ 4
    double pcDepth3 = 0;        // Σ prefix_cost of tables at depth == 3
    bool  hashJoin = false;          // NEW – any table used hash join‐buffer
};

struct ColStats {
    double   avg_width = 8;        // bytes
    double   ndv       = 1000;     // distinct values
    ColDType dtype     = COL_INT;
    double rows_c = 0, rows_s = 1, cost_c = 0, cost_s = 1;
    std::unordered_map<std::string,int> op2id;  // local operator IDs
};

struct Split {
    vector<string> train_dirs;   // 用于 5-fold CV
    vector<string> test_dirs;    // 独立 hold-out
};

/* ───── cv5.h ───── */
struct Fold {
    vector<string> tr_dirs;   // train 子集
    vector<string> val_dirs;  // val  子集
};



struct DTNode { int feat=-1,left=-1,right=-1; float thr=0,prob=0; };


/* ---------- CART decision-tree ------------------------------------ */
class DecisionTree
{
public:
    /* ctor: md = max depth, ms = min samples per leaf, mg = min gain (预留) */
    DecisionTree(int md = 16, int ms = 4, double mg = 0.0);

    /* 拟合二分类树（支持 sample-权重） */
    void fit(const std::vector<std::array<float,NUM_FEATS>>& X,
             const std::vector<int>&                        y,
             const std::vector<float>&                      w);

    /* 预测某一行特征概率（返回 [0,1]） */
    float predict(const float* feat) const;

    /* 序列化 / 反序列化 */
    nlohmann::json to_json() const;
    void           from_json(const nlohmann::json& arr);

private:
    /* —— 帮助函数（实现见 .cpp） ——————————————— */
    static double gini(double pos, double tot);
    int  build(const std::vector<int>&                        idx,
               const std::vector<std::array<float,NUM_FEATS>>& X,
               const std::vector<int>&                        y,
               const std::vector<float>&                      w,
               int depth, int max_depth, int min_samples);

    /* —— 数据成员 ——————————————————————————————— */
    std::vector<DTNode> nodes_;   // 扁平存储的节点数组
    double              min_gain_;
    int                 max_depth_;
    int                 min_samples_;
};


/* ────────────────── log / 进度 & 文件工具 ────────────────── */
void logI(const std::string& msg);
void logW(const std::string& msg);
void logE(const std::string& msg);

bool        file_exists   (const std::string& path);
bool        is_directory  (const std::string& path);
bool        has_ext       (const std::string& filename,
                           const std::string& ext);
std::string strip_ext     (const std::string& filename);

void progress(const std::string& tag,
              size_t cur, size_t tot, size_t barWidth = 40);

/* ────────────────── JSON / 数值小工具 ────────────────── */
double parsePossibleNumber(const nlohmann::json& j,
                           const std::string& key);
double convert_data_size_to_numeric(std::string s);
inline double safeLog1p(double v);                // 实现可继续放 inline
double safe_f(const nlohmann::json& v);
double safe_f(const nlohmann::json& obj, const char* key);
double log1p_clip(double v);
double str_size_to_num(std::string s);
bool   getBool(const nlohmann::json& obj, const char* key);





/* 数据统计加载 / 查询 */
bool populate_col_stats(const std::string& host, int port,
                        const std::string& user, const std::string& pass,
                        const std::vector<std::string>& dbs);

bool populate_tbl_stats(const std::string& host, int port,
                        const std::string& user, const std::string& pass,
                        const std::vector<std::string>& dbs);

ColStats             lookup_col_stats(const std::string& id);
const TblStats&      lookup_tbl      (const std::string& id);




/* ────────────────── 计划解析 / 样本构建 ────────────────── */
bool plan2feat(const nlohmann::json& plan, float f[NUM_FEATS]);

Graph parseColPlan(const nlohmann::json& planJson,
                   const ColStats& st,
                   const std::unordered_map<std::string,int>& globalOp2id);

/* 对列计划目录做一次扫描，结果缓存到 STATS_CACHE */
ColStats buildColStats(const std::string& datasetDir);

/* 获取（并缓存）某目录的列统计 */
ColStats& get_col_stats_for_dir(const std::string& dir);




/* 2. 全局 operator → id  map  ─────────────────────────────── */
extern std::unordered_map<std::string,int> GLOBAL_OP2ID;

inline std::unordered_map<std::string,int>& global_op2id()
{
    return GLOBAL_OP2ID;
}

/* 主入口：加载行/列计划文件，抽特征 & 图 */
bool load_plan_file(const std::string& fp_row, const std::string& qid,
                    const std::string& rowDir, const std::string& colDir,
                    float feat[NUM_FEATS], double& qcost, Graph& colGraph,
                    bool need_col = g_need_col_plans);

void load_all_index_defs(const string &host,int port,
                                const string &user,const string &pass,
                                const vector<string>& dbs);

/* ────────────────── 数据集拆分 / 评估 ────────────────── */
struct Split;



std::vector<Fold> make_lodo(const std::vector<std::string>& dirs);

std::vector<Fold> make_cv3(const std::vector<std::string>& dirs);


/* 批量加载整个目录的数据集 */
using DirSamples = std::unordered_map<std::string,
                                      std::vector<Sample>>;
DirSamples load_all_datasets(const std::string& base,
                             const std::vector<std::string>& dirs);

std::vector<Sample> build_subset(const std::vector<std::string>& dirs,
                                 const DirSamples& all);

std::vector<std::string> pick_test3(std::vector<std::string> dirs,
                                    uint32_t seed = 42);

/* 输出各策略评估结果 */
void report_metrics(const std::vector<int>& pred_lgb,
                    const std::vector<Sample>& testSet);

/* 即时从 SQL 生成单条样本 */
Sample build_sample_from_sql(const std::string& sql, const std::string& db,
                             bool need_col_plan = g_need_col_plans,
                             const std::string& host = "127.0.0.1",
                             int port = 44444,
                             const std::string& user = "root",
                             const std::string& pass = "");

/* 从官方 CSV 批量生成样本 */
std::vector<Sample> build_samples_from_csv(const std::string& csv_path,
        bool need_col_plan = g_need_col_plans,
        const std::string& host = "127.0.0.1", int port = 44444,
        const std::string& user = "root", const std::string& pass = "");

inline std::unordered_set<int>
load_feat_blk(const std::string& model_path) {
    std::unordered_set<int> blk;
    std::ifstream in(model_path + ".blk");
    int id;
    while (in >> id) blk.insert(id);
    return blk;
}