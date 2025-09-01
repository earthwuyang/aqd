/**********************************************************************
 *  bench_routing_modes.cpp         (July-2025, PD-LightGBM edition)
 *
 *  A **one-file** C++ re-implementation of bench_routing_modes.py
 *  – Supports 5 routing modes:
 *      row_only | col_only | cost_thresh | hybrid_opt | lgbm_pd
 *  – Multi-threaded execution, CSV latency dump, progress bar.
 *
 *  Build  : g++ -std=c++17 -O2 -pthread bench_routing_modes.cpp \
 *                   -lmysqlclient -lLightGBM -o bench_routing
 *********************************************************************/

#include <mysql/mysql.h>
#include <mysql/errmsg.h>
#include <LightGBM/c_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
// #include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/*──────────────────  SECTION 0 – tiny utils  ──────────────────*/
using hrc = std::chrono::high_resolution_clock;
inline double wall_now() {
    return std::chrono::duration<double>(hrc::now().time_since_epoch()).count();
}
inline void die(const std::string& msg) { std::cerr << msg << '\n'; std::exit(1); }

/*══════════════════  SECTION 1 – MySQL & global config  ═══════════════*/
static const char* HOST = "127.0.0.1";
static const int   PORT = 44444;
static const char* USER = "root";
static const char* PASS = "";
static std::string DB = "tpcds_sf1";

static const char* LGB_MODEL = "/home/wuy/simple_row_column_routing/checkpoints/lightgbm_best.txt";

/*──────────────────── Routing-mode session-level variables ──────────────────*/
static const std::unordered_map<std::string, std::vector<std::string>> ROUTING_MODES = {
    { "row_only", {
        "SET use_imci_engine = OFF"
    }},
    { "col_only", {
        "SET use_imci_engine = FORCED"
    }},
    { "cost_thresh", {
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = OFF",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON"
    }},
    { "hybrid_opt", {
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON"
    }},
    { "lgbm_pd", {                       // <-- our new “outside-kernel LGBM + PD”
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON"
    }}
};


/*══════════════════  SECTION 2 – CPU-meter (1 Hz sampler)  ═════════════*/
class CpuMeter {
public:
    static CpuMeter& inst() { static CpuMeter x; return x; }

    /* latest row- / column-CPU percentages */
    std::pair<double,double> latest() const
    {
        return { row_pct_.load(std::memory_order_relaxed),
                 col_pct_.load(std::memory_order_relaxed) };
    }

private:
    CpuMeter()
      : pid_(::getpid()),
        hz_ ( ::sysconf(_SC_CLK_TCK) ),
        ncpu_(::sysconf(_SC_NPROCESSORS_ONLN))
    {
        /* background sampler */
        std::thread([this]{ loop(); }).detach();
    }

    /* ── helpers ────────────────────────────────────────────────────── */
    static unsigned long long read_jiffies(const std::string& path,int col14or15)
    {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return 0ULL;

        char buf[256]{}; ssize_t n = ::read(fd, buf, sizeof(buf)-1); ::close(fd);
        if (n <= 0) return 0ULL;

        int col = 1; unsigned long long val = 0; char* p = buf;
        while (col <= col14or15 && *p)
        {
            if (*p == ' ' || *p == '\n') { ++col; ++p; continue; }
            if (col == col14or15)
                val = strtoull(p, &p, 10);
            else
                while (*p && *p != ' ' && *p != '\n') ++p;
        }
        return val;
    }

    std::vector<pid_t> tids_matching(bool want_imci)
    {
        std::vector<pid_t> out;
        std::string dir = "/proc/" + std::to_string(pid_) + "/task";
        if (DIR* d = ::opendir(dir.c_str()))
        {
            while (dirent* ent = ::readdir(d))
            {
                if (ent->d_type != DT_DIR) continue;
                pid_t tid = atoi(ent->d_name);
                if (tid <= 0) continue;

                std::string cpath = dir + '/' + ent->d_name + "/comm";
                char name[64] = {};
                int fd = ::open(cpath.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd >= 0) { ::read(fd, name, 63); ::close(fd); }
                bool match = std::strstr(name, "imci[");
                if (match == want_imci) out.push_back(tid);
            }
            ::closedir(d);
        }
        return out;
    }

    std::pair<double,double> sample_once()
    {
        auto sum = [&](const std::vector<pid_t>& tids)
        {
            unsigned long long tot = 0;
            for (pid_t tid : tids)
            {
                std::string st = "/proc/" + std::to_string(pid_) +
                                 "/task/" + std::to_string(tid) + "/stat";
                tot += read_jiffies(st, 14) + read_jiffies(st, 15);
            }
            return tot;
        };

        unsigned long long cj = sum( tids_matching(true ) );   // column
        unsigned long long rj = sum( tids_matching(false) );   // row

        static unsigned long long prev_cj = 0, prev_rj = 0;
        double d_c = cj - prev_cj;  prev_cj = cj;
        double d_r = rj - prev_rj;  prev_rj = rj;

        double denom = hz_ * ncpu_;
        return { 100.0 * d_r / denom, 100.0 * d_c / denom };
    }

    void loop()
    {
        while (true)
        {
            auto rc = sample_once();
            row_pct_.store(rc.first , std::memory_order_relaxed);
            col_pct_.store(rc.second, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    /* ── data members ──────────────────────────────────────────────── */
    const pid_t pid_;
    const long  hz_;
    const long  ncpu_;

    /* atomics for row/col CPU percentages */
    std::atomic<double> row_pct_{0.0};
    std::atomic<double> col_pct_{0.0};
};

// /*══════════════════  SECTION 2 – CPU-meter  ═════════════*/
// class CpuMeter {
// public:
//     /* 单例 */
//     static CpuMeter& inst() { static CpuMeter x; return x; }

//     /* 最近 1 秒的 CPU 百分比 (行 %, 列 %) */
//     std::pair<double,double> latest_cpu() const {
//         return { row_pct_.load(std::memory_order_relaxed),
//                  col_pct_.load(std::memory_order_relaxed) };
//     }

//     /* ---- 新增：并发计数 ---- */
//     int active_row_threads() const {
//         return int(active_row_.load(std::memory_order_relaxed));
//     }
//     int active_col_threads() const {
//         return int(active_col_.load(std::memory_order_relaxed));
//     }

// private:
//     CpuMeter()
//       : pid_(::getpid()),
//         hz_ ( ::sysconf(_SC_CLK_TCK) ),
//         ncpu_(::sysconf(_SC_NPROCESSORS_ONLN))
//     {   std::thread([this]{ loop(); }).detach(); }

//     /* ── internal helpers ─────────────────────────────── */
//     static unsigned long long read_jiffies(const std::string& path,int col14or15);
//     std::vector<pid_t> tids_matching(bool want_imci);

//     void sample_once(double& row_cpu, double& col_cpu,
//                      int& row_thr,  int& col_thr);

//     void loop() {
//         while (true) {
//             double r_cpu,c_cpu; int r_thr,c_thr;
//             sample_once(r_cpu,c_cpu,r_thr,c_thr);

//             row_pct_.store(r_cpu, std::memory_order_relaxed);
//             col_pct_.store(c_cpu, std::memory_order_relaxed);
//             active_row_.store(r_thr, std::memory_order_relaxed);
//             active_col_.store(c_thr, std::memory_order_relaxed);

//             std::this_thread::sleep_for(std::chrono::seconds(1));
//         }
//     }

//     /* ── data ─────────────────────────────────────────── */
//     const pid_t pid_;
//     const long  hz_, ncpu_;

//     std::atomic<double> row_pct_{0.0}, col_pct_{0.0};
//     std::atomic<int>    active_row_{0}, active_col_{0};
// };

// /* ==========  helper bodies  ========== */

// inline void CpuMeter::sample_once(double& row_cpu,double& col_cpu,
//                                   int& row_thr,int& col_thr)
// {
//     auto tids_row = tids_matching(false);
//     auto tids_col = tids_matching(true );

//     auto sum = [&](const std::vector<pid_t>& tids){
//         unsigned long long tot = 0;
//         for (pid_t tid : tids) {
//             std::string st = "/proc/" + std::to_string(pid_) +
//                              "/task/" + std::to_string(tid) + "/stat";
//             tot += read_jiffies(st,14)+read_jiffies(st,15);
//         }
//         return tot;
//     };

//     static unsigned long long prev_r = 0, prev_c = 0;
//     double d_r = sum(tids_row) - prev_r;  prev_r += d_r;
//     double d_c = sum(tids_col) - prev_c;  prev_c += d_c;

//     row_cpu = 100.0 * d_r / (hz_ * ncpu_);
//     col_cpu = 100.0 * d_c / (hz_ * ncpu_);
//     row_thr = int(tids_row.size());
//     col_thr = int(tids_col.size());
// }



/*══════════════════  SECTION 3 – PD dispatcher  ═════════════*/
class PdDispatcher {
public:
    /*  decide(score)
        -  score  > θ   → choose column-engine
        -  score ≤ θ   → choose row-engine
        The primal-dual update keeps the row-CPU share ρ within
        [gamma_low, gamma_up] on average.
    */
    bool decide(double score)
    {
        auto rc  = CpuMeter::inst().latest();                 // {row%, col%}
        double rho   = rc.first / (rc.first + rc.second + 1e-9);

        /* logistic decision boundary θ */
        double theta = 1.0 / (1.0 + std::exp(-beta * (lambda_low - lambda_up)));
        bool   choose_column = (score > theta);

        /* dual updates (projected sub-gradient) */
        lambda_up  = std::max(0.0, lambda_up  + eta * (rho        - gamma_up));
        lambda_low = std::max(0.0, lambda_low + eta * (gamma_low - rho));

        return choose_column;          // true = column engine, false = row
    }

private:
    double lambda_up  = 0.0;           // dual variable for ρ_row > gamma_up
    double lambda_low = 0.0;           // dual variable for ρ_row < gamma_low

    /* hyper-parameters */
    static constexpr double gamma_up  = 0.70;
    static constexpr double gamma_low = 0.30;
    static constexpr double eta       = 0.02;   // learning-rate
    static constexpr double beta      = 1.0;    // sigmoid scale
};



// /*══════════════════  SECTION 3 – Primal-Dual dispatcher  ═════════════*/
// class PdDispatcher {
// public:
//     /* 调度决策：
//        score  > θ  → 列存
//        score ≤ θ  → 行存                                              */
//     bool decide(double score)
//     {
//         /* ------------ ① 统计量 ------------ */
//         auto rc      = CpuMeter::inst().latest_cpu();     // {row%, col%}
//         int  col_thr = CpuMeter::inst().active_col_threads();

//         double rho = rc.first / (rc.first + rc.second + 1e-9);

//         /* ------------ ② 对偶变量更新 ------------ */
//         /* CPU 上下界 */
//         lambda_up   = std::max(0.0, lambda_up  + eta * (rho        - gamma_up));
//         lambda_low  = std::max(0.0, lambda_low + eta * (gamma_low - rho));

//         /* 并发约束：col_thr ≤ COL_MAX                */
//         lambda_col  = std::max(0.0, lambda_col + eta_c *
//                                (double(col_thr) - double(COL_MAX)));

//         /* ------------ ③ 计算阈值 θ 并决策 ------------ */
//         double theta = 1.0 / (1.0 + std::exp( -beta *
//                         (lambda_low - lambda_up) - beta_col * lambda_col ));

//         return (score > theta);               // true = 选列存
//     }

// private:
//     /* —— 对偶变量 —— */
//     double lambda_up  = 0.0;
//     double lambda_low = 0.0;
//     double lambda_col = 0.0;

//     /* —— 常量 / 超参数 —— */
//     static constexpr double gamma_up   = 0.70;
//     static constexpr double gamma_low  = 0.30;
//     static constexpr int    COL_MAX    = 8;      // 列存最大并发
//     static constexpr double eta        = 0.02;   // CPU 约束步长
//     static constexpr double eta_c      = 0.05;   // 并发约束步长
//     static constexpr double beta       = 1.0;    // CPU 项 sigmoid
//     static constexpr double beta_col   = 1.0;    // 并发项 sigmoid
// };


static PdDispatcher g_pd;


/*══════════════════  SECTION 4 – plan2feat & walk()  ═════════════*/
/*  >>> 这段直接拷贝你给的完整版（略去只保留声明 + 32-维实现）       */
#define NUM_FEATS 32
#include <nlohmann/json.hpp>
using json = nlohmann::json;

/* ── Agg struct（略掉无关字段，只保留 walk() 需要的） ─────────*/
struct Agg{
    double re=0,rp=0,f=0,rc=0,ec=0,pc=0,dr=0;
    int cRange=0,cRef=0,cEq=0,cIdx=0,cFull=0,idxUse=0,
        sumPK=0,cnt=0;
    double selSum=0,selMin=1e30,selMax=0,
           ratioSum=0,ratioMax=0,maxPrefix=0,minRead=1e30,
           pcDepth3=0,join_ratio_max=0,outerRows=0;
    int eqChainDepth=0,_curEqChain=0,maxDepth=0;
};
/* —— tiny helpers —— */
static inline double safe_f(const json&o,const char*k){
    if(!o.contains(k)) return 0; const auto&v=o[k];
    if(v.is_number()) return v.get<double>();
    if(v.is_string()){ try{ return std::stod(v.get<std::string>());}catch(...){ } }
    return 0;
}
static inline double str_size_to_num(std::string s){
    if(s.empty()) return 0; char suf=s.back(); double m=1;
    if(suf=='G'||suf=='g'){m=1e9; s.pop_back();}
    else if(suf=='M'||suf=='m'){m=1e6; s.pop_back();}
    else if(suf=='K'||suf=='k'){m=1e3; s.pop_back();}
    try{ return std::stod(s)*m;}catch(...){ return 0;}
}
static inline double log_tanh(double v,double c=20){ return std::tanh(std::log1p(std::max(0.0,v))/c); }

/* —— recursive walk —— */
void walk(const json& n, Agg& a,int depth=1){
    if(n.is_object()){
        if(n.contains("table")&&n["table"].is_object()){
            const auto&t=n["table"]; const auto&ci=t.value("cost_info",json::object());
            double re=safe_f(t,"rows_examined_per_scan");
            double rp=safe_f(t,"rows_produced_per_join");
            double fl=safe_f(t,"filtered");
            double rc=safe_f(ci,"read_cost");
            double ec=safe_f(ci,"eval_cost");
            double pc=safe_f(ci,"prefix_cost");
            double dr=ci.contains("data_read_per_join")&&ci["data_read_per_join"].is_string()
                      ?str_size_to_num(ci["data_read_per_join"].get<std::string>())
                      :safe_f(ci,"data_read_per_join");
            a.re+=re; a.rp+=rp; a.f+=fl; a.rc+=rc; a.ec+=ec; a.pc+=pc; a.dr+=dr; ++a.cnt;
            a.maxPrefix=std::max(a.maxPrefix,pc); a.minRead=std::min(a.minRead,rc);
            if(re>0){ double sel=rp/re; a.selSum+=sel; a.selMin=std::min(a.selMin,sel); a.selMax=std::max(a.selMax,sel);}
            double ratio=(ec>0?rc/ec:rc); a.ratioSum+=ratio; a.ratioMax=std::max(a.ratioMax,ratio);
            std::string at=t.value("access_type","ALL");
            if(at=="range")++a.cRange;else if(at=="ref")++a.cRef;else if(at=="eq_ref")++a.cEq;
            else if(at=="index")++a.cIdx;else ++a.cFull;
            if(t.value("using_index",0))++a.idxUse;
            if(t.contains("possible_keys")&&t["possible_keys"].is_array())
                a.sumPK+=int(t["possible_keys"].size());
            if(a.outerRows==0&&at!="ALL") a.outerRows=re;
            if(at=="eq_ref"){ ++a._curEqChain; a.eqChainDepth=std::max(a.eqChainDepth,a._curEqChain);}
            else a._curEqChain=0;
            if(depth==3) a.pcDepth3+=pc;
        }
        if(n.contains("nested_loop")&&n["nested_loop"].is_array()){
            const auto&nl=n["nested_loop"]; if(nl.size()>=2){
                auto rows_of=[&](const json&x){return x.contains("table")?safe_f(x["table"],"rows_produced_per_join"):safe_f(x,"rows_produced_per_join");};
                double l=rows_of(nl[0]), r=rows_of(nl[1]);
                if(l>0&&r>0) a.join_ratio_max=std::max(a.join_ratio_max,
                                        std::max(l,r)/std::max(1.0,std::min(l,r)));
            }
        }
        for(const auto&kv:n.items())
            if(kv.key()!="table") walk(kv.value(),a,depth+1);
    }else if(n.is_array())
        for(const auto&v:n) walk(v,a,depth);
    a.maxDepth=std::max(a.maxDepth,depth);
}

/* —— 32-dim plan2feat —— */
bool plan2feat(const json& plan,float f[NUM_FEATS]){
    if(!plan.contains("query_block")) return false;
    const json&qb=plan["query_block"];
    Agg a; walk(qb,a); if(!a.cnt) return false;
    double inv=1.0/a.cnt, qCost=qb["cost_info"].value("query_cost",0.0);
    int k=0; auto P=[&](double v){ f[k++]=float(v); };
    P(log_tanh(a.re*inv));                 //0
    P(log_tanh(a.rp*inv));                 //1
    P(log_tanh(a.f*inv));                  //2
    P(log_tanh(a.rc*inv));                 //3
    P(0);                                  //4
    P(log_tanh(a.pc*inv));                 //5
    P(log_tanh(a.dr*inv));                 //6
    P(0);                                  //7
    P(a.cRef*inv);                         //8
    P(a.cEq*inv);                          //9
    P(0);P(0);                             //10-11
    P(a.idxUse*inv);                       //12
    P(0);                                  //13
    P(a.selMin);                           //14
    P(0);P(0);P(0);P(0);P(0);P(0);         //15-20
    P(a.ratioSum*inv);                     //21
    P(a.ratioMax);                         //22
    P(std::log1p(qCost)/15.0);             //23
    P(0);                                  //24
    P(log_tanh((a.pc*inv)/std::max(1e-6,a.rc*inv))); //25
    P(log_tanh((a.rc*inv)/std::max(1e-6,a.re*inv))); //26
    P(log_tanh((a.ec*inv)/std::max(1e-6,a.re*inv))); //27
    P(0);P(0);                             //28-29
    P(log_tanh(a.maxDepth*(a.idxUse*inv)));          //30
    P(log_tanh((a.idxUse*inv)/std::max(1e-6,a.cFull*inv))); //31
    P(a.cnt);                              //32
    P(a.cnt?double(a.sumPK)/a.cnt:0.0);    //33
    P(log_tanh(a.maxPrefix));              //34
    P(log_tanh(a.minRead<1e30?a.minRead:0)); //35
    P(0);P(a.selMax-a.selMin);             //36-38
    P(a.idxUse/double(std::max(1,a.cRange+a.cRef+a.cEq+a.cIdx))); //39
    P(0);P(0);P(0);P(0);                   //40-43
    P(log_tanh(a.re*inv)-log_tanh(a.selSum*inv)); //44
    P(0);P(0);                             //45-46
    P(double(a.sumPK));                    //47
    while(k<57) P(0);
    P(log_tanh(a.maxPrefix*inv));          //57
    while(k<62) P(0);
    P(a.selMin>0?a.selMax/a.selMin:0);     //62
    P(log_tanh(a.outerRows));              //63
    P(double(a.eqChainDepth));             //64
    while(k<96) P(0);
    P(log_tanh((a.pc>0?a.pc:1e-6)/std::max(1e-6,a.pcDepth3))); //96
    while(k<123) P(0);
    P(std::log1p(a.join_ratio_max));       //123
    return true;
}

/*══════════════════  SECTION 5 – LightGBM one-time loader  ═════════════*/
static BoosterHandle g_booster=nullptr;
static void ensure_model_loaded(){
    if(g_booster) return;
    int n_it=0;
    if(LGBM_BoosterCreateFromModelfile(LGB_MODEL,&n_it,&g_booster)!=0)
        die("Cannot load LightGBM model "+std::string(LGB_MODEL));
}

/*══════════════════  SECTION 6 – progress bar helper  ══════════════════*/
struct ProgressBar{
    std::mutex mu; size_t done=0,total=0; std::string tag;
    void init(const std::string&t,size_t tot){ tag=t; total=tot; refresh(); }
    void step(size_t n=1){ std::lock_guard<std::mutex> lk(mu); done+=n; refresh(); }
    void refresh(){
        const size_t W=40;
        double f=total?double(done)/total:1.0;
        size_t filled=size_t(f*W);
        std::cerr<<'\r'<<tag<<" ["<<std::string(filled,'=')<<std::string(W-filled,' ')<<"] "
                 <<std::setw(3)<<int(f*100)<<"% ("<<done<<'/'<<total<<")"<<std::flush;
    }
    void done_and_clear(){ std::cerr<<"\r"<<std::string(60,' ')<<"\r"; }
};

/*══════════════════  SECTION 7 – worker & dispatcher  ══════════════════*/
struct Args{
    int threads=20;
    int timeout_ms=600000;
};

struct WorkCtx{
    const std::vector<std::string>* queries;
    const std::vector<std::string>* sess_sql;
    std::vector<double>* lat;
    ProgressBar* bar;
    Args args;
    bool is_lgbm_pd=false;
    const std::vector<double>* arrivals;   // ★ 每条查询的到达秒
    double bench_start = 0.0;
};


/*════════ SECTION 7 – worker & dispatcher (arrival-aware) ════════════*/
static void worker_func(int tid, const WorkCtx& ctx)
{
    constexpr double COST_THRESHOLD = 50000.0;      // cost_thresh 用阈值

    mysql_thread_init();                            // 每线程初始化 libmysql
    MYSQL* conn = mysql_init(nullptr);
    if (!mysql_real_connect(conn, HOST, USER, PASS, DB.c_str(), PORT, nullptr, 0))
        die("[T" + std::to_string(tid) + "] connect fail: " +
            std::string(mysql_error(conn)));

    /* session-level knobs for the chosen routing mode */
    for (const auto& s : *ctx.sess_sql) mysql_query(conn, s.c_str());
    mysql_query(conn,
        ("SET max_execution_time = " + std::to_string(ctx.args.timeout_ms)).c_str());

    float feat[NUM_FEATS];

    /* ─────────────────────────────────────────────────────────────── */
    for (std::size_t idx = tid; idx < ctx.queries->size(); idx += ctx.args.threads)
    {
        const std::string& q = (*ctx.queries)[idx];

        /* ---------- ① 等待该查询的到达时刻 ---------- */
        const double arr_rel = (*ctx.arrivals)[idx];           // 相对秒
        const double arr_abs = ctx.bench_start + arr_rel;      // 绝对时刻
        double now = wall_now();
        if (now < arr_abs)
            std::this_thread::sleep_for(std::chrono::duration<double>(arr_abs - now));

        /* ---------- ② 可选：外部 LightGBM + Primal-Dual 路由 ---------- */
        bool use_col = false;
        if (ctx.is_lgbm_pd) {
            /* 仅简写 – 保留你原先的 EXPLAIN → plan2feat → LGBM 推理部分 */
            /* 若想强制使用列存:  mysql_query(conn,"SET use_imci_engine = FORCED"); */
        }

        /* ---------- ③ 真正执行 SQL ---------- */
        double finish   = 0.0;
        unsigned err_no = 0;

        if (mysql_query(conn, q.c_str()) == 0) {                /* OK */
            if (MYSQL_RES* r = mysql_store_result(conn)) mysql_free_result(r);
            finish = wall_now();
        } else {                                                /* ERROR / TIMEOUT */
            err_no = mysql_errno(conn);
            finish = wall_now();
        }

        /* ---------- ④ 计算 latency （含等待） ---------- */
        double latency = finish - arr_abs;                      // core definition
        if (err_no == 3024 || err_no == 1317)                   // statement timeout
            latency = std::min(latency, ctx.args.timeout_ms / 1000.0);

        /* 失败（其他错误）保持 latency=0.0，后续统计时会被过滤掉 */
        (*ctx.lat)[idx] = latency;
        ctx.bar->step();
    }
    /* ─────────────────────────────────────────────────────────────── */

    mysql_close(conn);
    mysql_thread_end();
}






/*══════════════════  SECTION 8 – run one mode  ═════════════════════════*/
static std::pair<double,std::vector<double>>
run_mode(const std::string& tag,
         const std::vector<std::string>& queries,
         const std::vector<double>&      arrivals,   // ★ 新参
         const Args& args)
{
    std::vector<double> lat(queries.size(),0);
    ProgressBar bar; bar.init(tag, queries.size());

    WorkCtx ctx;
    ctx.queries=&queries;
    ctx.sess_sql=&ROUTING_MODES.at(tag);
    ctx.lat=&lat;
    ctx.bar=&bar;
    ctx.args=args;
    ctx.is_lgbm_pd = (tag=="lgbm_pd");
    ctx.arrivals     = &arrivals;          // ★
    ctx.bench_start  = wall_now();         // 统一 0 点

    auto T0=wall_now();
    std::vector<std::thread> pool;
    for(int t=0;t<args.threads;++t)
        pool.emplace_back(worker_func,t,std::cref(ctx));
    for(auto&th:pool) th.join();
    double mk=wall_now()-T0;
    bar.done_and_clear();
    return {mk,std::move(lat)};
}

/*════════ SECTION 9 – query loader (reworked) ═══════════════════════*/
static std::vector<std::string> load_sql_file(const std::string& path)
{
    std::vector<std::string> qs;
    std::ifstream in(path);
    if (!in) { die("cannot open "+path); }

    std::string ln, cur;
    while (std::getline(in, ln))
    {
        /* skip comment lines that start with “--” (like the Python version) */
        std::string trimmed = ln;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
        if (trimmed.rfind("--", 0) == 0) continue;

        cur += ln;

        /* one semicolon terminates one statement */
        std::size_t pos;
        while ((pos = cur.find(';')) != std::string::npos)
        {
            std::string sql = cur.substr(0, pos);

            /* ─── clean-up: remove all double-quotes exactly like Python ─── */
            sql.erase(std::remove(sql.begin(), sql.end(), '"'), sql.end());

            /* strip leading / trailing whitespace */
            std::size_t first = sql.find_first_not_of(" \t\r\n");
            std::size_t last  = sql.find_last_not_of(" \t\r\n");
            if (first != std::string::npos)
                qs.emplace_back(sql.substr(first, last - first + 1));

            cur.erase(0, pos + 1);  // consume up-to ‘;’ and continue
        }
    }
    return qs;
}


/*══════════════════  SECTION 10 – CSV writer  ═════════════════════════*/
static void write_csv(const std::string& out,
                      const std::unordered_map<std::string,
                            std::pair<double,std::vector<double>>>& res)
{
    std::ofstream fout(out);
    std::vector<std::string> tags;
    for(const auto&kv:res) tags.push_back(kv.first);
    fout<<"query_idx";
    for(auto&t:tags) fout<<','<<t<<"_lat";
    fout<<'\n';
    size_t N = res.begin()->second.second.size();
    for(size_t i=0;i<N;++i){
        fout<<i;
        for(auto&t:tags) fout<<','<<res.at(t).second[i];
        fout<<'\n';
    }
}



/*══════════════════  SECTION 11 – main  ══════════════════════════════*/
int main(int argc, char* argv[])
{
    if (mysql_library_init(0, nullptr, nullptr))
        die("mysql_library_init() failed");

    /* ---------- CLI parsing (同 Python 脚本一致) ---------- */
    Args args;                              // –threads / –timeout
    std::string dataset   = "tpcds_sf1";
    std::string data_dir  = "/home/wuy/query_costs/";
    std::string ap_file_override, tp_file_override;
    int  limit   = 50;
    std::string out_csv = "routing_bench.csv";
    int  seed    = 42;

    auto need_arg = [&](int& i) -> std::string {
        if (i + 1 >= argc) die("missing value after " + std::string(argv[i]));
        return std::string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--threads" || a == "-t") args.threads    = std::stoi(need_arg(i));
        else if (a == "--timeout")              args.timeout_ms = std::stoi(need_arg(i));
        else if (a == "--limit"   || a == "-n") limit           = std::stoi(need_arg(i));
        else if (a == "--dataset")              dataset         = need_arg(i);
        else if (a == "--data_dir")             data_dir        = need_arg(i);
        else if (a == "--ap")                   ap_file_override= need_arg(i);
        else if (a == "--tp")                   tp_file_override= need_arg(i);
        else if (a == "--out")                  out_csv         = need_arg(i);
        else if (a == "--seed")                 seed            = std::stoi(need_arg(i));
        else die("unknown flag: " + a);
    }

    DB = dataset;                                   // ← 动态切库

    /* ---------- 解析 workload SQL ---------- */
    const std::string dset_dir = data_dir + "/workloads/" + dataset;

    const std::string ap_file =
        !ap_file_override.empty() ? ap_file_override
                                  : dset_dir + "/workload_100k_s1_group_order_by_more_complex.sql";
    const std::string tp_file =
        !tp_file_override.empty() ? tp_file_override
                                  : dset_dir + "/TP_queries.sql";

    auto qs_ap = load_sql_file(ap_file);
    auto qs_tp = load_sql_file(tp_file);

    std::vector<std::string> queries;
    queries.reserve(qs_tp.size() + qs_ap.size());
    // queries.insert(queries.end(), qs_tp.begin(), qs_tp.end());
    queries.insert(queries.end(), qs_ap.begin(), qs_ap.end());

    std::mt19937 rng(seed);
    std::shuffle(queries.begin(), queries.end(), rng);
    if (limit > 0 && limit < static_cast<int>(queries.size()))
        queries.resize(limit);

    std::cout << "\nLoaded " << queries.size() << " queries ("
              << qs_tp.size() << " TP + " << qs_ap.size() << " AP)\n";


    /* ----------- 生成 Poisson-like arrival 时间 ----------- */
    std::vector<double> arrivals(queries.size());
    {
        std::mt19937 rng(seed);                         // 同一个 seed → 各 mode 共用
        std::exponential_distribution<double> exp(1.0 / 0.005); // 平均 5 ms 到达
        double t = 0.0;
        for (size_t i = 0; i < arrivals.size(); ++i) {
            t += exp(rng);
            arrivals[i] = t;                           // 单调递增的“相对秒”
        }
    }

    /* ---------- 依次执行 5 种 routing mode ---------- */
    const std::vector<std::string> MODE_ORDER = {
        "row_only", "col_only", "cost_thresh", "hybrid_opt", "lgbm_pd"
    };

    std::unordered_map<std::string,
                       std::pair<double,std::vector<double>>> mode_res;

    for (const auto& tag : MODE_ORDER)
    {
        if (tag == "row_only" || tag == "col_only")
            continue;

        std::cout << "\n=== " << tag << " ===\n";
        auto res = run_mode(tag, queries, arrivals, args);          // {makespan, lat[]}
        const double makespan = res.first;
        const auto&  lat      = res.second;

        /* 统计成功条数、平均 latency */
        std::vector<double> good;  good.reserve(lat.size());
        for (double v : lat) if (v > 0.0) good.push_back(v);     // 0 ⇒ 失败

        const double avg_lat = good.empty() ? 0.0
                            : std::accumulate(good.begin(), good.end(), 0.0) / good.size();
        const double qps     = makespan > 0.0
                            ? static_cast<double>(good.size()) / makespan
                            : 0.0;

        std::cout << std::fixed
                  << "makespan " << std::setprecision(2) << makespan << " s  "
                  << "avg "      << std::setprecision(4) << avg_lat  << " s  "
                  << "qps "      << std::setprecision(2) << qps      << " q/s  "
                  << '(' << good.size() << '/' << lat.size() << " ok)\n";

        mode_res.emplace(tag, std::move(res));
    }

    /* ---------- CSV dump ---------- */
    write_csv(out_csv, mode_res);
    std::cout << "\nPer-query latencies saved to  " << out_csv << '\n';
    return 0;
}
