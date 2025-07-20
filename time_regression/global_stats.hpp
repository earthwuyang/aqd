#pragma once
#include <unordered_map>
#include <string>
#include <mutex>

struct ColNorm {
    double rows_c = 0, rows_s = 1;
    double cost_c = 0, cost_s = 1;
};
class GlobalStats {
    ColNorm          norm_;
    std::mutex       mtx_;
    bool             frozen_ = false;     // 训练集遍历完即“冻结”
public:
    void update(double ln_rows, double ln_cost);   // 见 cpp
    ColNorm get() const { return norm_; }
    void   freeze()     { frozen_ = true; }
};
GlobalStats& global_stats();              // 单例
