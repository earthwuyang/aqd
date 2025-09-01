#include "global_stats.hpp"
#include <cmath>
#include <algorithm>
GlobalStats& global_stats(){
    static GlobalStats inst; return inst;
}
void GlobalStats::update(double r, double c){
    std::lock_guard<std::mutex> lk(mtx_);
    if (frozen_) return;                 // 冻结后不再更新
    /* 简单在线算法：移动平均 + 移动 IQR（P² 算法可自行替换） */
    static double n = 0, mr = 0, mc = 0;
    ++n;
    mr += (r - mr)/n;   mc += (c - mc)/n;
    norm_.rows_c = mr;  norm_.cost_c = mc;
    /* rows_s / cost_s 用滑动方差近似 */
    static double sr=0, sc=0;
    sr += (r - mr)*(r - mr);
    sc += (c - mc)*(c - mc);
    norm_.rows_s = std::sqrt(sr / std::max(1.0,n-1));
    norm_.cost_s = std::sqrt(sc / std::max(1.0,n-1));
}
