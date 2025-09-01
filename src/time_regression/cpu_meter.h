/* cpu_meter.h  –  lightweight row/col CPU sampler  (C++17) */
#pragma once
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>

class CpuMeter {
public:
    /* 单例：第一次调用 instance() 时启动后台线程 */
    static CpuMeter& instance();

    /* 最近 1 s 的 {row%, col%}  */
    std::pair<double,double> sample() const {
        return last_;
    }

private:
    CpuMeter();                   /* spawns background thread */
    ~CpuMeter();                  /* joins thread            */

    void loop();                  /* background tick         */
    std::pair<double,double> read_once() const;

    const pid_t   pid_;
    const long    hz_, ncpu_;
    std::atomic<bool> stop_{false};
    std::thread  bg_;
    std::atomic<std::pair<double,double>> last_;
};
