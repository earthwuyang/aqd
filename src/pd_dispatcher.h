/* pd_dispatcher.h – online primal-dual policy */
#pragma once
#include "cpu_meter.h"
#include <cmath>

class PdDispatcher {
public:
    /* 返回 true=列存，false=行存，并更新 λ */
    bool decide_and_update(double score){
        auto rc  = CpuMeter::instance().sample();
        double rho = rc.first/(rc.first+rc.second+1e-9);   // row ratio

        double theta = 1.0/(1.0+std::exp(-beta*(λ_up-λ_lo)));
        bool   choose_col = (score > theta);   /* primal step */

        /* dual update */
        λ_up = std::max(0.0, λ_up + η*(rho - γ_up ));
        λ_lo = std::max(0.0, λ_lo + η*(γ_lo - rho));

        return choose_col;
    }
private:
    double λ_up = 0, λ_lo = 0;
    /* same constants as内核 */
    static constexpr double γ_up = 0.70;
    static constexpr double γ_lo = 0.30;
    static constexpr double η    = 0.02;
    static constexpr double β    = 1.0;
};
