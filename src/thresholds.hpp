struct Thres { double cost75=5e4,  qSmall=1e4, qLarge=5e4; };
Thres& global_thres();                 // 同样用单例
void   collect_qcost(double v);        // load 样本时调用一次
void   finalise_thres();               // 所有样本读完后调用一次
