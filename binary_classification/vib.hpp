#pragma once
#include <vector>
#include <Eigen/Dense>
#include <random>

struct VibSelector {
    std::vector<int>     keep;      // 训练后保留的列
    std::vector<double>  w;         // 所有特征权重

    /*  λ = L1 正则；max_it = 迭代次数 */
    template<std::size_t F>
    void fit(const std::vector<std::array<float,F>>& X,
             const std::vector<int>& y,
             double lambda = 1e-3,
             int max_it = 200)
    {
        const int N = (int)X.size();
        if (!N) return;

        Eigen::MatrixXd A(N, F);
        Eigen::VectorXd t(N);
        for (int i=0;i<N;++i){
            for (int j=0;j<F;++j) A(i,j)=X[i][j];
            t(i)=y[i];
        }

        Eigen::VectorXd wv(F); wv.setZero();
        Eigen::VectorXd z  = A*wv;               // logits
        Eigen::VectorXd p  = 1.0/(1.0+(-z).array().exp());

        for(int it=0; it<max_it; ++it){
            for(int j=0;j<F;++j){
                // 1. 计算除 j 以外的残差
                Eigen::VectorXd r = t - p + A.col(j)*(wv(j));
                double aj = A.col(j).squaredNorm();
                double cj = (A.col(j).dot(r));

                // 2. 软阈值
                double w_new;
                if (cj >  lambda) w_new = (cj-lambda)/aj;
                else if(cj < -lambda) w_new = (cj+lambda)/aj;
                else w_new = 0.0;

                // 3. 更新
                if (w_new!=wv(j)){
                    Eigen::VectorXd delta = A.col(j)*(w_new-wv(j));
                    z += delta;
                    p  = 1.0/(1.0+(-z).array().exp());
                    wv(j)=w_new;
                }
            }
        }
        w.assign(wv.data(), wv.data()+F);
        keep.clear();
        for(int j=0;j<F;++j) if (std::fabs(w[j])>0) keep.push_back(j);
    }
};
