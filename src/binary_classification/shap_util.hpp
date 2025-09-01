#pragma once
#include <vector>
#include <LightGBM/c_api.h>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "common.hpp"     // for chk()

/* ---------- 1. 取 |SHAP| 均值 ---------- */
inline std::vector<double>
shap_mean_abs(const std::vector<float>& X, int N, int F, BoosterHandle booster)
{
    const int OUT_F = F + 1;                      // +bias
    std::vector<double> shap(size_t(N) * OUT_F);
    int64_t out_len = 0;
    chk(!LGBM_BoosterPredictForMat(
            booster, X.data(), C_API_DTYPE_FLOAT32,
            N, F, 1, C_API_PREDICT_CONTRIB,
            -1, 0, "", &out_len, shap.data()),
        "SHAP predict failed");

    std::vector<double> mu(F, 0.0);
    for (int i = 0; i < N; ++i) {
        const double* row = shap.data() + size_t(i) * OUT_F;
        for (int j = 0; j < F; ++j) mu[j] += std::fabs(row[j]);
    }
    for (double& v : mu) v /= N;
    return mu;
}

/* ---------- 2. 相关系数矩阵 & 去冗余 ---------- */
inline std::vector<int>
reduce_by_corr(const std::vector<float>& X, int N,
               const std::vector<int>& idx, double thr_abs=0.9)
{
    const int K = (int)idx.size();
    if (K <= 1) return idx;

    /* 拷贝子矩阵到 Eigen (N × K) */
    Eigen::MatrixXd A(N, K);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k)
            A(i,k) = X[size_t(i)*NUM_FEATS + idx[k]];

    /* 标准化到 0 均值 */
    for (int k = 0; k < K; ++k) {
        double mean = A.col(k).mean();
        A.col(k).array() -= mean;
        double sd = std::sqrt((A.col(k).array().square().sum()) / N);
        if (sd > 0) A.col(k) /= sd;
    }

    /* corr = AᵀA / (N-1) */
    Eigen::MatrixXd C = (A.transpose() * A) / double(N - 1);

    std::vector<int> keep; keep.reserve(K);
    std::vector<char> removed(K, 0);

    for (int i = 0; i < K; ++i) if (!removed[i]) {
        keep.push_back(idx[i]);
        for (int j = i + 1; j < K; ++j)
            if (std::fabs(C(i,j)) >= thr_abs) removed[j] = 1;
    }
    return keep;
}
