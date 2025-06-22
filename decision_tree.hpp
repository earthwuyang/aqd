#pragma once
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "json.hpp"   // nlohmann/json

/* ───────────────────── 数据结构 ────────────────────── */
struct DTNode {
    int   feat = -1;       // -1 表 leaf
    float thresh = 0.f;    // 分裂阈值
    int   left  = -1;      // 左右子节点 index
    int   right = -1;
    float prob  = 0.f;     // 叶节点：列存概率
};

class DecisionTree {
public:
    /* 训练入口：X=行数×d，y=0/1 */
    void fit(const std::vector<std::array<float,32>>& X,
             const std::vector<int>& y,
             int max_depth   = 8,
             int min_samples = 50)
    {
        feats_ = X[0].size();
        nodes_.clear();
        std::vector<int> idx(X.size());
        std::iota(idx.begin(), idx.end(), 0);
        build_node(idx, X, y, 0, max_depth, min_samples);
    }

    /* 0-1 概率预测（递归走树） */
    float predict(const float feats[32]) const {
        int id = 0;
        while(nodes_[id].feat != -1) {
            id = (feats[nodes_[id].feat] < nodes_[id].thresh) ?
                 nodes_[id].left : nodes_[id].right;
        }
        return nodes_[id].prob;
    }

    /* JSON 保存 / 加载（单文件） */
    bool save(const std::string& path) const {
        nlohmann::json j = nodes_;
        std::ofstream(path) << j.dump(2);
        return true;
    }
    bool load(const std::string& path) {
        std::ifstream f(path);
        if(!f) return false;
        nlohmann::json j;  f >> j;
        nodes_ = j.get<std::vector<DTNode>>();
        feats_ = 32;
        return true;
    }

private:
    std::vector<DTNode> nodes_;
    int feats_;

    /* 递归建树：返回当前节点 index */
    int build_node(const std::vector<int>& idx,
                   const std::vector<std::array<float,32>>& X,
                   const std::vector<int>& y,
                   int depth, int max_depth, int min_samples)
    {
        DTNode node;
        float pos = 0.f;
        for(int i: idx) pos += y[i];
        node.prob = pos / idx.size();

        /* 停止条件 */
        if(depth >= max_depth || idx.size() <= min_samples ||
           node.prob <= 0.01f || node.prob >= 0.99f)
        {
            node.feat = -1;      // leaf
            nodes_.push_back(node);
            return nodes_.size()-1;
        }

        /* 尝试所有特征，选 Gini 最优 */
        int   best_feat = -1;
        float best_thr  = 0.f;
        float best_gini = 1e9;
        std::vector<int> left_idx, right_idx;

        std::vector<float> sorted(idx.size());
        for(int f=0; f<feats_; ++f) {
            for(size_t k=0;k<idx.size();++k)
                sorted[k] = X[idx[k]][f];
            std::sort(sorted.begin(), sorted.end());
            /* 取 11 个候选分位点 */
            for(int s=1;s<=10;++s) {
                float thr = sorted[ s*sorted.size()/11 ];
                int nl=0,nr=0,pl=0,pr=0;
                for(int i: idx){
                    if(X[i][f] < thr){ nl++; pl+=y[i]; }
                    else             { nr++; pr+=y[i]; }
                }
                if(nl<min_samples || nr<min_samples) continue;
                float gini =
                    (nl? 2.0f*(pl/(float)nl)*(1-pl/(float)nl):0) +
                    (nr? 2.0f*(pr/(float)nr)*(1-pr/(float)nr):0);
                if(gini < best_gini){
                    best_gini = gini;
                    best_feat = f;
                    best_thr  = thr;
                }
            }
        }

        if(best_feat==-1){        // 无合法分裂 → 叶
            node.feat = -1;
            nodes_.push_back(node);
            return nodes_.size()-1;
        }

        /* 根据 best_feat,best_thr 再分样本 */
        std::vector<int> left,right;
        for(int i: idx){
            ((X[i][best_feat] < best_thr)? left:right).push_back(i);
        }

        node.feat = best_feat;
        node.thresh = best_thr;
        int cur = nodes_.size();
        nodes_.emplace_back(node);            // 预留当前节点

        int left_id  = build_node(left ,X,y,depth+1,max_depth,min_samples);
        int right_id = build_node(right,X,y,depth+1,max_depth,min_samples);

        nodes_[cur].left  = left_id;
        nodes_[cur].right = right_id;
        return cur;
    }
};

/* JSON <-> DTNode 序列化 */
inline void to_json(nlohmann::json& j, const DTNode& n){
    j = {{"feat",n.feat}, {"thresh",n.thresh},
         {"left",n.left}, {"right",n.right}, {"prob",n.prob}};
}
inline void from_json(const nlohmann::json& j, DTNode& n){
    j.at("feat").get_to(n.feat); j.at("thresh").get_to(n.thresh);
    j.at("left").get_to(n.left); j.at("right").get_to(n.right);
    j.at("prob").get_to(n.prob);
}
