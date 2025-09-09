#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <nlohmann/json.hpp>

/*
 Minimal GNN trainer:
 - Parses JSONL plan logs (each line: {"query_hash":..., "plan":[...]})
 - Parses labels CSV: query_hash,target (log_time_ratio)
 - Computes a fixed hidden representation at the root using the same recursive
   mean-aggregator as the kernel with W1 = Identity, b1=0, ReLU.
 - Trains a linear readout W2,b2 via ridge regression: y ~ W2^T h_root + b2
 - Writes a model file compatible with kernel gnn_inference.c with:
     in_features hidden_dim
     W1 (identity)
     b1 (zeros)
     W2 (learned)
     b2 (learned)
*/

using json = nlohmann::json;

static int op_type_id(const std::string& node_type) {
    if (node_type == "Seq Scan") return 1;
    if (node_type == "Index Scan") return 2;
    if (node_type == "Bitmap Heap Scan") return 3;
    if (node_type == "Nested Loop") return 4;
    if (node_type == "Merge Join") return 5;
    if (node_type == "Hash Join") return 6;
    if (node_type == "Sort") return 7;
    if (node_type == "Aggregate") return 8;
    if (node_type == "Group") return 9;
    return 0;
}

struct NodeFeat {
    std::vector<double> h; // hidden vector
};

static void relu(std::vector<double>& v) {
    for (auto& x : v) if (x < 0.0) x = 0.0;
}

static void add_inplace(std::vector<double>& a, const std::vector<double>& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

static void div_inplace(std::vector<double>& a, double d) {
    if (d == 0.0) return;
    for (auto& x : a) x /= d;
}

// Build fixed hidden from plan tree: h = ReLU(x + mean(child_h)) with x as one-hot and numeric fields
static std::vector<double> aggregate_plan(const json& plan, int in_features) {
    const int k = 10; // one-hot ops
    std::vector<double> x(in_features, 0.0);
    std::string node_type = plan.value("Node Type", "");
    int id = op_type_id(node_type);
    if (id >= 0 && id < k && id < in_features) x[id] = 1.0;
    int idx = k;
    double plan_rows = plan.value("Plan Rows", 0.0);
    double plan_width = plan.value("Plan Width", 0.0);
    double total_cost = plan.value("Total Cost", 0.0);
    double cpr = (plan_rows > 0.0) ? (total_cost / plan_rows) : total_cost;
    if (idx < in_features) x[idx++] = plan_rows;
    if (idx < in_features) x[idx++] = plan_width;
    if (idx < in_features) x[idx++] = cpr;

    // Children
    std::vector<double> agg(in_features, 0.0);
    int child_count = 0;
    if (plan.contains("Plans") && plan["Plans"].is_array()) {
        for (const auto& ch : plan["Plans"]) {
            auto hc = aggregate_plan(ch, in_features);
            add_inplace(agg, hc);
            child_count++;
        }
    }
    if (child_count > 0) div_inplace(agg, (double)child_count);
    add_inplace(agg, x);
    relu(agg);
    return agg;
}

static bool parse_labels(const std::string& labels_path, std::unordered_map<std::string,double>& ymap) {
    std::ifstream f(labels_path);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line)) return false; // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto pos = line.find(',');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos+1);
        try {
            double t = std::stod(val);
            ymap[key] = t;
        } catch (...) {}
    }
    return true;
}

static void solve_ridge(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double lambda,
                        std::vector<double>& w, double& b) {
    // Solve (Xb^T Xb + lambda I) w = Xb^T y, where Xb = [X | 1]
    const size_t n = X.size();
    if (n == 0) { w.clear(); b = 0.0; return; }
    const size_t d = X[0].size();
    const size_t D = d + 1; // bias
    std::vector<double> A(D*D, 0.0);
    std::vector<double> rhs(D, 0.0);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> xb(D, 1.0);
        for (size_t j = 0; j < d; ++j) xb[j] = X[i][j];
        // accumulate
        for (size_t r = 0; r < D; ++r) {
            rhs[r] += xb[r] * y[i];
            for (size_t c = 0; c < D; ++c) A[r*D + c] += xb[r] * xb[c];
        }
    }
    // ridge
    for (size_t j = 0; j < d; ++j) A[j*D + j] += lambda;
    // Solve by Gaussian elimination
    // Augment A|rhs
    for (size_t r = 0; r < D; ++r) {
        // pivot
        size_t piv = r;
        for (size_t rr = r+1; rr < D; ++rr) if (std::fabs(A[rr*D + r]) > std::fabs(A[piv*D + r])) piv = rr;
        if (piv != r) {
            for (size_t c = 0; c < D; ++c) std::swap(A[r*D + c], A[piv*D + c]);
            std::swap(rhs[r], rhs[piv]);
        }
        double diag = A[r*D + r];
        if (std::fabs(diag) < 1e-12) continue;
        for (size_t c = r; c < D; ++c) A[r*D + c] /= diag;
        rhs[r] /= diag;
        for (size_t rr = 0; rr < D; ++rr) if (rr != r) {
            double f = A[rr*D + r];
            for (size_t c = r; c < D; ++c) A[rr*D + c] -= f * A[r*D + c];
            rhs[rr] -= f * rhs[r];
        }
    }
    // solution in rhs
    w.assign(d, 0.0);
    for (size_t j = 0; j < d; ++j) w[j] = rhs[j];
    b = rhs[d];
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <plans.jsonl> <labels.csv> <out_model.txt>\n";
        return 1;
    }
    std::string plans_path = argv[1];
    std::string labels_path = argv[2];
    std::string out_path = argv[3];

    // Load labels
    std::unordered_map<std::string,double> ymap;
    if (!parse_labels(labels_path, ymap)) {
        std::cerr << "Failed to open labels file\n";
        return 1;
    }
    // Iterate plan JSONL and build dataset
    std::ifstream fin(plans_path);
    if (!fin) {
        std::cerr << "Failed to open plans file\n";
        return 1;
    }
    const int in_features = 16; // matches kernel default
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::string line;
    size_t parsed = 0, matched = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        json j;
        try { j = json::parse(line); } catch (...) { continue; }
        parsed++;
        if (!j.contains("query_hash") || !j.contains("plan")) continue;
        std::string qh = j.value("query_hash", "");
        auto it = ymap.find(qh);
        if (it == ymap.end()) continue;
        // plan is array; take first entry's "Plan"
        try {
            const json& arr = j["plan"];
            if (!arr.is_array() || arr.empty()) continue;
            const json& plan_root = arr[0].contains("Plan") ? arr[0]["Plan"] : arr[0];
            auto h = aggregate_plan(plan_root, in_features);
            X.push_back(std::move(h));
            y.push_back(it->second);
            matched++;
        } catch (...) {
            continue;
        }
    }
    fin.close();

    if (X.empty()) {
        std::cerr << "No matched plans/labels; cannot train.\n";
        return 1;
    }

    // Train linear readout
    std::vector<double> W2; double b2 = 0.0;
    solve_ridge(X, y, 1e-3, W2, b2);

    // Write model: W1=Identity, b1=0, W2 learned, b2 learned
    const int hidden = in_features;
    std::ofstream fout(out_path);
    if (!fout) { std::cerr << "Failed to open output model file\n"; return 1; }
    fout << in_features << " " << hidden << "\n";
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < hidden; j++) {
            double val = (i == j) ? 1.0 : 0.0;
            fout << val << (j + 1 == hidden ? '\n' : ' ');
        }
    }
    for (int j = 0; j < hidden; j++) fout << 0.0 << (j + 1 == hidden ? '\n' : ' '); // b1
    for (int j = 0; j < hidden; j++) fout << W2[j] << (j + 1 == hidden ? '\n' : ' ');
    fout << b2 << "\n";
    fout.close();

    std::cerr << "Trained GNN readout on " << matched << "/" << parsed << " plans; model -> " << out_path << "\n";
    return 0;
}
