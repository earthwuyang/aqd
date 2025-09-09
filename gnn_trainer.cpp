#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <nlohmann/json.hpp>

/*
 Simple GNN trainer scaffold:
 - Reads JSONL plan logs ({"query_hash":..., "plan":[...]})
 - Reads a CSV file with labels: query_hash, target (log_time_ratio)
 - Produces a weights file understood by gnn_inference.c:
     in_features hidden_dim
     W1 (in_features x hidden_dim)
     b1 (hidden_dim)
     W2 (hidden_dim)
     b2 (1)

 This is a placeholder trainer that initializes small random weights.
 Extend with real optimization as needed.
*/

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <plans.jsonl> <labels.csv> <out_model.txt>\n";
        return 1;
    }
    std::string plans_path = argv[1];
    std::string labels_path = argv[2];
    std::string out_path = argv[3];

    // Load plans (not used in placeholder trainer)
    std::ifstream fin(plans_path);
    if (!fin) {
        std::cerr << "Failed to open plans file\n";
        return 1;
    }
    std::string line;
    size_t plan_count = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            (void)j;
            plan_count++;
        } catch (...) {
            // skip
        }
    }
    fin.close();

    // Load labels (not used in placeholder)
    std::ifstream flabels(labels_path);
    if (!flabels) {
        std::cerr << "Failed to open labels file\n";
        return 1;
    }
    size_t label_count = 0;
    std::string l;
    std::getline(flabels, l); // header
    while (std::getline(flabels, l)) {
        if (!l.empty()) label_count++;
    }
    flabels.close();

    // Initialize small random weights
    int in_features = 16;
    int hidden = 16;
    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 0.01);

    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "Failed to open output model file\n";
        return 1;
    }
    fout << in_features << " " << hidden << "\n";
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < hidden; j++) {
            fout << nd(rng) << (j + 1 == hidden ? '\n' : ' ');
        }
    }
    for (int j = 0; j < hidden; j++) fout << 0.0 << (j + 1 == hidden ? '\n' : ' ');
    for (int j = 0; j < hidden; j++) fout << nd(rng) << (j + 1 == hidden ? '\n' : ' ');
    fout << 0.0 << "\n";
    fout.close();

    std::cerr << "Wrote placeholder GNN model to " << out_path << " using "
              << plan_count << " plans and " << label_count << " labels\n";
    return 0;
}

