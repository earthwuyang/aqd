#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
extern "C" {
#include "rginn.h"
}

using json = nlohmann::json;

static int op_type_id(const std::string& t){
    if (t=="Seq Scan") return 1; 
    if (t=="Index Scan") return 2; 
    if (t=="Bitmap Heap Scan") return 3;
    if (t=="Nested Loop") return 4; 
    if (t=="Merge Join") return 5; 
    if (t=="Hash Join") return 6;
    if (t=="Sort") return 7; 
    if (t=="Aggregate") return 8; 
    if (t=="Group") return 9; 
    return 0;
}

static int rel_id_for_child(const std::string& t){
    int id = op_type_id(t);
    if (id==1 || id==2 || id==3) return 0; // scans
    if (id==4 || id==5 || id==6) return 1; // joins
    return 2; // others
}

struct GraphBuild {
    std::vector<double> X;
    std::vector<int> indptr;
    std::vector<int> indices;
    int N=0; int F=0; int R=1;
};

static void build_graph_from_plan(const json& plan, int in_dim, GraphBuild& gb){
    std::vector<json> nodes;
    std::vector<std::vector<int>> children;
    std::vector<std::string> types;
    
    json root_plan = plan;
    if (plan.contains("Plan") && plan["Plan"].is_object()) {
        root_plan = plan["Plan"];
    }
    
    nodes.push_back(root_plan);
    children.emplace_back();
    types.push_back(root_plan.value("Node Type",""));
    
    for (size_t i=0; i<nodes.size(); ++i){
        const json& n = nodes[i];
        if (n.contains("Plans") && n["Plans"].is_array()){
            for (const auto& ch : n["Plans"]) {
                int idx = (int)nodes.size();
                json child = ch.contains("Plan")?ch["Plan"]:ch;
                nodes.push_back(child);
                children.emplace_back();
                types.push_back(child.value("Node Type",""));
                children[i].push_back(idx);
            }
        }
    }
    
    gb.N = (int)nodes.size(); 
    gb.F = in_dim; 
    gb.R = 3;
    gb.X.assign((size_t)gb.N * gb.F, 0.0);
    gb.indptr.assign((size_t)gb.R * (gb.N + 1), 0);
    
    std::vector<std::vector<std::vector<int>>> adj(gb.R, std::vector<std::vector<int>>(gb.N));
    
    const int k = 10;
    for (int i=0;i<gb.N;++i){
        const json& n = nodes[i];
        std::string t = n.value("Node Type", "");
        int id = op_type_id(t);
        if (id>=0 && id<k) gb.X[i*gb.F + id] = 1.0;
        int idx = k;
        double rows = n.value("Plan Rows", 0.0);
        double width = n.value("Plan Width", 0.0);
        double tc = n.value("Total Cost", 0.0);
        
        if (idx<gb.F) gb.X[i*gb.F + idx++] = std::log(rows + 1.0) / 10.0;
        if (idx<gb.F) gb.X[i*gb.F + idx++] = width / 100.0;
        if (idx<gb.F) gb.X[i*gb.F + idx++] = std::log(tc + 1.0) / 10.0;
        
        for (int u : children[i]) {
            int r = rel_id_for_child(types[u]);
            if (r < 0 || r >= gb.R) r = 2;
            adj[r][i].push_back(u);
        }
    }
    
    gb.indices.clear();
    int base = 0;
    for (int r = 0; r < gb.R; ++r) {
        int *ind = &gb.indptr[(size_t)r * (gb.N + 1)];
        ind[0] = base;
        for (int i = 0; i < gb.N; ++i) {
            base += (int)adj[r][i].size();
            ind[i+1] = base;
            for (int dst : adj[r][i]) gb.indices.push_back(dst);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // Load GNN model
    RGINNModel model;
    if (rginn_load(&model, model_path.c_str()) != 0) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        return 1;
    }
    
    // Read JSON from stdin
    json input;
    try {
        std::cin >> input;
    } catch(const std::exception& e) {
        std::cerr << "ERROR: Invalid JSON input" << std::endl;
        return 1;
    }
    
    // Build graph
    GraphBuild gb;
    build_graph_from_plan(input, 16, gb);
    
    if (gb.N == 0) {
        std::cout << "0.0" << std::endl;
        return 0;
    }
    
    // Prepare graph structure
    RGGraph g;
    g.N = gb.N;
    g.in_dim = 16;
    g.X = gb.X.data();
    g.num_rel = gb.R;
    g.indptr = gb.indptr.data();
    g.indices = gb.indices.data();
    
    // Allocate buffers for forward pass
    int hidden = model.hidden_dim;
    std::vector<double> h0((size_t)g.N * hidden), m1((size_t)g.N * hidden), 
                       h1((size_t)g.N * hidden), gr(hidden);
    
    // Make prediction
    double prediction = rginn_forward(&model, &g, h0.data(), m1.data(), h1.data(), gr.data());
    
    // Output prediction (positive means DuckDB is faster)
    std::cout << prediction << std::endl;
    
    return 0;
}