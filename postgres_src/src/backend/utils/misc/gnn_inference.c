#include "postgres.h"
#include "gnn_inference.h"
#include "utils/memutils.h"
#include <string.h>

static int op_type_id(NodeTag tag){
    switch(tag){
        case T_SeqScan: return 1; case T_IndexScan: return 2; case T_BitmapHeapScan: return 3;
        case T_NestLoop: return 4; case T_MergeJoin: return 5; case T_HashJoin: return 6;
        case T_Sort: return 7; case T_Agg: return 8; case T_Group: return 9; default: return 0;
    }
}

static int rel_id_for_child(NodeTag tag){ int id=op_type_id(tag); if (id>=1 && id<=3) return 0; if (id>=4 && id<=6) return 1; return 2; }

static int get_node_id(Plan **order, int N, Plan *p){ for (int i=0;i<N;++i) if (order[i]==p) return i; return -1; }

static void plan_node_features(Plan *plan, double *feat, int d){
    memset(feat,0,sizeof(double)*d);
    int k=10; int id=op_type_id(nodeTag(plan)); if (id>=0 && id<k && id<d) feat[id]=1.0;
    int idx=k; if (idx<d) feat[idx++]= (double)plan->plan_rows; if (idx<d) feat[idx++]= (double)plan->plan_width;
    if (idx<d){ double cpr=(plan->plan_rows>0)?(plan->total_cost/plan->plan_rows):plan->total_cost; feat[idx++]=cpr; }
}

GNNModel *gnn_create_model(void){ GNNModel*m=(GNNModel*)palloc0(sizeof(GNNModel)); m->loaded=false; return m; }
void gnn_free_model(GNNModel *model){ if(!model) return; rginn_free(&model->core); pfree(model); }
bool gnn_load_model(GNNModel *model, const char *path){ if(!model||!path) return false; if (rginn_load(&model->core, path)!=0) return false; model->loaded=true; return true; }

double gnn_predict_plan(GNNModel *model, PlannedStmt *planned_stmt){
    if (!model || !model->loaded || !planned_stmt || !planned_stmt->planTree) return 0.0;
    // Build nodes list BFS
    Plan *root = planned_stmt->planTree;
    // For simplicity traverse left/right tree; ignoring subplans for now
    // First count nodes
    int count=0; Plan *stack[2048]; int sp=0; stack[sp++]=root; while(sp){ Plan *p=stack[--sp]; count++; if (p->lefttree) stack[sp++]=p->lefttree; if (p->righttree) stack[sp++]=p->righttree; if (sp>=2048) break; }
    int N=count, F=model->core.in_dim, R=3, H=model->core.hidden_dim;
    double *X = (double*)palloc0((size_t)N*F*sizeof(double));
    int *indptr = (int*)palloc0((size_t)R*(N+1)*sizeof(int));
    int *indices = (int*)palloc0((size_t)(2*N)*sizeof(int)); // rough upper bound
    // second pass assemble arrays
    int idx=0; sp=0; stack[sp++]=root; Plan *order[2048]; while(sp){ Plan *p=stack[--sp]; order[idx++]=p; if (p->lefttree) stack[sp++]=p->lefttree; if (p->righttree) stack[sp++]=p->righttree; }
    // map plan* to node id
    // simple array search (small trees); could use hash if needed
    // features and neighbors
    int edge_count[3]={0,0,0};
    for (int i=0;i<N;++i){ plan_node_features(order[i], &X[(size_t)i*F], F); for(int r=0;r<R;++r){ int *ind=&indptr[r*(N+1)]; ind[i+1]=ind[i]; }
        if (order[i]->lefttree){ int r=rel_id_for_child(nodeTag(order[i]->lefttree)); int *ind=&indptr[r*(N+1)]; ind[i+1]++; indices[ind[r*(N+1)+N] + edge_count[r]++] = get_node_id(order, N, order[i]->lefttree); }
        if (order[i]->righttree){ int r=rel_id_for_child(nodeTag(order[i]->righttree)); int *ind=&indptr[r*(N+1)]; ind[i+1]++; indices[ind[r*(N+1)+N] + edge_count[r]++] = get_node_id(order, N, order[i]->righttree); }
    }
    // convert per-node counts to prefix sums per relation and compact indices array
    int total=0; for(int r=0;r<R;++r){ int *ind=&indptr[r*(N+1)]; int sum=0; for(int i=0;i<=N;++i){ int tmp=ind[i]; ind[i]=total+sum; sum+=tmp; } total+=sum; }
    // compact indices (we wrote into reserved area; for robustness, rebuild by scanning again)
    pfree(indices); indices=(int*)palloc0((size_t)total*sizeof(int)); int cursor[3]={0,0,0};
    for (int i=0;i<N;++i){ if (order[i]->lefttree){ int r=rel_id_for_child(nodeTag(order[i]->lefttree)); int base=indptr[r*(N+1)+i]; indices[base + cursor[r]++] = get_node_id(order, N, order[i]->lefttree); } if (order[i]->righttree){ int r=rel_id_for_child(nodeTag(order[i]->righttree)); int base=indptr[r*(N+1)+i]; indices[base + cursor[r]++] = get_node_id(order, N, order[i]->righttree); } }

    RGGraph g; g.N=N; g.in_dim=F; g.X=X; g.num_rel=R; g.indptr=indptr; g.indices=indices;
    double *h0=(double*)palloc0((size_t)N*H*sizeof(double)); double *m1=(double*)palloc0((size_t)N*H*sizeof(double)); double *h1=(double*)palloc0((size_t)N*H*sizeof(double)); double *gr=(double*)palloc0((size_t)H*sizeof(double));
    double y=rginn_forward(&model->core, &g, h0, m1, h1, gr);
    pfree(h0); pfree(m1); pfree(h1); pfree(gr); pfree(X); pfree(indptr); pfree(indices);
    return y;
}
