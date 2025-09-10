#ifndef AQD_GNN_C_API_H
#define AQD_GNN_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load R-GINN model from text file. Returns 1 on success, 0 on failure. */
int aqd_gnn_load_model(const char *model_path);

/* Returns 1 if a model is loaded, 0 otherwise. */
int aqd_gnn_is_loaded(void);

/* Predict using already loaded model on a compact RG graph */
double aqd_gnn_predict(int N, int in_dim, const double *X,
                       int num_rel, const int *indptr, const int *indices);

/* Unload model and free resources. */
void aqd_gnn_unload(void);

#ifdef __cplusplus
}
#endif

#endif /* AQD_GNN_C_API_H */
