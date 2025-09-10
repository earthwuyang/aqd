#ifndef AQD_LIGHTGBM_C_API_H
#define AQD_LIGHTGBM_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load LightGBM model from text file. Returns 1 on success, 0 on failure. */
int aqd_lgb_load_model(const char *model_path);

/* Predict using named features. Names/values arrays must have length n. */
double aqd_lgb_predict_named(const char **names, const double *values, int n);

/* Unload model and free resources. */
void aqd_lgb_unload(void);

#ifdef __cplusplus
}
#endif

#endif /* AQD_LIGHTGBM_C_API_H */

