/*
 * rginn_inference.h
 * 
 * R-GINN (Relational Graph Isomorphism Network) inference in PostgreSQL kernel
 */

#ifndef RGINN_INFERENCE_H
#define RGINN_INFERENCE_H

#include "postgres.h"
#include "nodes/plannodes.h"

/* Default model path - same as in gnn_trainer_real.cpp */
#define DEFAULT_RGINN_MODEL_PATH "models/rginn_routing_model.txt"

/* Initialize and load R-GINN model */
extern bool rginn_load_model(const char *path);

/* Main prediction function - returns routing score (positive = DuckDB, negative = PostgreSQL) */
extern double rginn_predict_plan(PlannedStmt *planned_stmt);

#endif /* RGINN_INFERENCE_H */