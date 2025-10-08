#!/usr/bin/env python3
"""
Neural network training script for the 85-feature schema using PyTorch.

Supports class imbalance handling via optional focal loss and class-weighted
sample weighting. Outputs validation metrics and saves the trained model plus
feature scaler for later inference.
"""

import argparse
import glob
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is required for train_NN.py. Please install torch.") from exc


FEATURE_NAMES = [
    "num_tables", "num_joins", "query_depth", "complexity_score",
    "has_aggregates", "has_group_by", "has_order_by", "has_limit", "has_distinct",
    "has_window_functions", "has_outer_joins", "estimated_join_complexity",
    "has_subqueries", "has_correlated_subqueries", "has_large_tables", "all_tables_small",
    "has_complex_expressions", "has_user_functions", "has_text_operations", "has_numeric_heavy_ops",
    "num_aggregate_funcs", "analytical_pattern", "transactional_pattern", "etl_pattern", "command_type",
    "join_type_inner", "join_type_left", "join_type_right", "join_type_full", "join_type_cross",
    "predicate_simple_eq", "predicate_range", "predicate_like", "predicate_in", "predicate_exists",
    "has_parameters", "num_cte", "max_subquery_depth", "has_recursive_cte", "has_lateral_join",
    "selectivity_high", "selectivity_medium", "selectivity_low", "cardinality_large", "cardinality_medium",
    "index_usage_likely", "partition_pruning_likely", "parallel_safe", "has_volatile_funcs", "cost_estimate_high",
    "total_projected_bytes", "avg_projected_row_fraction", "max_projected_row_fraction", "projected_column_count",
    "projected_text_columns", "projected_numeric_columns", "projected_json_columns", "output_row_width",
    "limit_value", "has_order_by_limit", "avg_scan_fraction", "max_scan_fraction", "total_rowstore_bytes_est",
    "total_columnar_bytes_est", "has_covering_index", "covering_index_score", "order_by_index_match",
    "predicate_correlation_max", "predicate_correlation_avg", "group_ndv_est", "groups_per_input_row",
    "fk_to_pk_joins", "many_to_many_joins", "star_schema_score", "topk_indexed", "topk_log_limit",
    "text_predicate_indexable", "text_predicate_nonindexable", "duckdb_table_count", "duckdb_parquet_table_count",
    "duckdb_pushdown_score", "volatile_function_count", "parallel_unsafe_function_count",
    "estimated_rows_output", "estimated_result_bytes"
]


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def load_data(data_dir, filename="training_data.csv"):
    logger = setup_logger()
    patterns = [
        os.path.join(data_dir, "training_data_*.csv"),
        os.path.join(data_dir, "training_data*.csv"),
        os.path.join(data_dir, "training_data_combined.csv"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    if files:
        frames = []
        for fp in files:
            try:
                df_part = pd.read_csv(fp)
                logger.info("Loaded %d samples from %s", len(df_part), os.path.basename(fp))
                frames.append(df_part)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fp, exc)
        if not frames:
            raise RuntimeError("No valid training CSV files could be loaded")
        df = pd.concat(frames, ignore_index=True)
        logger.info("Total: %d samples from %d files", len(df), len(frames))
    else:
        fp = os.path.join(data_dir, filename)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"No training data found in {data_dir}")
        df = pd.read_csv(fp)
        logger.info("Loaded %d samples from %s", len(df), fp)

    max_time_ms = 60000
    df = df[(df['pg_time_ms'] > 0) & (df['pg_time_ms'] < max_time_ms) &
            (df['duck_time_ms'] > 0) & (df['duck_time_ms'] < max_time_ms)]

    for feature in FEATURE_NAMES:
        if feature not in df.columns:
            df[feature] = 0

    return df


def prepare_dataframe(df):
    df = df.copy()
    eps = 1e-3
    df['target'] = np.log((df['pg_time_ms'].values + eps) / (df['duck_time_ms'].values + eps))
    df['class_label'] = (df['target'] > 0).astype(int)
    return df


def compute_sample_weights(labels, mode, neg_weight, pos_weight, neg_tilt):
    if mode == 'fixed':
        return np.where(labels == 1, float(pos_weight), float(neg_weight)).astype(np.float32)

    counts = np.bincount(labels.astype(int), minlength=2)
    total = counts.sum()
    w0 = total / (2.0 * max(counts[0], 1))
    w1 = total / (2.0 * max(counts[1], 1))
    if mode == 'balanced_tilt':
        w0 *= float(neg_tilt)
    weights = np.where(labels == 1, w1, w0).astype(np.float32)
    return weights


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def focal_loss_with_logits(logits, targets, *, alpha=0.25, gamma=2.0):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * torch.pow(1.0 - pt, gamma) * bce
    return loss


def compute_loss(logits, targets, weights, *, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
    if use_focal:
        loss = focal_loss_with_logits(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
    else:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def parse_hidden_layers(spec):
    if isinstance(spec, (list, tuple)):
        return tuple(int(v) for v in spec)
    parts = [p.strip() for p in str(spec).split(',') if p.strip()]
    return tuple(int(p) for p in parts) if parts else (256, 128)


def evaluate_predictions(prob, true_labels, pg_ms, duck_ms):
    pred = (prob > 0.5).astype(int)
    acc = accuracy_score(true_labels, pred)
    prec = precision_score(true_labels, pred, zero_division=0)
    rec = recall_score(true_labels, pred, zero_division=0)
    cm = confusion_matrix(true_labels, pred)
    latency = np.where(pred == 1, duck_ms, pg_ms)
    avg_latency = float(np.mean(latency))
    pred_fraction = float(np.mean(pred)) if len(pred) else 0.0
    return acc, prec, rec, cm, avg_latency, pred_fraction


def accuracy_score(y_true, y_pred):
    return float((y_true == y_pred).mean())


def precision_score(y_true, y_pred, zero_division=0):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    denom = tp + fp
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def recall_score(y_true, y_pred, zero_division=0):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    denom = tp + fn
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description='Train neural network with optional focal loss')
    parser.add_argument('--data-dir', default='lightgbm_training_data_new')
    parser.add_argument('--data-file', default='training_data.csv')
    parser.add_argument('--heldout', type=float, default=0.2, help='Validation fraction')
    parser.add_argument('--hidden-layers', default='256,128', help='Comma-separated hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--class-weight-mode', choices=['fixed', 'balanced', 'balanced_tilt'], default='balanced_tilt')
    parser.add_argument('--neg-weight', type=float, default=3.0)
    parser.add_argument('--pos-weight', type=float, default=0.6)
    parser.add_argument('--neg-tilt', type=float, default=1.5)
    parser.add_argument('--min-train-loggap', type=float, default=0.0, help='Filter training samples with |loggap| < value')
    parser.add_argument('--focal-loss', action='store_true', default=True, help='Enable focal loss for training')
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    args = parser.parse_args()

    logger = setup_logger()
    df = load_data(args.data_dir, args.data_file)
    df = prepare_dataframe(df)

    if args.min_train_loggap > 0.0:
        mask = df['target'].abs() >= args.min_train_loggap
        kept = mask.sum()
        logger.info("Training on %d/%d samples with |loggap| >= %.3f", kept, len(df), args.min_train_loggap)
        if kept < len(df) * 0.1:
            logger.warning("Too few strong-signal samples; reverting to full dataset")
        else:
            df = df[mask]

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df['class_label'].astype(int).to_numpy()
    pg_ms = df['pg_time_ms'].to_numpy(dtype=np.float32)
    duck_ms = df['duck_time_ms'].to_numpy(dtype=np.float32)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val, pg_train, pg_val, duck_train, duck_val = train_test_split(
        X, y, pg_ms, duck_ms,
        test_size=args.heldout,
        stratify=stratify,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    sample_weights = compute_sample_weights(
        y_train,
        args.class_weight_mode,
        args.neg_weight,
        args.pos_weight,
        args.neg_tilt,
    )
    sample_weights = sample_weights / np.mean(sample_weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    w_train_tensor = torch.from_numpy(sample_weights).float()

    X_val_tensor = torch.from_numpy(X_val_scaled).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, w_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    model = MLPNet(input_dim=X_train_tensor.shape[1], hidden_layers=hidden_layers, dropout=args.dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    logger.info(
        "Training MLP: layers=%s, epochs=%d, lr=%.2e, focal=%s",
        hidden_layers,
        args.epochs,
        args.learning_rate,
        args.focal_loss,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            xb, yb, wb = [t.to(device) for t in batch]
            logits = model(xb)
            loss = compute_loss(
                logits,
                yb,
                wb,
                use_focal=args.focal_loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)

        avg_train_loss = total_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = compute_loss(
                    logits,
                    yb,
                    weights=None,
                    use_focal=args.focal_loss,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                )
                val_loss += float(loss.item()) * len(xb)

        avg_val_loss = val_loss / len(val_dataset)
        logger.info("Epoch %3d | train loss %.5f | val loss %.5f", epoch, avg_train_loss, avg_val_loss)

        if avg_val_loss + 1e-6 < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()

    acc, prec, rec, cm, avg_lat, pred_fraction = evaluate_predictions(prob, y_val, pg_val, duck_val)

    logger.info("Validation metrics (proba > 0.5 -> DuckDB):")
    logger.info("  Accuracy:  %.3f", acc)
    logger.info("  Precision: %.3f", prec)
    logger.info("  Recall:    %.3f", rec)
    logger.info("  Predicted DuckDB fraction: %.4f", pred_fraction)
    logger.info("Confusion Matrix:")
    logger.info("                 Predicted PG  Predicted DuckDB")
    logger.info("  Actual PG:     %8d     %8d", cm[0, 0], cm[0, 1])
    logger.info("  Actual DuckDB: %8d     %8d", cm[1, 0], cm[1, 1])
    logger.info("Average latency (ms) under NN routing: %.2f", avg_lat)

    model_dir = os.path.join(args.data_dir, "nn_models")
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"nn_model_{timestamp}.pt")

    torch.save(
        {
            'state_dict': model.state_dict(),
            'input_dim': X_train_tensor.shape[1],
            'hidden_layers': hidden_layers,
            'dropout': args.dropout,
            'scaler_mean': scaler.mean_.astype(np.float32),
            'scaler_scale': scaler.scale_.astype(np.float32),
            'feature_names': FEATURE_NAMES,
            'class_weight_mode': args.class_weight_mode,
            'neg_weight': args.neg_weight,
            'pos_weight': args.pos_weight,
            'neg_tilt': args.neg_tilt,
            'min_train_loggap': args.min_train_loggap,
            'focal_loss': args.focal_loss,
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma,
        },
        model_path,
    )
    logger.info("Model saved to: %s", model_path)


if __name__ == '__main__':
    main()
