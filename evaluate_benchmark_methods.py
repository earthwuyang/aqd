#!/usr/bin/env python3
"""
Offline Benchmark of Routing Methods Using Previously Executed Queries

Reads unified training data from data/execution_data/*_unified_training_data.json,
filters to queries that succeeded on both PostgreSQL and DuckDB, and evaluates
the aggregate runtime for each routing method by summing the time of the engine
chosen by that method without re-executing queries.

Methods evaluated:
- optimal: chooses the faster of (postgres_time, duckdb_time)
- default: heuristic approximating pg_duckdb default routing
- cost_threshold: heuristic threshold on estimated complexity
- lightgbm: placeholder heuristic (server-side model not invoked here)
- gnn: placeholder heuristic (server-side model not invoked here)

Outputs a comparison summary and saves a JSON report.
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
EXEC_DIR = BASE_DIR / 'data' / 'execution_data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_query_signals(query_text: str):
    qt = query_text.upper()
    has_join = ' JOIN ' in qt
    has_group_by = ' GROUP BY ' in qt
    has_order_by = ' ORDER BY ' in qt
    has_agg = any(tok in qt for tok in [' SUM(', ' COUNT(', ' AVG(', ' MIN(', ' MAX('])
    join_count = qt.count(' JOIN ')
    length = len(qt)
    return {
        'has_join': has_join,
        'has_group_by': has_group_by,
        'has_order_by': has_order_by,
        'has_agg': has_agg,
        'join_count': join_count,
        'length': length,
    }


def default_decision(signals):
    # Approximate pg_duckdb default heuristic
    # - Simple selects without heavy operations -> Postgres
    # - Aggregations, many joins, or complex queries -> DuckDB
    if signals['has_agg'] or signals['has_group_by']:
        return 'duckdb'
    if signals['join_count'] > 2:
        return 'duckdb'
    # Short/simple queries to Postgres
    if signals['length'] < 200 and signals['join_count'] == 0:
        return 'postgres'
    return 'postgres'


def complexity_estimate(signals, features: dict | None):
    # Build a rough complexity score from signals and optional feature hints
    score = 0.0
    score += signals['join_count'] * 1000
    if signals['has_agg']:
        score += 5000
    if signals['has_group_by']:
        score += 3000
    score += min(signals['length'], 2000) * 1.0
    # If kernel-like features exist (cost-ish), incorporate them lightly
    if features:
        # Try a few known-like keys
        for k in ('aqd_feature_14', 'total_cost', 'plan_total_cost'):
            v = features.get(k)
            if isinstance(v, (int, float)):
                score += float(v)
                break
    return score


def cost_threshold_decision(signals, features, threshold: float = 10000.0):
    cost = complexity_estimate(signals, features)
    return 'duckdb' if cost > threshold else 'postgres'


def lightgbm_decision(signals, features):
    # Placeholder: lean towards DuckDB for analytical patterns
    if signals['has_agg'] or signals['has_group_by']:
        return 'duckdb'
    if signals['join_count'] >= 2:
        return 'duckdb'
    return 'postgres'


def gnn_decision(signals, features):
    # Placeholder: emphasizes structure (joins) more heavily
    if signals['join_count'] >= 3:
        return 'duckdb'
    if signals['has_agg'] and signals['join_count'] >= 1:
        return 'duckdb'
    return 'postgres'


def evaluate_methods(records, default_thresh=10000.0):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for rec in records:
        pt = rec.get('postgres_time')
        dt = rec.get('duckdb_time')
        q = rec.get('query_text', '')
        feats = rec.get('features') or rec.get('aqd_features') or {}
        if not isinstance(feats, dict):
            feats = {}
        if pt is None or dt is None:
            continue

        sig = parse_query_signals(q)

        # Optimal
        opt_engine = 'postgres' if pt <= dt else 'duckdb'
        totals['optimal'] += pt if opt_engine == 'postgres' else dt
        counts['optimal'] += 1

        # Default
        d_engine = default_decision(sig)
        totals['default'] += pt if d_engine == 'postgres' else dt
        counts['default'] += 1

        # Cost threshold
        c_engine = cost_threshold_decision(sig, feats, threshold=default_thresh)
        totals['cost_threshold'] += pt if c_engine == 'postgres' else dt
        counts['cost_threshold'] += 1

        # LightGBM (placeholder)
        l_engine = lightgbm_decision(sig, feats)
        totals['lightgbm'] += pt if l_engine == 'postgres' else dt
        counts['lightgbm'] += 1

        # GNN (placeholder)
        g_engine = gnn_decision(sig, feats)
        totals['gnn'] += pt if g_engine == 'postgres' else dt
        counts['gnn'] += 1

    return totals, counts


def load_records(datasets=None, limit=None):
    files = []
    if datasets:
        for d in datasets:
            files.append(EXEC_DIR / f"{d}_unified_training_data.json")
    else:
        files = sorted(EXEC_DIR.glob('*_unified_training_data.json'))
    all_records = []
    for fp in files:
        if not fp.exists():
            continue
        try:
            data = json.load(open(fp, 'r'))
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")
            continue
        for rec in data:
            ok_pg = rec.get('executed_postgres', rec.get('postgres_time') is not None)
            ok_duck = rec.get('executed_duckdb', rec.get('duckdb_time') is not None)
            if not (ok_pg and ok_duck):
                continue
            all_records.append(rec)
            if limit and len(all_records) >= limit:
                break
        if limit and len(all_records) >= limit:
            break
    logger.info(f"Loaded {len(all_records)} executed records for evaluation")
    return all_records


def main():
    ap = argparse.ArgumentParser(description='Offline evaluation of routing methods using executed queries')
    ap.add_argument('--datasets', nargs='*', default=None, help='Datasets to include (default: all with execution data)')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of records')
    ap.add_argument('--threshold', type=float, default=10000.0, help='Cost threshold for cost_threshold method')
    ap.add_argument('--out', type=str, default=str(RESULTS_DIR / 'offline_routing_evaluation.json'))
    args = ap.parse_args()

    records = load_records(args.datasets, args.limit)
    if not records:
        logger.error('No executed records found to evaluate')
        return 1

    totals, counts = evaluate_methods(records, default_thresh=args.threshold)
    n = next(iter(counts.values())) if counts else 0

    # Build summary with total time and avg per query, and speedups vs default
    summary = {}
    for method in ['optimal', 'default', 'cost_threshold', 'lightgbm', 'gnn']:
        total_time = totals.get(method, 0.0)
        cnt = counts.get(method, 0)
        avg = (total_time / cnt) if cnt else float('inf')
        summary[method] = {
            'queries': cnt,
            'total_time_seconds': total_time,
            'avg_time_seconds': avg,
        }
    base = summary.get('default', {})
    base_total = base.get('total_time_seconds', None)
    if base_total and base_total > 0:
        for method in summary:
            summary[method]['speedup_vs_default'] = base_total / summary[method]['total_time_seconds'] if summary[method]['total_time_seconds'] > 0 else float('inf')

    # Save and print
    with open(args.out, 'w') as f:
        json.dump({'summary': summary, 'threshold': args.threshold, 'datasets': args.datasets}, f, indent=2)
    logger.info(f"Saved offline evaluation to {args.out}")

    print('\n=== Offline Routing Evaluation (executed queries) ===')
    print(f"Datasets: {args.datasets or 'ALL'} | Records: {sum(counts.values())//5}")
    for method in ['optimal', 'default', 'cost_threshold', 'lightgbm', 'gnn']:
        s = summary[method]
        print(f"- {method:15s} | total {s['total_time_seconds']:.2f}s | avg {s['avg_time_seconds']:.4f}s | speedup x{(s.get('speedup_vs_default') or 0):.3f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

