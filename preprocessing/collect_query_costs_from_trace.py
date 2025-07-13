#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect latency + routing decisions (5 modes) and dump 24-dim features.

Author : you
Date   : 2025-07-13
"""

import os, csv, json, time, argparse, logging
from typing import Any, List

import numpy as np
import pymysql
from pymysql.err import InternalError, ProgrammingError
from tqdm import tqdm


# ────────────────────────── 0. Trace helpers ──────────────────────────
def _enable_trace(cur, mem_mb: int = 128) -> None:
    cur.execute(f"SET optimizer_trace_max_mem_size={mem_mb * 1024 * 1024 * 1024}")
    cur.execute("SET optimizer_trace='enabled=on,one_line=off'")


def _disable_trace(cur) -> None:
    cur.execute("SET optimizer_trace='enabled=off'")


def _fetch_trace(cur) -> str:
    """
    不再依赖 QUERY_ID：
    直接把整张表读进来，返回最后一条 TRACE（即当前 Session 最新）。
    """
    cur.execute("SELECT TRACE FROM INFORMATION_SCHEMA.OPTIMIZER_TRACE")
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("OPTIMIZER_TRACE is empty")
    return rows[-1][0]           # 最后一行即最新一条


def _extract_24feats(trace_obj: Any) -> List[float]:
    stack = [trace_obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "row_column_dispatch_features" in node:
                fn = node["row_column_dispatch_features"]
                return [fn[str(i)] for i in range(24)]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise KeyError("row_column_dispatch_features not found")


# ────────────────────────── 1. Session config ────────────────────────
def _apply_mode(cur, mode: str, timeout_ms: int) -> None:
    cur.execute(f"SET max_execution_time={timeout_ms}")

    if mode == "row_only":
        cur.execute("SET use_imci_engine=off")

    elif mode == "col_only":
        cur.execute("SET use_imci_engine=forced")

    elif mode in ("cost_threshold", "hybrid", "fann"):
        cur.execute("SET use_imci_engine=on")
        cur.execute("SET cost_threshold_for_imci=1")
        cur.execute("SET imci_optimizer_switch='fast_opt_trivial_query=off'")
        cur.execute("SET imci_auto_update_statistic='SYNC'")
        cur.execute("SET hybrid_opt_compatible_transform_switch=4095")
        cur.execute("SET global hybrid_opt_fetch_imci_stats_thread_enabled=on")

        if mode == "cost_threshold":
            cur.execute("SET cost_threshold_for_imci=50000")   # ← 5 万阈值
            cur.execute("SET hybrid_opt_dispatch_enabled=off")
            cur.execute("SET fann_model_routing_enabled=off")

        elif mode == "hybrid":
            cur.execute("SET hybrid_opt_dispatch_enabled=on")
            cur.execute("SET fann_model_routing_enabled=off")

        elif mode == "fann":
            cur.execute("SET hybrid_opt_dispatch_enabled=on")
            cur.execute("SET fann_model_routing_enabled=on")
            _enable_trace(cur)
    else:
        raise ValueError(f"unknown mode {mode}")


# ─────────────────────────── 2. Run one mode ─────────────────────────
def _run_mode(conn, sql: str, mode: str, timeout_ms: int):
    cur = conn.cursor()
    _apply_mode(cur, mode, timeout_ms)

    t0 = time.time()
    if mode in ("row_only", "col_only"):
        cur.execute(sql);  cur.fetchall()
    else:
        cur.execute(f"EXPLAIN FORMAT='json' {sql}");  cur.fetchall()
    latency = time.time() - t0

    imci_flag = int(mode == "col_only") if mode in ("row_only", "col_only") else 0
    feats = None

    if mode == "fann":
        trace_str = _fetch_trace(cur)
        feats = _extract_24feats(json.loads(trace_str))
        _disable_trace(cur)

    cur.close()
    return imci_flag, latency, feats


# ────────────────────────────── 3. main ──────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset",   default="tpch_sf1")
    pa.add_argument("--data_dir",  default="/home/wuy/query_costs_trace/")
    pa.add_argument("--db",        default="tpch_sf1")
    pa.add_argument("--port",      type=int, default=44444)
    pa.add_argument("--timeout",   type=int, default=60000)
    pa.add_argument("--debug",     action="store_true")
    args = pa.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    root = os.path.join(args.data_dir, args.dataset)
    os.makedirs(root, exist_ok=True)
    fp_costs = os.path.join(root, "query_costs.csv")
    fp_feats = os.path.join(root, "features_24d.csv")

    # workload
    queries: List[str] = []
    for file in ("workload_100k_s1_group_order_by_more_complex.sql",
                 "TP_queries.sql"):
        path = os.path.join(args.data_dir, "workloads", args.dataset, file)
        with open(path) as f:
            queries += [ln.strip().replace('"', "")
                        for ln in f if ln.strip() and not ln.startswith("#")]
    np.random.shuffle(queries)

    conn = pymysql.connect(host="127.0.0.1", user="root",
                           port=args.port, db=args.db)

    # CSV writers
    w_cost = csv.writer(open(fp_costs, "a", newline=""))
    w_feat = csv.writer(open(fp_feats, "a", newline=""))
    if os.stat(fp_costs).st_size == 0:
        w_cost.writerow(["query_id", "use_imci",
                         "row_time", "column_time",
                         "cost_use_imci", "hybrid_use_imci", "fann_use_imci",
                         "query"])
    if os.stat(fp_feats).st_size == 0:
        w_feat.writerow(["query_id"] + [f"f{i:02d}" for i in range(24)])

    for qid, raw in tqdm(list(enumerate(queries)), total=len(queries)):
        sql = raw.rstrip(";") + ";"
        try:
            row_flag, row_t, _   = _run_mode(conn, sql, "row_only", args.timeout)
            col_flag, col_t, _   = _run_mode(conn, sql, "col_only", args.timeout)
            cost_flag, _, _      = _run_mode(conn, sql, "cost_threshold", args.timeout)
            hyb_flag, _, _       = _run_mode(conn, sql, "hybrid", args.timeout)
            fann_flag, _, feats  = _run_mode(conn, sql, "fann", args.timeout)

            if row_t is None and col_t is None:
                continue
            truth = 1 if row_t is None else 0 if col_t is None else int(col_t < row_t)

            w_cost.writerow([qid, truth,
                             row_t, col_t,
                             cost_flag, hyb_flag, fann_flag,
                             sql.replace("\n", " ")])
            if feats is not None:
                w_feat.writerow([qid] + feats)

        except Exception as e:
            logging.error(f"[{qid}] failed: {e}")

    conn.close()


if __name__ == "__main__":
    main()
