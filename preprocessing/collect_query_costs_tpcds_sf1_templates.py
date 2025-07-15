#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_query_costs_tpcds_templates.py
--------------------------------------
• 针对 TPC-DS 模板生成的查询，在 5 种 ROUTING 模式下执行 / EXPLAIN，
  采集真实行/列延迟、各模式决策，以及 24-维 dispatch 特征。
• 另解析 optimizer `query_cost` → 作为第 4 列写入 CSV。
  输出：
      query_costs_traces/tpcds_sf1_templates/query_costs.csv
      query_costs_traces/tpcds_sf1_templates/features_24d.csv
"""

import os, csv, json, time, argparse, logging, random
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pymysql
from pymysql.err import OperationalError
from tqdm import tqdm


# ────────────────────── 0. 读取 TPC-DS 查询 ───────────────────────
def split_sql(text: str) -> List[str]:
    stmts, buff = [], []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("--"):
            continue
        buff.append(ln)
        if ln.endswith(";"):
            stmts.append(" ".join(buff)[:-1])   # 去掉末尾分号
            buff.clear()
    if buff:
        stmts.append(" ".join(buff))
    return stmts


def load_queries(sql_dir: Path,
                 limit: Optional[int] = None,
                 seed: Optional[int] = None) -> List[str]:
    qs: List[str] = []
    for p in sorted(sql_dir.glob("*.sql")):
        qs.extend(split_sql(p.read_text(encoding="utf-8", errors="ignore")))
    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(qs)
    return qs[:limit] if limit and limit > 0 else qs


# ────────────────── 1. Trace / Feature 工具函数 ───────────────────
def _enable_trace(cur, mem_mb: int = 128) -> None:
    cur.execute(f"SET optimizer_trace_max_mem_size={mem_mb * 1024 * 1024}")
    cur.execute("SET optimizer_trace='enabled=on,one_line=off'")


def _disable_trace(cur) -> None:
    cur.execute("SET optimizer_trace='enabled=off'")


def _fetch_trace(cur) -> str:
    cur.execute("SELECT TRACE FROM INFORMATION_SCHEMA.OPTIMIZER_TRACE")
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("OPTIMIZER_TRACE is empty")
    return rows[-1][0]           # 最新一条


def _extract_24feats(trace_obj: Any) -> List[float]:
    stack = [trace_obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "row_column_dispatch_features" in node:
                feats = node["row_column_dispatch_features"]
                return [feats[str(i)] for i in range(24)]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise KeyError("row_column_dispatch_features not found")


# ────────────────── 2. 会话级参数（5 种模式） ───────────────────
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
            cur.execute("SET cost_threshold_for_imci=50000")
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


# ────────────────────── 3. 执行 / 解释一条 SQL ────────────────────
def _run_mode(conn, sql: str, mode: str, timeout_ms: int
              ) -> Tuple[int, Optional[float], Optional[List[float]], Optional[float]]:
    """
    返回 (imci_flag, latency, feats24, qcost)
      • row_only / col_only 不填 feats、qcost
      • cost_threshold / hybrid / fann 解析 EXPLAIN，并给出 qcost
      • 仅 fann 模式抓取 24-维特征
    """
    cur = conn.cursor()
    _apply_mode(cur, mode, timeout_ms)

    latency: Optional[float] = None
    feats:   Optional[List[float]] = None
    qcost:   Optional[float] = None

    try:
        if mode in ("row_only", "col_only"):
            t0 = time.time()
            cur.execute(sql);  cur.fetchall()
            latency = time.time() - t0
        else:
            t0 = time.time()
            cur.execute(f"EXPLAIN FORMAT='json' {sql}")
            plan_str = cur.fetchall()[0][0]
            latency = time.time() - t0

            # 解析 optimizer cost
            try:
                plan = json.loads(plan_str)
                qcost = plan.get("query_block", {}) \
                            .get("cost_info", {})  \
                            .get("query_cost", 0.0)
            except Exception:
                qcost = 0.0

            if mode == "fann":
                trace_str = _fetch_trace(cur)
                feats = _extract_24feats(json.loads(trace_str))
                _disable_trace(cur)
    except OperationalError:
        latency = None
    finally:
        cur.close()

    imci_flag = int(mode == "col_only") if mode in ("row_only", "col_only") else 0
    return imci_flag, latency, feats, qcost


# ──────────────────────────── 4. main ───────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--sql_dir",
                    default="/home/wuy/datasets/tpcds-kit/tpcds-kit/tools/queries",
                    help="TPC-DS 模板生成 .sql 文件目录")
    pa.add_argument("--data_dir",  default="/home/wuy/query_costs_trace/")
    pa.add_argument("--dataset",   default="tpcds_sf1_templates")
    pa.add_argument("--db",        default="tpcds_sf1")
    pa.add_argument("--port",      type=int, default=44444)
    pa.add_argument("--timeout",   type=int, default=60000)
    pa.add_argument("--limit",     type=int, help="前 N 条查询", default=100000)
    pa.add_argument("--seed",      type=int, default=42)
    pa.add_argument("--debug",     action="store_true")
    args = pa.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # ── 读取查询 ───────────────────────────────────────────────
    qlist = load_queries(Path(args.sql_dir), args.limit, args.seed)
    if not qlist:
        print(f"No queries found under {args.sql_dir}")
        return
    print(f"Loaded {len(qlist)} queries")

    # ── 输出目录 / CSV ─────────────────────────────────────────
    root = os.path.join(args.data_dir, args.dataset)
    os.makedirs(root, exist_ok=True)
    fp_costs = os.path.join(root, "query_costs.csv")
    fp_feats = os.path.join(root, "features_24d.csv")

    cost_csv = open(fp_costs, "a", newline="")
    feat_csv = open(fp_feats, "a", newline="")
    w_cost = csv.writer(cost_csv)
    w_feat = csv.writer(feat_csv)

    if cost_csv.tell() == 0:
        w_cost.writerow(["query_id", "use_imci",
                         "row_time", "column_time", "qcost",
                         "cost_use_imci", "hybrid_use_imci", "fann_use_imci",
                         "query"])
    if feat_csv.tell() == 0:
        w_feat.writerow(["query_id"] + [f"f{i:02d}" for i in range(24)])

    # ── MySQL 连接 ────────────────────────────────────────────
    conn = pymysql.connect(host="127.0.0.1", user="root",
                           port=args.port, db=args.db)

    # ── 主循环 ───────────────────────────────────────────────
    for qid, raw_sql in tqdm(list(enumerate(qlist)), ncols=72):
        sql = raw_sql.rstrip(";") + ";"
        try:
            row_flag, row_t, _, _        = _run_mode(conn, sql, "row_only", args.timeout)
            col_flag, col_t, _, _        = _run_mode(conn, sql, "col_only", args.timeout)
            cost_flag, _,  _, qcost      = _run_mode(conn, sql, "cost_threshold", args.timeout)
            hyb_flag,  _,  _, _          = _run_mode(conn, sql, "hybrid", args.timeout)
            fann_flag, _,  feats, _      = _run_mode(conn, sql, "fann", args.timeout)

            if row_t is None and col_t is None:
                continue
            truth = 1 if row_t is None else 0 if col_t is None else int(col_t < row_t)
            qcost = qcost or 0.0          # 若解析失败给 0

            w_cost.writerow([qid, truth,
                             row_t, col_t, qcost,
                             cost_flag, hyb_flag, fann_flag,
                             sql.replace("\n", " ")])
            cost_csv.flush()
            os.fsync(cost_csv.fileno())
            if feats is not None:
                w_feat.writerow([qid] + feats)
                feat_csv.flush()
                os.fsync(feat_csv.fileno())

        except Exception as e:
            logging.error(f"[{qid}] failed: {e}")

    conn.close()
    cost_csv.close()
    feat_csv.close()
    print(f"Done. CSV files written under {root}")


if __name__ == "__main__":
    main()
