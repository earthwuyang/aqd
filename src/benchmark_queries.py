#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_queries.py  –  sequential execution of SQL queries under
different routing modes, measuring per-query latency (RT).

用法示例：
    python benchmark_queries.py \
        --dataset hybench_sf10 \
        --data_dir /home/wuy/query_costs \
        --timeout 600000 \
        --limit 50 \
        --out seq_bench.csv
"""

from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from typing import List

import pymysql
import pymysql.err
from tqdm import tqdm

# ───────────────────────────── config ──────────────────────────────
HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""

ROUTING_MODES = {
    "row_only":      ["SET use_imci_engine = OFF"],
    "col_only":      ["SET use_imci_engine = FORCED"],
    "cost_thresh": [
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = OFF",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "hybrid_opt": [
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "lgbm_kernel": [
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
}

# ─────────────────────────── utilities ─────────────────────────────
def load_queries(path: Path) -> List[str]:
    qs: List[str] = []
    with path.open() as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("--"):
                qs.append(s.rstrip(";").replace('"', ""))  # 仍保留去引号
    return qs


def log_fail(tag: str, idx: int, err: Exception, sql: str) -> None:
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    if isinstance(err, pymysql.MySQLError) and err.args:
        pass  # 打印太多会刷屏，如需调试可放开
        # print(f"[{tag}][#{idx}] errno {err.args[0]}: {err.args[1]} "
        #       f"\n    SQL: {snippet}", file=sys.stderr)
    else:
        print(f"[{tag}][#{idx}] {err}\n    SQL: {snippet}", file=sys.stderr)


# ───────────────────────── one routing mode ─────────────────────────
def run_mode(tag: str, queries: List[str], args):
    n = len(queries)
    lat: List[float | None] = [None] * n
    TIMEOUT_S = args.timeout / 1000.0

    bar = tqdm(total=n, desc=tag, ncols=70, leave=False)
    bench_start = time.perf_counter()

    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=args.dataset,
                               autocommit=True)
    except Exception as e:
        print(f"[{tag}] cannot connect to MySQL: {e}", file=sys.stderr)
        return 0.0, lat

    cur = conn.cursor()

    # 设置模式相关变量
    for s in ROUTING_MODES[tag]:
        try:
            cur.execute(s)
        except Exception as e:
            print(f"[{tag}] failed to execute '{s}': {e}", file=sys.stderr)

    cur.execute(f"SET max_execution_time = {args.timeout}")

    for idx, sql in enumerate(queries):
        t0 = time.perf_counter()
        try:
            cur.execute(sql)
            cur.fetchall()
            lat[idx] = time.perf_counter() - t0
        except pymysql.err.OperationalError as e:
            # 1317 = Query interrupted, 3024 = max_execution_time exceeded
            if e.args and e.args[0] in (1317, 3024):
                lat[idx] = TIMEOUT_S
            else:
                lat[idx] = None
                log_fail(tag, idx, e, sql)
        except Exception as e:
            lat[idx] = None
            log_fail(tag, idx, e, sql)
        bar.update()

    bar.close()
    cur.close()
    conn.close()

    makespan = time.perf_counter() - bench_start
    return makespan, lat


# ─────────────────────────── csv 结果输出 ────────────────────────────
def write_csv(path: Path, mode_res: dict):
    hdr = ["query_idx"] + [f"{m}_lat" for m in mode_res]
    n = len(next(iter(mode_res.values()))[1])
    rows = [[i] + [mode_res[m][1][i] for m in mode_res] for i in range(n)]
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)


# ────────────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hybench_sf1")
    ap.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    ap.add_argument("--ap")
    ap.add_argument("--tp")
    ap.add_argument("--timeout", type=int, default=600000,
                    help="per-query timeout (ms)")
    ap.add_argument("--limit", "-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="seq_bench.csv")
    args = ap.parse_args()

    # workload
    dset_dir = Path(args.data_dir) / "workloads" / args.dataset
    qs_ap = load_queries(Path(args.ap) if args.ap
                         else dset_dir / "workload_100k_s1_group_order_by_more_complex.sql")
    qs_tp = load_queries(Path(args.tp) if args.tp
                         else dset_dir / "TP_queries.sql")
    queries = qs_tp + qs_ap
    # queries = qs_ap
    random.Random(args.seed).shuffle(queries)
    if 0 < args.limit < len(queries):
        queries = queries[:args.limit]

    print(f"Loaded {len(queries)} queries "
          f"({len(qs_tp)} TP + {len(qs_ap)} AP)")

    mode_res = {}
    for tag in ROUTING_MODES:
        # 如果只想测部分模式可用 continue 筛掉
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, queries, args)

        ok   = [v for v in lats if v not in (None, args.timeout/1000.0)]
        to   = [v for v in lats if v == args.timeout/1000.0]
        fail = len(lats) - len(ok) - len(to)

        avg_rt = mean(ok + to) if ok or to else float("nan")

        print(f"makespan {mk:.2f}s  "
              f"avg_rt {avg_rt:.4f}s  "
              f"(ok {len(ok)} | timeout {len(to)} | fail {fail})")

        mode_res[tag] = (mk, lats)

    write_csv(Path(args.out), mode_res)
    print(f"\nPer-query latencies saved to {args.out}")


if __name__ == "__main__":
    main()
