#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py  –  Poisson arrivals, one-thread-per-query.
Latency = 查询提交到 MySQL 后直至结果返回（含 MySQL 内部排队）。
"""

from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from threading import Thread, Lock
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
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "lgbm_kernel": [
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
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


def poisson_arrivals(n: int, mean_sec: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        out.append(t)
    return out


def log_fail(tag: str, idx: int, err: Exception, sql: str) -> None:
    """把失败原因打印到 stderr；缩短 SQL 片段避免刷屏"""
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    if isinstance(err, pymysql.MySQLError) and err.args:
        pass
        # print(f"[{tag}][#{idx}] errno {err.args[0]}: {err.args[1]}\n"
        #       f"    SQL: {snippet}",
        #       file=sys.stderr)
    else:
        print(f"[{tag}][#{idx}] {err}\n"
              f"    SQL: {snippet}",
              file=sys.stderr)


# ───────────────────── per-query execution thread ───────────────────
def execute_query(idx: int,
                  sql: str,
                  arrival_rel: float,
                  sess_sql: list[str],
                  args,
                  lat: list,
                  bench_start: float,
                  bar,
                  lock: Lock,
                  tag: str) -> None:
    TIMEOUT_S = args.timeout / 1000.0

    # 1) sleep 到到达时刻
    target = bench_start + arrival_rel
    remain = target - time.perf_counter()
    if remain > 0:
        time.sleep(remain)

    # 2) 建立连接并执行
    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=args.dataset,
                               autocommit=True)
        cur = conn.cursor()
        for s in sess_sql:
            cur.execute(s)
        cur.execute(f"SET max_execution_time = {args.timeout}")
    except Exception as e:
        log_fail(tag, idx, e, sql)
        with lock:
            lat[idx] = None
            bar.update()
        return

    t0 = time.perf_counter()
    try:
        cur.execute(sql)
        cur.fetchall()
        with lock:
            lat[idx] = time.perf_counter() - t0
    except pymysql.err.OperationalError as e:
        if e.args and e.args[0] in (3024, 1317):      # 超时
            with lock:
                lat[idx] = TIMEOUT_S
        else:                                         # 其它 SQL 错误
            log_fail(tag, idx, e, sql)
            with lock:
                lat[idx] = None
    except Exception as e:                            # 连接 / 网络等
        log_fail(tag, idx, e, sql)
        with lock:
            lat[idx] = None
    finally:
        with lock:
            bar.update()
        cur.close()
        conn.close()


# ───────────────────────── one routing mode ─────────────────────────
def run_mode(tag: str,
             queries: List[str],
             arrivals: List[float],
             args):
    n = len(queries)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=70, leave=False)
    bench_start = time.perf_counter()

    threads: List[Thread] = []
    lock = Lock()

    for i, (sql, rel) in enumerate(zip(queries, arrivals)):
        th = Thread(target=execute_query,
                    args=(i, sql, rel,
                          ROUTING_MODES[tag],
                          args, lat, bench_start, bar, lock, tag),
                    daemon=True)
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    bar.close()
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
    ap.add_argument("--dataset", default="hybench_sf10")
    ap.add_argument("--data_dir", default="/home/wuy/query_costs/")
    ap.add_argument("--ap")
    ap.add_argument("--tp")
    ap.add_argument("--timeout", type=int, default=600000,
                    help="per-query timeout (ms)")
    ap.add_argument("--limit", "-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mean_interval", type=float, default=5,
                    help="平均到达间隔 (ms)")
    ap.add_argument("--qps", type=float,
                    help="目标 QPS（设后覆盖 mean_interval）")
    ap.add_argument("--out", default="routing_bench.csv")
    args = ap.parse_args()

    mean_ms  = 1000 / args.qps if args.qps else args.mean_interval
    mean_sec = mean_ms / 1000.0

    # workload
    dset_dir = Path(args.data_dir) / "workloads" / args.dataset
    qs_ap = load_queries(Path(args.ap) if args.ap
                         else dset_dir / "workload_100k_s1_group_order_by_more_complex.sql")
    qs_tp = load_queries(Path(args.tp) if args.tp
                         else dset_dir / "TP_queries.sql")
    # queries = qs_tp + qs_ap
    queries = qs_ap
    random.Random(args.seed).shuffle(queries)
    if 0 < args.limit < len(queries):
        queries = queries[:args.limit]

    print(f"Loaded {len(queries)} queries "
          f"({len(qs_tp)} TP + {len(qs_ap)} AP)")

    arrivals = poisson_arrivals(len(queries), mean_sec, args.seed)

    mode_res = {}
    for tag in ROUTING_MODES:
        if tag == "row_only" or tag == "col_only":
            continue
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, queries, arrivals, args)

        ok      = [v for v in lats if v not in (None, args.timeout/1000.0)]
        to      = [v for v in lats if v == args.timeout/1000.0]
        fail    = len(lats) - len(ok) - len(to)
        avg_lat = mean(ok + to) if ok or to else float("nan")
        qps_eff = len(ok) / mk if mk else 0.0

        print(f"makespan {mk:.2f}s  "
              f"avg {avg_lat:.4f}s  "
              f"qps {qps_eff:.2f}/s  "
              f"(ok {len(ok)} | timeout {len(to)} | fail {fail})")

        mode_res[tag] = (mk, lats)

    write_csv(Path(args.out), mode_res)
    print(f"\nPer-query latencies saved to {args.out}")


if __name__ == "__main__":
    main()
