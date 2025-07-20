#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_tpcds_sf100.py  –  Poisson arrivals, concurrent execution, TPC-DS SF100.

• 自动遍历 --sql_dir 下的 *.sql，将文件里的多条查询全部拆分出来。
• 对每条查询随机生成泊松到达时间，在 3 种核心 ROUTING_MODES 下并发执行
  （cost_thresh / hybrid_opt / lgbm_kernel）。
• 结果写 CSV：qid, cost_threshold_time, hybrid_optimizer_time, lightgbm_time, sql_statement。
  timeout 记为 args.timeout/1000。
"""

from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from threading import Thread, Lock
from typing import List, Optional, Tuple

import pymysql
import pymysql.err
from tqdm import tqdm

# ───────────────────────────── config ──────────────────────────────
HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""

ROUTING_MODES = {
    # 仅保留需要写 CSV 的三种模式；其余模式可按需添加/测试
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
    # "lgbm_kernel_mm1": [
    #     "SET use_imci_engine = ON",
    #     "SET cost_threshold_for_imci = 1",
    #     "SET hybrid_opt_dispatch_enabled = ON",
    #     "SET fann_model_routing_enabled  = ON",
    #     "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    #     "SET use_mm1_time = ON"
    # ],
}
# 写 CSV 时的列顺序
BENCH_MODES = ["cost_thresh", "hybrid_opt", "lgbm_kernel"]
# BENCH_MODES = ["lgbm_kernel_mm1"]

# ─────────────────────────── utilities ─────────────────────────────
def split_sql(text: str) -> List[str]:
    """按分号切分 SQL 文件，忽略行注释。"""
    stmts, buff = [], []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("--") or not ln:
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
    """读取目录下所有 .sql，打乱并返回指定数量。"""
    qs: List[str] = []
    for p in sorted(sql_dir.glob("*.sql")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        qs.extend(split_sql(text))

    # 可按需过滤 /with/ 查询
    qs = [q for q in qs if "with" not in q.lower()]

    rng = random.Random(seed)
    rng.shuffle(qs)
    return qs[:limit] if (limit and limit > 0) else qs


def poisson_arrivals(n: int, mean_sec: float, rng: random.Random) -> List[float]:
    t, arr = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        arr.append(t)
    return arr


def log_fail(tag: str, idx: int, err: Exception, sql: str) -> None:
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    if isinstance(err, pymysql.MySQLError):
        pass  # 可按需打印 errno
    else:
        print(f"[{tag} #{idx}] {err}\n   SQL: {snippet}", file=sys.stderr)

# ───────────────────── per-query execution thread ───────────────────
def execute_query(idx: int,
                  sql: str,
                  db: str,
                  arrival: float,
                  sess_sql: List[str],
                  args,
                  lat: List,
                  bench_start: float,
                  bar,
                  lock: Lock,
                  tag: str) -> None:
    TIMEOUT_S = args.timeout / 1000.0

    # 1) 等待到达
    remain = bench_start + arrival - time.perf_counter()
    if remain > 0:
        time.sleep(remain)

    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=db, autocommit=True)
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
        if e.args and e.args[0] in (3024, 1317):      # 3024 = timeout
            with lock:
                lat[idx] = TIMEOUT_S
        else:
            log_fail(tag, idx, e, sql)
            with lock:
                lat[idx] = None
    except Exception as e:
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
             db: str,
             args) -> Tuple[float, List[Optional[float]]]:
    n = len(queries)
    lat: List[Optional[float]] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=72, leave=False)
    bench_start = time.perf_counter()

    lock = Lock()
    threads: List[Thread] = []
    for i, (q, arr) in enumerate(zip(queries, arrivals)):
        th = Thread(
            target=execute_query,
            args=(i, q, db, arr,
                  ROUTING_MODES[tag],
                  args, lat, bench_start, bar, lock, tag),
            daemon=True
        )
        threads.append(th)
        th.start()

    for th in threads:
        th.join()
    bar.close()
    return time.perf_counter() - bench_start, lat

# ─────────────────────────── csv 输出 ────────────────────────────
def write_csv(path: Path, queries: List[str], mode_res: dict) -> None:
    hdr = ["qid",
           "cost_threshold_time",
           "hybrid_optimizer_time",
           "lightgbm_time",
           "sql_statement"]
    rows = []
    n = len(queries)
    for i in range(n):
        row = [i]
        for m in BENCH_MODES:
            row.append(mode_res[m][1][i] if m in mode_res else None)
        row.append(queries[i])
        rows.append(row)

    with path.open("w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)

# ──────────────────────────── main ────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql_dir",
                    default="/home/wuy/datasets/tpcds-kit/tpcds-kit/tools/queries",
                    help="生成的 TPC-DS .sql 文件所在目录")
    ap.add_argument("--db", default="tpcds_sf100",
                    help="数据库名（已加载好 SF100 数据）")
    ap.add_argument("--limit", "-n", type=int, default=100,
                    help="抽取前 n 条查询")
    ap.add_argument("--timeout", type=int, default=600000,
                    help="per-query timeout (ms)")
    ap.add_argument("--mean_interval", type=float, default=50,
                    help="泊松平均到达间隔 (ms)")
    ap.add_argument("--qps", type=float,
                    help="目标 QPS（设置后覆盖 mean_interval）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tpcds_bench.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mean_ms  = 1000 / args.qps if args.qps else args.mean_interval
    mean_sec = mean_ms / 1000.0

    sql_dir = Path(args.sql_dir)
    if not sql_dir.exists():
        print(f"--sql_dir {sql_dir} 不存在", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(sql_dir, args.limit, seed=args.seed)
    if not queries:
        print("没有读到任何查询，请确认 --sql_dir 里有 .sql 文件", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(queries)} queries from {sql_dir} (db = {args.db})")

    arrivals = poisson_arrivals(len(queries), mean_sec, rng)

    # ── benchmark 各 routing-mode ───────────────────────────────────
    mode_res = {}
    for tag in BENCH_MODES:
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, queries, arrivals, args.db, args)

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

    # ── 输出 CSV ───────────────────────────────────────────────────
    write_csv(Path(args.out), queries, mode_res)
    print(f"\nPer-query latencies saved to {args.out}")


if __name__ == "__main__":
    main()
