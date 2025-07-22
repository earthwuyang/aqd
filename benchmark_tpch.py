#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_tpch.py – Poisson arrivals, concurrent execution, TPC-H.

• 目录 --sql_dir 下每个 .sql 文件 == 1 条查询（文件名 qX_Y.sql）。
• 加载所有文件 → 洗牌(可复现) → 截取 --limit 条。
• 按泊松到达间隔并发执行；比较多种 ROUTING_MODES。
• 输出 CSV：每列 = 一种路由模式的延迟 (秒)，timeout 记 args.timeout/1000。
"""

from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from threading import Thread, Lock
from typing import List, Optional

import pymysql
import pymysql.err
from tqdm import tqdm

# ─────────────────────── config ─────────────────────────
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
    "lgbm_kernel_mm1": [
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = ON"
    ]
}

# ────────────────── utilities ────────────────────────────
def read_single_query(file_path: Path) -> str:
    """读取文件，去掉 '--' 注释行，返回一条 SQL（末尾不带 ';'）。"""
    lines = [
        ln.strip() for ln in file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip() and not ln.lstrip().startswith("--")
    ]
    sql = " ".join(lines)
    if sql.endswith(";"):
        sql = sql[:-1]
    return sql

def load_queries(sql_dir: Path,
                 limit: Optional[int] = None,
                 seed: Optional[int] = None) -> List[str]:
    """目录中每个 .sql → 1 查询；整体 shuffle，再按 limit 截取。"""
    files = sorted(sql_dir.glob("*.sql"))
    queries = [read_single_query(p) for p in files]
    rng = random.Random(seed)
    rng.shuffle(queries)
    return queries[:limit] if (limit and limit > 0) else queries

def poisson_arrivals(n: int, mean_sec: float, rng: random.Random) -> List[float]:
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        out.append(t)
    return out

def log_fail(tag: str, idx: int, err: Exception) -> None:
    if isinstance(err, pymysql.MySQLError) and err.args:
        # 静默语法/超时等错误，避免刷屏；如需调试可取消注释
        # print(f"[{tag} #{idx}] errno {err.args[0]}: {err.args[1]}", file=sys.stderr)
        pass
    else:
        print(f"[{tag} #{idx}] {err}", file=sys.stderr)

# ───────────── per-query execution thread ───────────────
def execute_query(idx: int,
                  sql: str,
                  db: str,
                  arrival: float,
                  sess_sql: list[str],
                  args,
                  lat: list,
                  bench_start: float,
                  bar,
                  lock: Lock,
                  tag: str) -> None:
    TIMEOUT_S = args.timeout / 1000.0
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
        log_fail(tag, idx, e)
        with lock:
            lat[idx] = None; bar.update()
        return

    t0 = time.perf_counter()
    try:
        cur.execute(sql)
        cur.fetchall()
        with lock:
            lat[idx] = time.perf_counter() - t0
    except pymysql.err.OperationalError as e:
        if e.args and e.args[0] in (3024, 1317):          # timeout / killed
            with lock:
                lat[idx] = TIMEOUT_S
        else:
            log_fail(tag, idx, e)
            with lock:
                lat[idx] = None
    except Exception as e:
        log_fail(tag, idx, e)
        with lock:
            lat[idx] = None
    finally:
        with lock:
            bar.update()
        cur.close(); conn.close()

# ─────────────── run one mode ───────────────────────────
def run_mode(tag: str,
             queries: List[str],
             arrivals: List[float],
             db: str,
             args):
    n = len(queries)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=72, leave=False)
    bench_start = time.perf_counter()

    threads, lock = [], Lock()
    for i, (q, arr) in enumerate(zip(queries, arrivals)):
        th = Thread(target=execute_query,
                    args=(i, q, db, arr, ROUTING_MODES[tag],
                          args, lat, bench_start, bar, lock, tag),
                    daemon=True)
        th.start(); threads.append(th)
    for th in threads:
        th.join()

    bar.close()
    return time.perf_counter() - bench_start, lat

# ─────────────── csv helper ─────────────────────────────
def write_csv(path: Path, mode_res: dict):
    hdr = ["query_idx"] + [f"{m}_lat" for m in mode_res]
    n = len(next(iter(mode_res.values()))[1])
    rows = [[i] + [mode_res[m][1][i] for m in mode_res] for i in range(n)]
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)

# ──────────────────── main ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql_dir", default="/home/wuy/datasets/tpch-dbgen/queries_sf1",
                    help="目录中每个 .sql 文件即一条查询")
    ap.add_argument("--db", default="tpch_sf1",
                    help="目标数据库 (已加载同规模数据)")
    ap.add_argument("--limit", "-n", type=int, default=200,
                    help="随机抽取前 n 条查询")
    ap.add_argument("--timeout", type=int, default=600000,
                    help="per-query timeout (ms)")
    ap.add_argument("--mean_interval", type=float, default=50,
                    help="Poisson 平均到达间隔 (ms)")
    ap.add_argument("--qps", type=float,
                    help="目标 QPS (设置后覆盖 mean_interval)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tpch_bench.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mean_ms  = 1000 / args.qps if args.qps else args.mean_interval
    mean_sec = mean_ms / 1000.0

    sql_dir = Path(args.sql_dir)
    if not sql_dir.exists():
        print(f"--sql_dir {sql_dir} 不存在", file=sys.stderr); sys.exit(1)

    queries = load_queries(sql_dir, args.limit, seed=args.seed)
    if not queries:
        print("目录中无可用查询", file=sys.stderr); sys.exit(1)
    print(f"Loaded {len(queries)} queries from {sql_dir} (db={args.db})")

    arrivals = poisson_arrivals(len(queries), mean_sec, rng)

    mode_res = {}
    for tag in ROUTING_MODES:
        if tag in ("row_only", "col_only"):   # 如需 baseline 去掉本行
            continue
        if not tag.startswith('lightgbm'):
            continue
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, queries, arrivals, args.db, args)

        ok   = [v for v in lats if v not in (None, args.timeout/1000.0)]
        to   = [v for v in lats if v == args.timeout/1000.0]
        fail = len(lats) - len(ok) - len(to)
        avg  = mean(ok + to) if ok or to else float("nan")
        qps_eff = len(ok) / mk if mk else 0.0

        print(f"makespan {mk:.2f}s  avg {avg:.4f}s  "
              f"qps {qps_eff:.2f}/s  (ok {len(ok)} | timeout {len(to)} | fail {fail})")
        mode_res[tag] = (mk, lats)

    write_csv(Path(args.out), mode_res)
    print(f"\nPer-query latencies saved to {args.out}")

if __name__ == "__main__":
    main()
