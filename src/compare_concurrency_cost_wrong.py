#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py  –  multiple arrival patterns, one-thread-per-query.

1. 读取 query_costs.csv 中 “cost_use_imci ≠ use_imci” 的 SQL。
2. 支持三种到达模型：
      • 默认 Poisson (`--poisson_qps`)
      • 爆发 Burst   (`--burst` …)
      • 锯齿 Sawtooth (`--saw`  …)
3. 并发执行多种 ROUTING_MODES（cost_thresh / hybrid_opt / lgbm_kernel / lgbm_kernel_mm1）。
4. 结果输出 CSV：query_idx, <mode>_lat …
"""

from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from threading import Thread, Lock
from typing import List, Tuple

import pymysql
import pymysql.err
from tqdm import tqdm

# ───────────────────────────── config ──────────────────────────────
HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""

ALL_DATASETS = [
    "tpch_sf1",  "tpch_sf10",  "tpch_sf100",
    "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
    "hybench_sf1", "hybench_sf10",
    "airline", "credit", "carcinogenesis",
    "employee", "financial", "geneea", "hepatitis",
]

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
    ],
}

# ─────────────────────────── utilities ─────────────────────────────
def load_mismatched_from_csv(path: Path) -> List[str]:
    """从 query_costs.csv 读取 (cost_use_imci ≠ use_imci) 的 SQL 列表"""
    if not path.exists():
        return []
    qs: List[str] = []
    with path.open(newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                use = int(row["use_imci"])
                cost = int(row["cost_use_imci"])
            except (KeyError, ValueError):
                continue
            if use != cost:
                qs.append(row["query"].rstrip(";").replace('"', ""))
    return qs


# ---------- arrival patterns ----------
def poisson_arrivals(n: int, mean_sec: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        out.append(t)
    return out


def burst_arrivals(n: int,
                   burst_qps: float,
                   pause_ms: float,
                   burst_len: int,
                   seed: int) -> List[float]:
    """爆发: 连续 burst_len 条查询以 burst_qps 速率到达，然后空档 pause_ms。"""
    rng = random.Random(seed)
    arrivals: list[float] = []
    t = 0.0
    intra = 1.0 / burst_qps
    while len(arrivals) < n:
        for _ in range(min(burst_len, n - len(arrivals))):
            arrivals.append(t)
            t += intra
        t += pause_ms / 1000.0
        rng.random()  # 抖动 RNG
    return arrivals


def sawtooth_arrivals(n: int,
                      base_qps: float,
                      peak_qps: float,
                      period_ms: float,
                      seed: int) -> List[float]:
    """锯齿: QPS 线性上升到峰值再骤降，再循环。"""
    rng = random.Random(seed)
    arrivals: list[float] = []
    t = 0.0
    step_cnt = 20                        # 一齿分 20 个线性段
    while len(arrivals) < n:
        for frac in (i / step_cnt for i in range(step_cnt + 1)):
            cur_qps = base_qps + (peak_qps - base_qps) * frac
            dt = 1.0 / cur_qps
            seg_len = period_ms / 1000.0 / step_cnt
            num = max(1, int(seg_len / dt))
            for _ in range(min(num, n - len(arrivals))):
                arrivals.append(t)
                t += dt
            if len(arrivals) >= n:
                break
    arrivals = [a + rng.uniform(-0.5, 0.5) * 1e-3 for a in arrivals]
    arrivals.sort()
    return arrivals


def log_fail(tag: str, idx: int, err: Exception, sql: str):
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    print(f"[{tag} #{idx}] {err}\n   SQL: {snippet}", file=sys.stderr)

# ───────────────────── per-query execution thread ───────────────────
def execute_query(idx: int,
                  task: Tuple[str, str],
                  arrival_rel: float,
                  sess_sql: list[str],
                  args,
                  lat: list,
                  bench_start: float,
                  bar, lock: Lock,
                  tag: str) -> None:
    db, sql = task
    TIMEOUT_S = args.timeout / 1000.0

    # wait until its arrival time
    remain = bench_start + arrival_rel - time.perf_counter()
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
            lat[idx] = None;  bar.update()
        return

    t0 = time.perf_counter()
    try:
        cur.execute(sql);  cur.fetchall()
        with lock:
            lat[idx] = time.perf_counter() - t0
    except pymysql.err.OperationalError as e:
        if e.args and e.args[0] in (3024, 1317):          # timeout
            with lock: lat[idx] = TIMEOUT_S
        else:
            log_fail(tag, idx, e, sql)
            with lock: lat[idx] = None
    except Exception as e:
        log_fail(tag, idx, e, sql)
        with lock: lat[idx] = None
    finally:
        with lock: bar.update()
        cur.close();  conn.close()

# ───────────────────────── one routing mode ─────────────────────────
def run_mode(tag: str,
             tasks: List[Tuple[str, str]],
             arrivals: List[float],
             args):
    n = len(tasks)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=70, leave=False)
    bench_start = time.perf_counter()

    threads: List[Thread] = []
    lock = Lock()
    for i, (job, rel) in enumerate(zip(tasks, arrivals)):
        th = Thread(target=execute_query,
                    args=(i, job, rel,
                          ROUTING_MODES[tag],
                          args, lat, bench_start, bar, lock, tag),
                    daemon=True)
        threads.append(th); th.start()
    for th in threads: th.join()

    bar.close()
    return time.perf_counter() - bench_start, lat

# ─────────────────────────── csv 输出 ────────────────────────────
def write_csv(path: Path, mode_res: dict):
    hdr = ["query_idx"] + [f"{m}_lat" for m in mode_res]
    n = len(next(iter(mode_res.values()))[1])
    rows = [[i] + [mode_res[m][1][i] for m in mode_res] for i in range(n)]
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)

# ──────────────────────────── main ────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", help="单库；省略则遍历全部数据集")
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--timeout", type=int, default=600000,
                    help="per-query timeout (ms)")
    pa.add_argument("--limit", "-n", type=int, default=100)
    pa.add_argument("--seed", type=int, default=42)

    # arrival pattern (互斥)
    grp = pa.add_mutually_exclusive_group()
    grp.add_argument("--poisson_qps", type=float,
                     help="Poisson 到达的目标 QPS")
    grp.add_argument("--burst", action="store_true",
                     help="使用 burst 到达模式")
    grp.add_argument("--saw", action="store_true",
                     help="使用锯齿到达模式")

    # burst parameters
    pa.add_argument("--burst_qps", type=float, default=800)
    pa.add_argument("--pause_ms", type=float, default=200)
    pa.add_argument("--burst_len", type=int, default=200)

    # sawtooth parameters
    pa.add_argument("--base_qps", type=float, default=50)
    pa.add_argument("--peak_qps", type=float, default=800)
    pa.add_argument("--period_ms", type=float, default=500)

    pa.add_argument("--out", default="routing_bench.csv")
    args = pa.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # ── 1. 收集 SQL ───────────────────────────────────────────
    tasks: List[Tuple[str, str]] = []
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    for ds in datasets:
        csv_path = data_dir / ds / "query_costs.csv"
        tasks.extend((ds, q) for q in load_mismatched_from_csv(csv_path))

    if not tasks:
        print("No mismatched queries found, abort.", file=sys.stderr)
        sys.exit(1)

    random.shuffle(tasks)
    if 0 < args.limit < len(tasks):
        tasks = tasks[:args.limit]

    print(f"Loaded {len(tasks)} mis-routed queries "
          f"from {len(set(db for db, _ in tasks))} dataset(s)")

    # ── 2. 生成到达时间 ─────────────────────────────────────────
    if args.burst:
        arrivals = burst_arrivals(
            n=len(tasks), seed=args.seed,
            burst_qps=args.burst_qps,
            pause_ms=args.pause_ms,
            burst_len=args.burst_len)
        print(f"Arrival pattern: BURST  (burst_qps={args.burst_qps}, "
              f"pause_ms={args.pause_ms}, burst_len={args.burst_len})")
    elif args.saw:
        arrivals = sawtooth_arrivals(
            n=len(tasks), seed=args.seed,
            base_qps=args.base_qps,
            peak_qps=args.peak_qps,
            period_ms=args.period_ms)
        print(f"Arrival pattern: SAWTOOTH "
              f"(base_qps={args.base_qps}, peak_qps={args.peak_qps}, "
              f"period_ms={args.period_ms})")
    else:
        qps = args.poisson_qps if args.poisson_qps else 20.0
        arrivals = poisson_arrivals(
            n=len(tasks), mean_sec=1.0 / qps, seed=args.seed)
        print(f"Arrival pattern: POISSON  (qps={qps})")

    # ── 3. benchmark 各 routing-mode ───────────────────────────
    mode_res = {}
    for tag in ROUTING_MODES:
        if tag in ("row_only", "col_only"):  # 需要时可打开
            continue
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, tasks, arrivals, args)

        ok   = [v for v in lats if v not in (None, args.timeout/1000.0)]
        to   = [v for v in lats if v == args.timeout/1000.0]
        fail = len(lats) - len(ok) - len(to)
        avg  = mean(ok + to) if (ok or to) else float("nan")
        qps  = len(ok) / mk if mk else 0.0

        print(f"makespan {mk:.2f}s  avg {avg:.4f}s  "
              f"qps {qps:.2f}/s  "
              f"(ok {len(ok)} | timeout {len(to)} | fail {fail})")
        mode_res[tag] = (mk, lats)

    # ── 4. 输出 CSV ────────────────────────────────────────────
    write_csv(Path(args.out), mode_res)
    print(f"\nPer-query latencies saved to {args.out}")


if __name__ == "__main__":
    main()
