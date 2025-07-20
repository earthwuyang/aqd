#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py  –  stress-arrival benchmark for routing modes.

功能：
1. 读取 query_costs.csv 中 cost_use_imci ≠ use_imci 的 SQL 作为测试集。
2. 支持三种到达：Poisson / Burst / Sawtooth。
3. 允许运行前强制设置 row/col 线程池，并可启动“阻塞行存查询”霸占 row worker。
4. 并行执行多种 ROUTING_MODES（cost_thresh / hybrid_opt / lgbm_kernel / lgbm_kernel_mm1）。
5. 输出 CSV：query_idx, <mode>_lat…
"""

from __future__ import annotations
import argparse, csv, random, sys, time, pathlib
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
    "tpch_sf1", "tpch_sf10", "tpch_sf100",
    "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
    "hybench_sf1", "hybench_sf10",
    "airline","carcinogenesis","credit","employee","financial","geneea","hepatitis"
]

ROUTING_MODES = {
    "cost_thresh": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = OFF",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "hybrid_opt": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "lgbm_kernel": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = OFF"
    ],
    "lgbm_kernel_mm1": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = ON",
    ],
}

MAX_MILISECONDS = 120000
# ─────────────────────────── utilities ─────────────────────────────
def load_mismatched_from_csv(path: Path, TP_heavy=False, AP_heavy=False) -> List[str]:
    if not path.exists():
        return []
    res: List[str] = []
    with path.open(newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                row_time = r["row_time"]
                column_time = r["column_time"]
                # if int(r["use_imci"]) != int(r["cost_use_imci"] and float(r["row_time"])<MAX_MILISECONDS and float(r["column_time"])<MAX_MILISECONDS):
                if int(r["use_imci"]) != int(r["cost_use_imci"]):
                    if not TP_heavy and not AP_heavy:
                        res.append(r["query"].rstrip(";").replace('"', ""))
                    elif TP_heavy:
                        if row_time < column_time:
                            res.append(r["query"].rstrip(";").replace('"', ""))
                    elif AP_heavy:
                        if column_time < row_time:
                            res.append(r["query"].rstrip(";").replace('"', ""))
            except Exception as e:
                # print(f"exception: {e}")
                pass
    return res


# ---------- arrival patterns ----------
def poisson_arrivals(n: int, mean_sec: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        out.append(t)
    return out


def burst_arrivals(n: int, burst_qps: float, pause_ms: float,
                   burst_len: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, arr, dt = 0.0, [], 1.0 / burst_qps
    while len(arr) < n:
        for _ in range(min(burst_len, n - len(arr))):
            arr.append(t); t += dt
        t += pause_ms / 1000.0
        rng.random()
    return arr


def sawtooth_arrivals(n: int, base_qps: float, peak_qps: float,
                      period_ms: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, arr, steps = 0.0, [], 20
    while len(arr) < n:
        for frac in (i / steps for i in range(steps + 1)):
            qps = base_qps + (peak_qps - base_qps) * frac
            dt = 1.0 / qps
            seg_len = period_ms / 1000.0 / steps
            cnt = max(1, int(seg_len / dt))
            for _ in range(min(cnt, n - len(arr))):
                arr.append(t); t += dt
            if len(arr) >= n:
                break
    arr = [a + rng.uniform(-0.5, 0.5) * 1e-3 for a in arr]
    arr.sort()
    return arr


def log_fail(tag: str, idx: int, err: Exception, sql: str):
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    print(f"[{tag} #{idx}] {err}\n   SQL: {snippet}", file=sys.stderr)


# ──────────── optional: apply server settings / blocker ────────────
def apply_server_settings(row_conc: int, col_pool: int):
    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, autocommit=True)
        cur = conn.cursor()
        if row_conc >= 0:
            cur.execute(f"SET GLOBAL innodb_thread_concurrency = {row_conc}")
        if col_pool >= 0:
            cur.execute(f"SET GLOBAL imci_thread_pool_size = {col_pool}")
        cur.close(); conn.close()
    except Exception as e:
        print(f"[WARN] failed to set server variables: {e}", file=sys.stderr)


def blocker_worker(db: str, sql_text: str, repeat: int):
    """连续 repeat 次执行 sql_text，强制行存，霸占 row worker。"""
    for _ in range(repeat):
        try:
            conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                                   password=PASS, db=db, autocommit=True)
            cur = conn.cursor()
            cur.execute("SET use_imci_engine = OFF")
            cur.execute(sql_text)
            cur.fetchall()
            cur.close(); conn.close()
        except Exception as e:
            print(f"[BLOCKER] {e}", file=sys.stderr)
            time.sleep(1)


# ───────────────────── per-query execution thread ───────────────────
def execute_query(idx: int, task: Tuple[str, str], arrival_rel: float,
                  sess_sql: list[str], args, lat: list,
                  bench_start: float, bar, lock: Lock, tag: str):
    db, sql = task
    TIMEOUT_S = args.timeout / 1000.0
    wait = bench_start + arrival_rel - time.perf_counter()
    if wait > 0:
        time.sleep(wait)

    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=db, autocommit=True)
        cur = conn.cursor()
        for s in sess_sql:
            cur.execute(s)
        cur.execute(f"SET max_execution_time = {args.timeout}")
    except Exception as e:
        log_fail(tag, idx, e, sql)
        with lock: lat[idx] = None; bar.update()
        return

    t0 = time.perf_counter()
    try:
        cur.execute(sql); cur.fetchall()
        with lock: lat[idx] = time.perf_counter() - t0
    except pymysql.err.OperationalError as e:
        with lock:
            lat[idx] = TIMEOUT_S if e.args and e.args[0] in (3024, 1317) else None
    except Exception as e:
        log_fail(tag, idx, e, sql)
        with lock: lat[idx] = None
    finally:
        with lock: bar.update()
        cur.close(); conn.close()


# ───────────────────────── one routing mode ─────────────────────────
def run_mode(tag: str, tasks: List[Tuple[str, str]], arrivals: List[float], args):
    n = len(tasks)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=68, leave=False)
    bench_start = time.perf_counter()

    lock = Lock()
    threads: List[Thread] = []
    for i, (job, rel) in enumerate(zip(tasks, arrivals)):
        th = Thread(target=execute_query,
                    args=(i, job, rel, ROUTING_MODES[tag],
                          args, lat, bench_start, bar, lock, tag),
                    daemon=True)
        th.start(); threads.append(th)
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
    pa.add_argument("--dataset")
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--timeout", type=int, default=60000)
    pa.add_argument("--limit", "-n", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=42)

    # arrival pattern
    grp = pa.add_mutually_exclusive_group()
    grp.add_argument("--poisson_qps", type=float, default=100)
    grp.add_argument("--burst", action="store_true")
    grp.add_argument("--saw", action="store_true")

    # burst / saw parameters
    pa.add_argument("--burst_qps", type=float, default=800)
    pa.add_argument("--pause_ms", type=float, default=200)
    pa.add_argument("--burst_len", type=int, default=200)
    pa.add_argument("--base_qps", type=float, default=50)
    pa.add_argument("--peak_qps", type=float, default=500)  # 800
    pa.add_argument("--period_ms", type=float, default=500)

    # server limits & blocker
    pa.add_argument("--row_conc", type=int, default=-1,
                    help="SET GLOBAL innodb_thread_concurrency")
    pa.add_argument("--col_pool", type=int, default=-1,
                    help="SET GLOBAL imci_thread_pool_size")
    pa.add_argument("--blocker_db", default='tpch_sf100')
    pa.add_argument("--blocker_repeat", type=int, default=5)

    pa.add_argument("--out", default="routing_bench.csv")
    pa.add_argument("--AP_heavy", action="store_true", default=False)
    pa.add_argument("--TP_heavy", action="store_true", default=False)
    args = pa.parse_args()

    assert (args.AP_heavy and not args.TP_heavy) or (args.TP_heavy and not args.AP_heavy or (not args.TP_heavy or not args.AP_heavy)), "only one of AP_heavy and TP_heavy can be set"

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # ── 0. server settings & blocker ─────────────────────────────
    apply_server_settings(args.row_conc, args.col_pool)

    if args.blocker_db:
        sql_text = """
        SELECT SUM(l_extendedprice)
        FROM lineitem
        WHERE l_shipdate < '1996-01-01'; 
        """
        print(f"[BLOCKER] running on {args.blocker_db}, repeat={args.blocker_repeat}")
        Thread(target=blocker_worker,
               args=(args.blocker_db, sql_text, args.blocker_repeat),
               daemon=True).start()
        time.sleep(1)  # 给 blocker 一点启动时间

    # ── 1. 收集 SQL ───────────────────────────────────────────
    tasks: List[Tuple[str, str]] = []
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    for ds in datasets:
        qs = load_mismatched_from_csv(data_dir / ds / "query_costs.csv", args.TP_heavy, args.AP_heavy)
        tasks.extend((ds, q) for q in qs)
    if not tasks:
        print("no mismatched queries found", file=sys.stderr); sys.exit(1)

    random.shuffle(tasks)
    tasks = tasks[: args.limit] if 0 < args.limit < len(tasks) else tasks
    print(f"Loaded {len(tasks)} queries from {len(set(db for db, _ in tasks))} dataset(s)")

    # ── 2. 生成到达时间 ─────────────────────────────────────────
    if args.burst:
        arrivals = burst_arrivals(len(tasks), args.burst_qps,
                                  args.pause_ms, args.burst_len, args.seed)
        print(f"Arrival=Burst  qps={args.burst_qps} len={args.burst_len} "
              f"pause={args.pause_ms}ms")
    elif args.saw:
        arrivals = sawtooth_arrivals(len(tasks), args.base_qps, args.peak_qps,
                                     args.period_ms, args.seed)
        print(f"Arrival=Saw   {args.base_qps}->{args.peak_qps} qps period={args.period_ms}ms")
    else:
        qps = args.poisson_qps or 20.0
        arrivals = poisson_arrivals(len(tasks), 1.0 / qps, args.seed)
        print(f"Arrival=Poisson  qps={qps}")

    # ── 3. benchmark ───────────────────────────────────────────
    mode_res = {}
    for tag in ROUTING_MODES:
        # if tag != "lgbm_kernel":
        #     continue
        # if not tag.startswith("lgbm"):
        #     continue
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, tasks, arrivals, args)
        ok = [v for v in lats if v not in (None, args.timeout / 1000.0)]
        to = [v for v in lats if v == args.timeout / 1000.0]
        fail = len(lats) - len(ok) - len(to)
        avg = mean(ok + to) if (ok or to) else float("nan")
        print(f"(seconds) makespan {mk:.2f}  avg {avg:.4f}  ok {len(ok)}  to {len(to)}  fail {fail}")
        mode_res[tag] = (mk, lats)

    # ── 4. CSV ────────────────────────────────────────────────
    write_csv(Path(args.out), mode_res)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
