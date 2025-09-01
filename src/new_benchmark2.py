#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py — stress-arrival benchmark (time-based).

1. 读取 query_costs.csv 中 cost_use_imci ≠ use_imci 的 SQL。
2. 压测按固定 QPS 运行指定分钟数，分三段：
      • 前 1/3：TP-heavy   （行存快）
      • 中 1/3：AP-heavy   （列存快）
      • 后 1/3：Balanced   （随机）
3. 可以调整行/列线程池并启动 blocker 查询。
4. 并行跑多种 ROUTING_MODES，输出 makespan / 平均延迟与 CSV 明细。
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from threading import Thread, Lock
from typing import List, Tuple, Dict

import pymysql, pymysql.err
from tqdm import tqdm

# ─────────────────────── 三个阶段：名称 & 占比 ───────────────────────
PHASES = [("TP", 1 / 3), ("AP", 1 / 3), ("BAL", 1 - 2 / 3)]
PHASE_SEED_STEP = 937  # 每段随机种子加的偏移，避免到达模式雷同

# ───────────────────────── 服务器参数 ─────────────────────────
HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""
MAX_MS = 120_000  # 过滤极端慢样本，上限 120 s

ALL_DATASETS = [
    "tpch_sf1", "tpch_sf10", "tpch_sf100",
    "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
    "hybench_sf1", "hybench_sf10",
    "airline", "carcinogenesis", "credit", "employee",
    "financial", "geneea", "hepatitis",
]

ROUTING_MODES: Dict[str, List[str]] = {
    "cost_thresh": [
        "set global max_user_connections = 10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = OFF",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "hybrid_opt": [
        "set global max_user_connections = 10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "lgbm_kernel": [
        "set global max_user_connections = 10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = OFF",
    ],
    "lgbm_kernel_mm1": [
        "set global max_user_connections = 10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = ON",
    ],
}

# ──────────────────────────── 工具函数 ────────────────────────────
def load_mismatched_from_csv(path: Path,
                             tp_only=False, ap_only=False) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                if int(r["use_imci"]) == int(r["cost_use_imci"]):
                    continue
                row_ms, col_ms = float(r["row_time"]), float(r["column_time"])
                if row_ms > MAX_MS or col_ms > MAX_MS:
                    continue
                if tp_only and not (row_ms < col_ms):
                    continue
                if ap_only and not (col_ms < row_ms):
                    continue
                out.append(r["query"].rstrip(";").replace('"', ""))
            except Exception:
                pass
    return out

# 到达序列（生成直到 ≥ duration_s）
def poisson_arrivals(rate: float, duration_s: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    while t < duration_s:
        t += rng.expovariate(rate)
        out.append(t)
    return out

def burst_arrivals(burst_qps: float, pause_ms: float,
                   burst_len: int, duration_s: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, arr = 0.0, []
    dt = 1.0 / burst_qps
    while t < duration_s:
        for _ in range(burst_len):
            if t >= duration_s:
                break
            arr.append(t); t += dt
        t += pause_ms / 1000.0
        rng.random()
    return arr

def sawtooth_arrivals(base_qps: float, peak_qps: float, period_ms: float,
                      duration_s: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, arr, steps = 0.0, [], 20
    while t < duration_s:
        for frac in (i / steps for i in range(steps + 1)):
            qps = base_qps + (peak_qps - base_qps) * frac
            dt = 1.0 / qps
            seg_len = period_ms / 1000.0 / steps
            cnt = max(1, int(seg_len / dt))
            for _ in range(cnt):
                if t >= duration_s:
                    break
                arr.append(t); t += dt
            if t >= duration_s:
                break
    # 抖动
    arr = [a + rng.uniform(-0.5, 0.5) * 1e-3 for a in arr]
    arr.sort()
    return arr

# ─────────────────── 可选：服务器参数 / blocker ───────────────────
def apply_server_settings(row_conc: int, col_pool: int):
    try:
        conn = pymysql.connect(host=HOST, port=PORT,
                               user=USER, password=PASS, autocommit=True)
        cur = conn.cursor()
        if row_conc >= 0:
            cur.execute(f"SET GLOBAL innodb_thread_concurrency = {row_conc}")
        if col_pool >= 0:
            cur.execute(f"SET GLOBAL imci_thread_pool_size = {col_pool}")
        cur.close(); conn.close()
    except Exception as e:
        print(f"[WARN] set server variables failed: {e}", file=sys.stderr)

def blocker_worker(db: str, sql_text: str, repeat: int):
    for _ in range(repeat):
        try:
            conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                                   password=PASS, db=db, autocommit=True)
            cur = conn.cursor()
            cur.execute("SET use_imci_engine = OFF")
            cur.execute(sql_text); cur.fetchall()
            cur.close(); conn.close()
        except Exception as e:
            print(f"[BLOCKER] {e}", file=sys.stderr)
            time.sleep(1)

# ─────────────── 每个查询执行线程（与旧版相同） ───────────────
def _exec(idx: int, task: Tuple[str, str], rel: float,
          sess_sql: List[str], args, lat: List,
          bench_start: float, bar, lock: Lock, tag: str):
    db, sql = task
    TIMEOUT_S = args.timeout / 1000.0
    wait = bench_start + rel - time.perf_counter()
    if wait > 0:
        time.sleep(wait)
    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=db, autocommit=True)
        cur = conn.cursor()
        for s in sess_sql: cur.execute(s)
        cur.execute(f"SET max_execution_time = {args.timeout}")
    except Exception as e:
        _log_fail(tag, idx, e, sql); _set_lat(lat, idx, None, bar, lock); return
    t0 = time.perf_counter()
    try:
        cur.execute(sql); cur.fetchall()
        _set_lat(lat, idx, time.perf_counter() - t0, bar, lock)
    except pymysql.err.OperationalError as e:
        _set_lat(lat, idx,
                 TIMEOUT_S if e.args and e.args[0] in (3024, 1317) else None,
                 bar, lock)
    except Exception as e:
        _log_fail(tag, idx, e, sql); _set_lat(lat, idx, None, bar, lock)
    finally:
        cur.close(); conn.close()

def _log_fail(tag, idx, err, sql):
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    print(f"[{tag} #{idx}] {err}\n   SQL: {snippet}", file=sys.stderr)

def _set_lat(lat, i, v, bar, lock):
    with lock:
        lat[i] = v
        bar.update()

# ───────────────────────── 跑一个 MODE ─────────────────────────
def run_mode(tag: str, tasks: List[Tuple[str, str]],
             arrivals: List[float], args):
    n = len(tasks)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=70, leave=False)
    bench_start = time.perf_counter()

    lock = Lock()
    threads: List[Thread] = []
    for i, (job, rel) in enumerate(zip(tasks, arrivals)):
        th = Thread(target=_exec,
                    args=(i, job, rel, ROUTING_MODES[tag],
                          args, lat, bench_start, bar, lock, tag),
                    daemon=True)
        th.start(); threads.append(th)
    for th in threads: th.join()
    bar.close()
    return time.perf_counter() - bench_start, lat

# ───────────────────────────── CSV 输出 ─────────────────────────────
def write_csv(outfile: Path, mode_res: Dict[str, Tuple[float, List]]):
    hdr = ["query_idx"] + [f"{m}_lat" for m in mode_res]
    n = len(next(iter(mode_res.values()))[1])
    rows = [[i] + [mode_res[m][1][i] for m in mode_res] for i in range(n)]
    with outfile.open("w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)

# ─────────────────────────────── main ───────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset")
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--timeout", type=int, default=60_000)
    pa.add_argument("--minutes", type=float, default=1.0,
                    help="total benchmark duration in minutes")
    pa.add_argument("--duration_s", type=float,
                    help="benchmark duration in seconds (overrides --minutes)")
    pa.add_argument("--seed", type=int, default=42)

    # QPS & arrival shape
    grp = pa.add_mutually_exclusive_group()
    grp.add_argument("--qps", type=float, default=20.0)
    grp.add_argument("--burst", action="store_true")
    grp.add_argument("--saw", action="store_true")

    pa.add_argument("--burst_qps", type=float, default=800)
    pa.add_argument("--pause_ms", type=float, default=200)
    pa.add_argument("--burst_len", type=int, default=200)
    pa.add_argument("--base_qps", type=float, default=50)
    pa.add_argument("--peak_qps", type=float, default=500)
    pa.add_argument("--period_ms", type=float, default=500)

    # server limits & blocker
    pa.add_argument("--row_conc", type=int, default=-1)
    pa.add_argument("--col_pool", type=int, default=-1)
    pa.add_argument("--blocker_db")
    pa.add_argument("--blocker_repeat", type=int, default=5)

    pa.add_argument("--out", default="routing_bench.csv")
    args = pa.parse_args()

    duration_s = args.duration_s if args.duration_s else args.minutes * 60
    random.seed(args.seed)

    # 0) server settings & blocker
    apply_server_settings(args.row_conc, args.col_pool)
    # if args.blocker_db:
    #     Thread(target=blocker_worker,
    #            args=(args.blocker_db,
    #                  "SELECT SUM(l_extendedprice) FROM lineitem "
    #                  "WHERE l_shipdate < '1996-01-01';",
    #                  args.blocker_repeat),
    #            daemon=True).start()
    #     time.sleep(1)

    # 1) 读 CSV，准备三类 SQL（不足时循环用）
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    tp_sql, ap_sql, bal_sql = [], [], []
    for ds in datasets:
        base = Path(args.data_dir) / ds / "query_costs.csv"
        tp_sql += load_mismatched_from_csv(base, tp_only=True)
        ap_sql += load_mismatched_from_csv(base, ap_only=True)
        bal_sql += load_mismatched_from_csv(base)

    if not (tp_sql and ap_sql and bal_sql):
        print("SQL 样本不足：TP/AP/BAL 至少各 1 条", file=sys.stderr)
        sys.exit(1)

    # 2) 生成到达时间（各段拼接）
    arrivals, offset, seg_idx = [], 0.0, 0
    for phase, ratio in PHASES:
        seg_dur = duration_s * ratio
        seed = args.seed + PHASE_SEED_STEP * seg_idx
        seg_idx += 1

        if args.burst:
            seg = burst_arrivals(args.burst_qps, args.pause_ms,
                                 args.burst_len, seg_dur, seed)
        elif args.saw:
            seg = sawtooth_arrivals(args.base_qps, args.peak_qps,
                                    args.period_ms, seg_dur, seed)
        else:
            seg = poisson_arrivals(args.qps, seg_dur, seed)

        arrivals.extend([t + offset for t in seg])
        offset += seg_dur

    # 3) 为每个 arrival 选 SQL
    tasks: List[Tuple[str, str]] = []
    idx_tp = idx_ap = idx_bal = 0
    for t in arrivals:
        frac = t / duration_s
        if frac < PHASES[0][1]:
            sql = tp_sql[idx_tp % len(tp_sql)]; idx_tp += 1
        elif frac < PHASES[0][1] + PHASES[1][1]:
            sql = ap_sql[idx_ap % len(ap_sql)]; idx_ap += 1
        else:
            sql = bal_sql[idx_bal % len(bal_sql)]; idx_bal += 1
        # 用文件路径作为 db 名（tpch_sf1 等）
        db = Path(sql).parts[0] if "/" in sql else (args.dataset or "tpch_sf1")
        tasks.append((db, sql))

    print(f"Total arrivals: {len(arrivals)} "
          f"(TP {idx_tp} | AP {idx_ap} | BAL {idx_bal}) "
          f"over {duration_s:.1f}s")

    # 4) benchmark
    mode_res = {}
    for tag in ROUTING_MODES:
        if not tag.startswith("lgbm"):    # 如需全部模式请移除此行
            continue
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, tasks, arrivals, args)
        ok = [v for v in lats if v not in (None, args.timeout / 1000.0)]
        to = [v for v in lats if v == args.timeout / 1000.0]
        fail = len(lats) - len(ok) - len(to)
        avg = mean(ok + to) if (ok or to) else float("nan")
        print(f"(sec) makespan {mk:.2f}  avg {avg:.4f}  "
              f"ok {len(ok)}  to {len(to)}  fail {fail}")
        mode_res[tag] = (mk, lats)

    # 5) CSV
    write_csv(Path(args.out), mode_res)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
