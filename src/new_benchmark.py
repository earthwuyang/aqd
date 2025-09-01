#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py  –  stress-arrival benchmark for routing modes.

功能：
1. 读取 query_costs.csv 中 cost_use_imci ≠ use_imci 的 SQL 作为测试集。
2. 支持 Poisson / Burst / Sawtooth 三种到达曲线，并串行模拟
      • 第一段：TP-heavy   (倾向行存更快)
      • 第二段：AP-heavy   (倾向列存更快)
      • 第三段：Balanced   (随机)
3. 允许运行前强制设置 row/col 线程池，并可启动“阻塞行存查询”霸占 row worker。
4. 并行执行多种 ROUTING_MODES（cost_thresh / hybrid_opt / lgbm_kernel / lgbm_kernel_mm1）。
5. 输出 CSV：query_idx, <mode>_lat…
"""

from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
from statistics import mean
from threading import Thread, Lock
from typing import List, Tuple, Dict

import pymysql
import pymysql.err
from tqdm import tqdm

# ───────────────────────────── 三段压测配置 ──────────────────────────────
PHASES = [("TP", 0.33),    # 33 %  行存更优
          ("AP", 0.33),    # 33 %  列存更优
          ("BAL", 0.34)]   # 其余   随机
PHASE_SEED_STEP = 1000      # 每段给随机种子加上的偏移量，避免到达时间雷同

# ───────────────────────────── 服务器配置 ──────────────────────────────
HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""

ALL_DATASETS = [
    "tpch_sf1", "tpch_sf10", "tpch_sf100",
    "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
    "hybench_sf1", "hybench_sf10",
    "airline", "carcinogenesis", "credit", "employee",
    "financial", "geneea", "hepatitis"
]

ROUTING_MODES: Dict[str, List[str]] = {
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
        "SET use_mm1_time = OFF",
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

MAX_MILLISECONDS = 120_000

# ─────────────────────────── utilities ─────────────────────────────
def load_mismatched_from_csv(path: Path,
                             TP_heavy: bool = False,
                             AP_heavy: bool = False) -> List[str]:
    """
    返回 cost 与实际路由不一致的 SQL；可加过滤：
        • TP_heavy=True  → 仅保留行存更快的
        • AP_heavy=True  → 仅保留列存更快的
    """
    if not path.exists():
        return []
    res: List[str] = []
    with path.open(newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                if int(r["use_imci"]) == int(r["cost_use_imci"]):
                    continue
                row_ms, col_ms = float(r["row_time"]), float(r["column_time"])
                if row_ms > MAX_MILLISECONDS or col_ms > MAX_MILLISECONDS:
                    continue

                if TP_heavy and not (row_ms < col_ms):
                    continue
                if AP_heavy and not (col_ms < row_ms):
                    continue

                res.append(r["query"].rstrip(";").replace('"', ""))
            except Exception:
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
    # 轻微抖动
    arr = [a + rng.uniform(-0.5, 0.5) * 1e-3 for a in arr]
    arr.sort()
    return arr

def log_fail(tag: str, idx: int, err: Exception, sql: str):
    snippet = sql.replace("\n", " ")[:120] + ("…" if len(sql) > 120 else "")
    print(f"[{tag} #{idx}] {err}\n   SQL: {snippet}", file=sys.stderr)

# ──────────── optional: apply server settings / blocker ────────────
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
                  sess_sql: List[str], args, lat: List,
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
def run_mode(tag: str, tasks: List[Tuple[str, str]],
             arrivals: List[float], args):
    n = len(tasks)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=70, leave=False)
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
def write_csv(path: Path, mode_res: Dict[str, Tuple[float, List]]):
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
    pa.add_argument("--limit", "-n", type=int, default=100)
    pa.add_argument("--seed", type=int, default=42)

    # arrival pattern
    grp = pa.add_mutually_exclusive_group()
    grp.add_argument("--poisson_qps", type=float)
    grp.add_argument("--burst", action="store_true")
    grp.add_argument("--saw", action="store_true")

    # burst / saw parameters
    pa.add_argument("--burst_qps", type=float, default=800)
    pa.add_argument("--pause_ms", type=float, default=200)
    pa.add_argument("--burst_len", type=int, default=200)
    pa.add_argument("--base_qps", type=float, default=50)
    pa.add_argument("--peak_qps", type=float, default=500)
    pa.add_argument("--period_ms", type=float, default=500)

    # server limits & blocker
    pa.add_argument("--row_conc", type=int, default=-1,
                    help="SET GLOBAL innodb_thread_concurrency")
    pa.add_argument("--col_pool", type=int, default=-1,
                    help="SET GLOBAL imci_thread_pool_size")
    pa.add_argument("--blocker_db", default='tpch_sf100')
    pa.add_argument("--blocker_repeat", type=int, default=5)

    pa.add_argument("--out", default="routing_bench.csv")
    args = pa.parse_args()

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

    # ── 1. 分阶段收集 SQL ─────────────────────────────────────
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    tasks: List[Tuple[str, str]] = []
    remain = args.limit

    for phase_name, ratio in PHASES:
        want = max(1, int(args.limit * ratio))
        if phase_name != PHASES[-1][0] and want > remain:
            want = remain
        remain -= want

        tp_flag = (phase_name == "TP")
        ap_flag = (phase_name == "AP")

        phase_tasks: List[Tuple[str, str]] = []
        for ds in datasets:
            qs = load_mismatched_from_csv(
                data_dir / ds / "query_costs.csv",
                TP_heavy=tp_flag, AP_heavy=ap_flag)
            random.shuffle(qs)
            take = qs[:max(0, want - len(phase_tasks))]
            phase_tasks.extend((ds, q) for q in take)
            if len(phase_tasks) >= want:
                break

        if len(phase_tasks) < want:
            print(f"[WARN] phase {phase_name} 想要 {want} 条，只拿到 {len(phase_tasks)} 条")
        tasks.extend(phase_tasks)

    if not tasks:
        print("no mismatched queries found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(tasks)} queries "
          f"(TP={int(args.limit*PHASES[0][1])}, "
          f"AP={int(args.limit*PHASES[1][1])}, "
          f"BAL={len(tasks)-int(args.limit*(PHASES[0][1]+PHASES[1][1]))})")

    # ── 2. 生成到达时间（分段平移） ─────────────────────────────
    arrivals: List[float] = []
    offset = 0.0
    seg_idx = 0
    for phase_name, ratio in PHASES:
        cnt = int(args.limit * ratio)
        if phase_name == PHASES[-1][0]:           # 最后一段吃掉误差
            cnt = len(tasks) - len(arrivals)

        seed_shift = args.seed + seg_idx * PHASE_SEED_STEP
        seg_idx += 1

        if args.burst:
            seg = burst_arrivals(cnt, args.burst_qps,
                                 args.pause_ms, args.burst_len, seed_shift)
        elif args.saw:
            seg = sawtooth_arrivals(cnt, args.base_qps, args.peak_qps,
                                    args.period_ms, seed_shift)
        else:
            qps = args.poisson_qps or 20.0
            seg = poisson_arrivals(cnt, 1.0 / qps, seed_shift)

        arrivals.extend([t + offset for t in seg])
        # 下一段整体右移一点点（一个平均到达间隔）
        offset = arrivals[-1] + (1.0 / (args.poisson_qps or 20.0))

    # ── 3. benchmark ───────────────────────────────────────────
    mode_res = {}
    for tag in ROUTING_MODES:
        # if not tag.startswith("lgbm"):    # 如需全部模式请删掉此行
        #     continue
        print(f"\n=== {tag} ===")
        mk, lats = run_mode(tag, tasks, arrivals, args)
        ok = [v for v in lats if v not in (None, args.timeout / 1000.0)]
        to = [v for v in lats if v == args.timeout / 1000.0]
        fail = len(lats) - len(ok) - len(to)
        avg = mean(ok + to) if (ok or to) else float("nan")
        print(f"(seconds) makespan {mk:.2f}  avg {avg:.4f}  "
              f"ok {len(ok)}  to {len(to)}  fail {fail}")
        mode_res[tag] = (mk, lats)

    # ── 4. CSV ────────────────────────────────────────────────
    write_csv(Path(args.out), mode_res)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
