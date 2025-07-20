#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py  –  stress-arrival benchmark for routing modes.

新增功能
========
* --rounds (默认 3)：重复基准测试多次。
* 统计四种模式（cost_thresh / hybrid_opt / lgbm_kernel / lgbm_kernel_mm1）
  的 3 个指标：
    • makespan (秒)
    • avg latency (包含 timeout)
    • p95 latency
  并给出 “平均 ± 标准差”。

其它逻辑与原脚本保持一致。
"""

from __future__ import annotations
import argparse, csv, random, sys, time, os
from pathlib import Path
from statistics import mean, stdev
from threading import Thread, Lock
from typing import List, Tuple

import pymysql, pymysql.err
from tqdm import tqdm
import psutil, matplotlib.pyplot as plt, collections, itertools
from threading import Event

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
def p95(vals: List[float]) -> float:
    """简单 p95 百分位实现；空列表返回 nan"""
    if not vals:
        return float("nan")
    vals_sorted = sorted(vals)
    k = int(0.95 * (len(vals_sorted) - 1))
    return vals_sorted[k]

### MON ###
# ——— thread → ROW/COL/SHA 分类 ———
COL_PREFIX = ("imci[", "IMCI_OPT", "imci0flush", "imci_writer",
              "bg_nci", "load_nci", "bg_nci_memtbl", "bg_rb_ctrl",
              "bg_nci_bp", "imci_stats")
SHR_PREFIX = ("checkpoint", "log_", "write_notifier",
              "mlog_timer", "csn_log_timer", "buf_flush",
              "purge_")

def _classify(comm: str) -> str:
    for p in COL_PREFIX:
        if comm.startswith(p): return "COL"
    for p in SHR_PREFIX:
        if comm.startswith(p): return "SHA"
    return "ROW"

_HZ   = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
_NCPU = os.cpu_count()

def _read_jiffies(stat_path: str) -> int:
    with open(stat_path) as f:
        txt = f.read()
    rpar = txt.rfind(')')
    fields = txt[rpar + 2:].split()
    return int(fields[11]) + int(fields[12])   # utime+stime

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
                if int(r["use_imci"]) != int(r["cost_use_imci"]):
                    if not TP_heavy and not AP_heavy:
                        res.append(r["query"].rstrip(";").replace('"', ""))
                    elif TP_heavy and row_time < column_time:
                        res.append(r["query"].rstrip(";").replace('"', ""))
                    elif AP_heavy and column_time < row_time:
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
    if wait > 0: time.sleep(wait)

    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=db, autocommit=True)
        cur = conn.cursor()
        for s in sess_sql: cur.execute(s)
        cur.execute(f"SET max_execution_time = {args.timeout}")
    except Exception as e:
        log_fail(tag, idx, e, sql)
        with lock: lat[idx] = None; bar.update(); return

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


### MON ###
def _plot_util(samples: list[tuple], fname: str, title: str):
    t0 = samples[0][0]
    xs  = [s[0]-t0 for s in samples]
    row = [s[1] for s in samples]
    col = [s[2] for s in samples]
    rss = [s[3] for s in samples]

    fig, ax1 = plt.subplots()
    ax1.plot(xs, row, label="ROW cpu%")
    ax1.plot(xs, col, label="COL cpu%")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("cpu %")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(xs, rss, "--", label="RSS MB")
    ax2.set_ylabel("RSS (MB)")
    ax2.legend(loc="upper right")

    plt.title(f"Engine utilisation – {title}")
    plt.tight_layout(); fig.savefig(fname, dpi=120)
    plt.close(fig)

class UtilMonitor(Thread):
    """后台采样 mysqld ROW/COL cpu% + 进程 RSS(MB)。"""
    def __init__(self, pid: int, out_list: list[tuple]):
        super().__init__(daemon=True)
        self.pid = pid
        self.out = out_list              # (t,row%,col%,rss)
        self._stop_evt = Event()         # ← 事件对象

    def stop(self):
        self._stop_evt.set()             # 触发停止

    def run(self):
        last, t_last = self._snapshot()
        while not self._stop_evt.is_set():   # ← 检查事件
            time.sleep(1)
            acc, t_now = self._snapshot()
            dt = t_now - t_last
            d_row = acc["ROW"] - last["ROW"]
            d_col = acc["COL"] - last["COL"]
            row_pct = 100 * d_row / (dt * _HZ * _NCPU)
            col_pct = 100 * d_col / (dt * _HZ * _NCPU)
            rss_mb  = psutil.Process(self.pid).memory_info().rss / (1 << 20)
            self.out.append((t_now, row_pct, col_pct, rss_mb))
            last, t_last = acc, t_now

    def _snapshot(self):
        base = f"/proc/{self.pid}/task"
        acc = collections.Counter(ROW=0, COL=0, SHA=0)
        for tid in os.listdir(base):
            comm = open(f"{base}/{tid}/comm").read().strip()
            cls  = _classify(comm)
            acc[cls] += _read_jiffies(f"{base}/{tid}/stat")
        return acc, time.time()



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

# ──────────────────────────── main ────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset")
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--timeout", type=int, default=60000)
    pa.add_argument("--limit", "-n", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--rounds", type=int, default=3)             ### NEW ###

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
    pa.add_argument("--AP_heavy", action="store_true", default=False)
    pa.add_argument("--TP_heavy", action="store_true", default=False)

    pa.add_argument("--mysqld_pid", type=int, required=True, help="mysqld process pid, for /proc sampling")
    args = pa.parse_args()

    assert (args.AP_heavy and not args.TP_heavy) or (args.TP_heavy and not args.AP_heavy)\
           or (not args.TP_heavy and not args.AP_heavy)

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
        time.sleep(1)  # give blocker a moment

    # ── 1. 收集 SQL ───────────────────────────────────────────
    tasks: List[Tuple[str, str]] = []
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    for ds in datasets:
        qs = load_mismatched_from_csv(data_dir / ds / "query_costs.csv",
                                      args.TP_heavy, args.AP_heavy)
        tasks.extend((ds, q) for q in qs)
    if not tasks:
        print("no mismatched queries found", file=sys.stderr); sys.exit(1)

    random.shuffle(tasks)
    tasks = tasks[: args.limit] if 0 < args.limit < len(tasks) else tasks
    print(f"Loaded {len(tasks)} queries from {len(set(db for db, _ in tasks))} dataset(s)")

    # ── 2. 生成到达时间 (一次即可，多轮复用) ─────────────────────
    if args.burst:
        arrivals = burst_arrivals(len(tasks), args.burst_qps,
                                  args.pause_ms, args.burst_len, args.seed)
        print(f"Arrival=Burst  qps={args.burst_qps} len={args.burst_len} pause={args.pause_ms}ms")
    elif args.saw:
        arrivals = sawtooth_arrivals(len(tasks), args.base_qps, args.peak_qps,
                                     args.period_ms, args.seed)
        print(f"Arrival=Saw   {args.base_qps}->{args.peak_qps} qps period={args.period_ms}ms")
    else:
        qps = args.poisson_qps or 20.0
        arrivals = poisson_arrivals(len(tasks), 1.0 / qps, args.seed)
        print(f"Arrival=Poisson  qps={qps}")

    # ── 3. 多轮 benchmark ─────────────────────────────────────
    agg = {m: {"mk": [], "avg": [], "p95": []} for m in ROUTING_MODES}

    for rd in range(1, args.rounds + 1):
        print(f"\n================  Round {rd}/{args.rounds}  ================")
        for tag in ROUTING_MODES:
            print(f"\n=== {tag} ===")
            samples: list[tuple] = []
            mon = UtilMonitor(args.mysqld_pid, samples)
            mon.start()

            mk, lats = run_mode(tag, tasks, arrivals, args)

            mon.stop(); mon.join()
            if samples:
                _plot_util(samples, fname=f"util_{tag}.png", title=tag)

            ok = [v for v in lats if v not in (None, args.timeout / 1000.0)]
            to = [v for v in lats if v == args.timeout / 1000.0]
            avg_lat = mean(ok + to) if (ok or to) else float("nan")
            p95_lat = p95(ok + to)

            print(f"(seconds) makespan {mk:.2f}  avg_lat {avg_lat:.4f}  p95_lat {p95_lat:.4f} "
                  f"ok {len(ok)}  to {len(to)}  fail {len(lats) - len(ok) - len(to)}")

            agg[tag]["mk"].append(mk)
            agg[tag]["avg"].append(avg_lat)
            agg[tag]["p95"].append(p95_lat)

        print("\n------------------------------------------------------------")

    # ── 4. 汇总输出 ────────────────────────────────────────────
    print("\n################### 最终汇总 ###################")
    def fmt(lst, prec=2):  # 带 ± 标准差
        if not lst: return "nan"
        m = mean(lst)
        s = stdev(lst) if len(lst) > 1 else 0.0
        return f"{m:.{prec}f} ±{s:.{prec}f}"

    header = ("MODE", "makespan(s)", "avg_latency(s)", "p95_latency(s)")
    print(f"{header[0]:<16}{header[1]:>15}{header[2]:>18}{header[3]:>18}")
    for tag in ("cost_thresh", "hybrid_opt", "lgbm_kernel", "lgbm_kernel_mm1"):
        mk_s  = fmt(agg[tag]["mk"], 2)
        avg_s = fmt(agg[tag]["avg"], 4)
        p95_s = fmt(agg[tag]["p95"], 4)
        print(f"{tag:<16}{mk_s:>15}{avg_s:>18}{p95_s:>18}")

    # # ── 5. CSV (最后一轮同原逻辑) ─────────────────────────────
    # write_csv(Path(args.out),
    #           {m: (agg[m]["mk"][-1], [*range(len(tasks))])  # dummy lat just to match signature
    #            for m in ROUTING_MODES})
    print(f"\nSaved last-round latencies to {args.out}")

if __name__ == "__main__":
    main()
