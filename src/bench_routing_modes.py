#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_routing_modes.py — stress‑arrival benchmark for routing modes.

新增功能
========
1. --rounds (默认 3)：重复基准测试多次。
2. --scenario 选择压测场景:
      • default         —— 原有逻辑（Poisson + mismatched 查询）
      • phase_switch    —— 轻 → 重 → 混合三阶段，锯齿流量
      • mixed_mismatch  —— 错配查询集中出现，后续随机
      • resource_spike  —— 高 QPS burst + 行存阻塞线程
3. 三个指标：makespan / avg_latency / p95_latency，输出 “均值 ± 标准差”。
4. 可选 burst / saw / poisson 自定义到达模式依旧可用。
"""

from __future__ import annotations

import argparse, csv, os, random, sys, time
from pathlib import Path
from statistics import mean, stdev
from threading import Event, Lock, Thread
from typing import List, Tuple

import collections
import psutil
import pymysql, pymysql.err
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt

# ───────────────────────────── config ──────────────────────────────
HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""

ALL_DATASETS = [
    "tpch_sf1", "tpch_sf10", "tpch_sf100",
    "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
    "hybench_sf1", "hybench_sf10",
    "airline", "carcinogenesis", "credit", "employee",
    "financial", "geneea", "hepatitis"
]

ROUTING_MODES = {
    "cost_threshold": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = OFF",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "hybrid_optimizer": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
    ],
    "lightgbm_static": [
        "set global max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled  = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = OFF",
    ],
    "lightgbm_dynamic": [
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


_thread_locals = threading.local()

def get_conn(db: str):
    """在当前线程上复用同一条 PyMySQL 连接（按 db 分库）。"""
    key = f"conn_{db}"
    if not hasattr(_thread_locals, key):
        setattr(
            _thread_locals,
            key,
            pymysql.connect(
                host=HOST, port=PORT, user=USER, password=PASS,
                db=db, autocommit=True, charset="utf8mb4"
            ),
        )
    return getattr(_thread_locals, key)

# ---------- decide ROW / COL ----------
# 你可以替换这里的判断方式：
def choose_engine(tag: str, sql: str) -> str:
    """
    返回 'ROW' 或 'COL' —— 供资源控制使用。
    * lightgbm_xxx 直接看模型标签
    * 其他模式用简化 heuristics（示例：含 GROUP BY + 大表 join 归为 COL）
    """
    if tag.startswith("lightgbm"):
        return "COL" if "cost_use_imci=1" in sql or "/*IMCI*/" in sql else "ROW"
    # fallback：用关键字粗分
    heavy_kw = ("GROUP BY", "JOIN", "WINDOW", "UNION")
    return "COL" if any(k in sql.upper() for k in heavy_kw) else "ROW"

# ---------- map 到 rc 名 ----------
ENGINE_RC = {"ROW": "rc_tp", "COL": "rc_ap"}

# ─────────────────────────── utilities ─────────────────────────────
def p95(vals: List[float]) -> float:
    """简易 p95 百分位；空输入返回 nan"""
    if not vals:
        return float("nan")
    vals_sorted = sorted(vals)
    k = int(0.95 * (len(vals_sorted) - 1))
    return vals_sorted[k]

# —— 线程名 → ROW / COL / SHA 分类 ——
COL_PREFIX = (
    "imci[", "IMCI_OPT", "imci0flush", "imci_writer",
    "bg_nci", "load_nci", "bg_nci_memtbl", "bg_rb_ctrl",
    "bg_nci_bp", "imci_stats"
)
SHR_PREFIX = (
    "checkpoint", "log_", "write_notifier",
    "mlog_timer", "csn_log_timer", "buf_flush",
    "purge_"
)

def _classify(comm: str) -> str:
    for p in COL_PREFIX:
        if comm.startswith(p):
            return "COL"
    for p in SHR_PREFIX:
        if comm.startswith(p):
            return "SHA"
    return "ROW"

_HZ = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
_NCPU = os.cpu_count()

def _read_jiffies(stat_path: str) -> int:
    with open(stat_path) as f:
        txt = f.read()
    rpar = txt.rfind(")")
    fields = txt[rpar + 2 :].split()
    return int(fields[11]) + int(fields[12])  # utime+stime

# ──────────── CSV loader（行/列错配） ────────────
def load_mismatched_from_csv(
    path: Path, TP_heavy: bool = False, AP_heavy: bool = False
) -> List[str]:
    if not path.exists():
        return []
    res: List[str] = []
    with path.open(newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                row_t = float(r["row_time"])
                col_t = float(r["column_time"])
                if int(r["use_imci"]) == int(r["cost_use_imci"]) and row_t < 60 and col_t < 60:
                    continue  # 不是错配
                if TP_heavy and row_t >= col_t:
                    continue
                if AP_heavy and col_t >= row_t:
                    continue
                res.append(r["query"].rstrip(";").replace('"', ""))
            except Exception:
                pass
    return res

# ──────────── arrival generators ────────────
def poisson_arrivals(n: int, mean_sec: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        out.append(t)
    return out

def burst_arrivals(
    n: int, burst_qps: float, pause_ms: float, burst_len: int, seed: int
) -> List[float]:
    rng = random.Random(seed)
    t, arr, dt = 0.0, [], 1.0 / burst_qps
    while len(arr) < n:
        for _ in range(min(burst_len, n - len(arr))):
            arr.append(t)
            t += dt
        t += pause_ms / 1000.0
        rng.random()
    return arr

def sawtooth_arrivals(
    n: int, base_qps: float, peak_qps: float, period_ms: float, seed: int
) -> List[float]:
    rng = random.Random(seed)
    t, arr, steps = 0.0, [], 20
    while len(arr) < n:
        for frac in (i / steps for i in range(steps + 1)):
            qps = base_qps + (peak_qps - base_qps) * frac
            dt = 1.0 / qps
            seg_len = period_ms / 1000.0 / steps
            cnt = max(1, int(seg_len / dt))
            for _ in range(min(cnt, n - len(arr))):
                arr.append(t)
                t += dt
            if len(arr) >= n:
                break
    arr = [a + rng.uniform(-0.5, 0.5) * 1e-3 for a in arr]
    arr.sort()
    return arr

# ──────────── stub 轻 / 重查询加载器 ────────────
def load_lightweight_queries(data_dir: Path) -> List[str]:
    """用户可替换为实际 SQL 集合，这里仅占位示例"""
    path = data_dir / "light_queries.sql"
    if not path.exists():
        return []
    return [l.strip().rstrip(";") for l in path.open() if l.strip()]

def load_heavy_analytic_queries(data_dir: Path) -> List[str]:
    path = data_dir / "heavy_queries.sql"
    if not path.exists():
        return []
    return [l.strip().rstrip(";") for l in path.open() if l.strip()]

# ──────────── 场景构造 ────────────
def build_phase_switch(data_dir: Path) -> List[Tuple[str, str]]:
    light = load_lightweight_queries(data_dir)
    heavy = load_heavy_analytic_queries(data_dir)
    tasks: List[Tuple[str, str]] = []
    tasks += [("db", q) for q in random.choices(light, k=200)]
    tasks += [("db", q) for q in random.choices(heavy, k=100)]
    for _ in range(20):
        tasks += [("db", q) for q in random.choices(heavy, k=5)]
        tasks += [("db", q) for q in random.choices(light, k=10)]
    return tasks

# ──────────── 场景构造 ────────────
def build_mixed_mismatch(
    data_dir: Path,
    args,
) -> List[Tuple[str, str]]:
    """
    返回长度 = 2 * k 的列表，且
        • 前 k 条 row_time < col_time   (ROW-wins)
        • 后 k 条 col_time < row_time   (COL-wins)
    保证两类数量相同，更均衡。
    """
    row_better: List[Tuple[str, str]] = []
    col_better: List[Tuple[str, str]] = []

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    for ds in datasets:
        csv_path = data_dir / ds / "query_costs.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    row_t = float(r["row_time"])
                    col_t = float(r["column_time"])
                    if int(r["use_imci"]) == int(r["cost_use_imci"]):
                        continue        # 不是错配
                    sql = r["query"].rstrip(";").replace('"', "")
                    if row_t < col_t:   # 行存更快
                        row_better.append((ds, sql))
                    elif col_t < row_t: # 列存更快
                        col_better.append((ds, sql))
                except Exception:
                    continue

    # —— 均衡采样 —— #
    random.shuffle(row_better)
    random.shuffle(col_better)
    k = min(len(row_better), len(col_better))
    balanced = row_better[:k] + col_better[:k]
    random.shuffle(balanced)            # 打乱先后顺序

    return balanced


# ──────────── server helper ────────────
def apply_server_settings(row_conc: int, col_pool: int):
    try:
        conn = pymysql.connect(
            host=HOST, port=PORT, user=USER, password=PASS, autocommit=True
        )
        cur = conn.cursor()
        if row_conc >= 0:
            cur.execute(f"SET GLOBAL innodb_thread_concurrency = {row_conc}")
        if col_pool >= 0:
            cur.execute(f"SET GLOBAL imci_thread_pool_size = {col_pool}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[WARN] failed to set server variables: {e}", file=sys.stderr)

# ────────── optional row‑engine blocker ──────────
def blocker_worker(db: str, sql_text: str, repeat: int):
    for _ in range(repeat):
        try:
            conn = pymysql.connect(
                host=HOST, port=PORT, user=USER, password=PASS, db=db, autocommit=True
            )
            cur = conn.cursor()
            cur.execute("SET use_imci_engine = OFF")
            cur.execute(sql_text)
            cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[BLOCKER] {e}", file=sys.stderr)
            time.sleep(1)

# ────────── query execution thread ──────────
def execute_query(
    idx: int,
    task: Tuple[str, str],
    arrival_rel: float,
    sess_sql: list[str],
    args,
    lat: list,
    bench_start: float,
    bar,
    lock: Lock,
    tag: str,
):
    db, sql = task
    TIMEOUT_S = args.timeout / 1000.0

    # —— 到达时间控制 —— #
    wait = bench_start + arrival_rel - time.perf_counter()
    if wait > 0:
        time.sleep(wait)

    try:
        # 1. 线程级持久连接
        conn = get_conn(db)
        cur  = conn.cursor()

        # 2. 首次进入线程：会话参数
        if not getattr(_thread_locals, "sess_inited", False):
            for s in sess_sql:
                cur.execute(s)
            _thread_locals.sess_inited = True

        # 3. 可选：动态资源控制
        if getattr(args, "enable_resource_control", False):
            eng = choose_engine(tag, sql)          # 'ROW' / 'COL'
            rc  = ENGINE_RC[eng]                   # 'rc_tp' / 'rc_ap'
            cid = conn.thread_id()                 # ← 数值线程 ID
            cur.execute(
                f"SET POLAR_RESOURCE_CONTROL {rc} FOR CONNECTION {cid}"
            )

        # 4. 单条查询超时
        cur.execute(f"SET max_execution_time = {args.timeout}")

    except Exception as e:
        snippet = sql.replace("\n", " ")[:120]
        print(f"[{tag} #{idx}] {e}\n   SQL: {snippet}", file=sys.stderr)
        with lock:
            lat[idx] = None
            bar.update()
        return

    # —— 执行查询并计时 —— #
    t0 = time.perf_counter()
    try:
        cur.execute(sql)
        cur.fetchall()
        with lock:
            lat[idx] = time.perf_counter() - t0
    except pymysql.err.OperationalError as e:
        with lock:
            lat[idx] = TIMEOUT_S if e.args and e.args[0] in (3024, 1317) else None
    except Exception as e:
        snippet = sql.replace("\n", " ")[:120]
        print(f"[{tag} #{idx}] {e}\n   SQL: {snippet}", file=sys.stderr)
        with lock:
            lat[idx] = None
    finally:
        with lock:
            bar.update()
        cur.close()          # 连接保留给本线程



# ────────── utilisation monitor (ROW/COL) ──────────
def _read_rss_pages(statm_path: str) -> int:
    """读取 /proc/.../statm 中的 RSS 页数，返回 KB"""
    try:
        with open(statm_path) as f:
            buf = f.read().split()
        # statm: size,resident,shared,... RSS 在第二列
        rss_pages = int(buf[1])
        return rss_pages * (os.sysconf("SC_PAGESIZE") // 1024)
    except:
        return 0

_MB = 1024 * 1024

# ────────── utilisation monitor (ROW/COL) ──────────
class UtilMonitor(Thread):
    """
    (t, row_cpu%, col_cpu%, row_MB, col_MB)
      row_MB = 已用 BP + 脏页(≈undo/tmp)      【会轻微波动】
      col_MB = LRU Cache + execution memory   【会随大查询跳动】
    """
    def __init__(self, pid: int, out_list: list[tuple]):
        super().__init__(daemon=True)
        self.pid = pid
        self.out = out_list
        self._stop_evt = Event()
        self._conn = pymysql.connect(
            host=HOST, port=PORT, user=USER, password=PASS,
            db="mysql", autocommit=True
        )
        self._lru_base = None          # ← 新增
        self.SAMPLE_INIT = 0.001          

    def stop(self):
        self._stop_evt.set()
        try: self._conn.close()
        except: pass

    # ---------- helpers ----------
    def _snapshot_cpu(self):
        base = f"/proc/{self.pid}/task"
        acc  = collections.Counter(ROW=0, COL=0, SHA=0)
        for tid in os.listdir(base):
            try:
                with open(f"{base}/{tid}/comm") as f:
                    cls = _classify(f.read().strip())
                acc[cls] += _read_jiffies(f"{base}/{tid}/stat")
            except:  # task vanished
                continue
        return acc, time.time()

        # ---------- helpers ----------
    # def _fetch_row_col_bytes(self, cur) -> tuple[int, int]:
    #     """
    #     返回 (row_bytes, col_bytes)

    #     row_bytes  = Buffer Pool 已用页 + 脏页            —— 行存
    #     col_bytes  = lru_cache_usage + exec_mem           —— 列存
    #     """
    #     # —— ROW (InnoDB) ───────────────────────────────
    #     cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages_data'")
    #     pages = int(cur.fetchone()[1])

    #     cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_bytes_dirty'")
    #     dirty = int(cur.fetchone()[1])

    #     cur.execute("SHOW GLOBAL VARIABLES LIKE 'innodb_page_size'")
    #     page_sz = int(cur.fetchone()[1])

    #     row_bytes = pages * page_sz + dirty

    #     # —— COL (IMCI) ────────────────────────────────
    #     # Get the total LRU cache usage (not just the delta)
    #     cur.execute("SHOW GLOBAL STATUS LIKE 'imci_lru_cache_usage'")
    #     lru_total = int(cur.fetchone()[1]) if cur.rowcount else 0

    #     # execution-time memory（只有查询执行时才 >0）
    #     cur.execute("SHOW GLOBAL STATUS LIKE 'imci_execution_memory_usage'")
    #     exec_mem = int(cur.fetchone()[1]) if cur.rowcount else 0

    #     # Use the total LRU cache usage + execution memory
    #     col_bytes = lru_total + exec_mem
        
    #     return row_bytes, col_bytes


    # # ---------- loop ----------
    # def run(self):
    #     cpu_prev, t_prev = self._snapshot_cpu()
    #     row_prev = col_prev = 0

    #     while not self._stop_evt.is_set():
    #         time.sleep(self.SAMPLE_INIT)
    #         cpu_now, t_now = self._snapshot_cpu()
    #         dt = t_now - t_prev
    #         if dt <= 0: continue

    #         d_row = max(0, cpu_now["ROW"] - cpu_prev["ROW"])
    #         d_col = max(0, cpu_now["COL"] - cpu_prev["COL"])
    #         row_pct = 100 * d_row / (_HZ * _NCPU * dt)
    #         col_pct = 100 * d_col / (_HZ * _NCPU * dt)

    #         try:
    #             with self._conn.cursor() as c:
    #                 row_b, col_b = self._fetch_row_col_bytes(c)
    #         except Exception:
    #             row_b, col_b = row_prev, col_prev

    #         row_prev, col_prev = row_b, col_b
    #         self.out.append((
    #             t_now,
    #             row_pct, col_pct,
    #             row_b / _MB,         # ROW mem MB
    #             col_b / _MB          # COL exec MB
    #         ))

    #         cpu_prev, t_prev = cpu_now, t_now


    def _fetch_row_col_bytes(self, cur) -> tuple[int, int]:
        """
        返回 (row_bytes, col_bytes)
        
        row_bytes = Buffer Pool used + additional InnoDB memory
        col_bytes = IMCI total memory usage
        """
        try:
            # ========== ROW (InnoDB) Memory ==========
            # 1. Buffer pool pages in use
            cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_bytes_data'")
            bp_data = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 2. Dirty pages (unflushed modifications)
            cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_bytes_dirty'")
            bp_dirty = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 3. Adaptive hash index memory
            cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_mem_adaptive_hash'")
            ahi_mem = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 4. Dictionary memory
            cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_mem_dictionary'")
            dict_mem = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # Total row-store memory
            row_bytes = bp_data + ahi_mem + dict_mem
            
            # ========== COL (IMCI) Memory ==========
            # 1. LRU cache for column data
            cur.execute("SHOW GLOBAL STATUS LIKE 'imci_lru_cache_usage'")
            lru_usage = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 2. Execution memory (temporary during query execution)
            cur.execute("SHOW GLOBAL STATUS LIKE 'imci_execution_memory_usage'")
            exec_mem = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 3. Dictionary/metadata memory
            cur.execute("SHOW GLOBAL STATUS LIKE 'imci_dictionary_memory_usage'")
            dict_mem_col = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 4. Block index memory
            cur.execute("SHOW GLOBAL STATUS LIKE 'imci_block_index_memory_usage'")
            idx_mem = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # 5. Compression dictionary memory
            cur.execute("SHOW GLOBAL STATUS LIKE 'imci_compress_dictionary_memory_usage'")
            comp_dict_mem = int(cur.fetchone()[1]) if cur.rowcount else 0
            
            # Total column-store memory
            col_bytes = lru_usage + exec_mem + dict_mem_col + idx_mem + comp_dict_mem
            
            # Alternative: Try to get total IMCI memory if available
            cur.execute("SHOW GLOBAL STATUS LIKE 'imci_total_memory_usage'")
            if cur.rowcount:
                total_imci = int(cur.fetchone()[1])
                col_bytes = max(col_bytes, total_imci)  # Use the larger value
                
        except Exception as e:
            print(f"[WARN] Error fetching memory stats: {e}", file=sys.stderr)
            return 0, 0
            
        return row_bytes, col_bytes


    def run(self):
        """Enhanced monitoring with more frequent sampling and debugging"""
        cpu_prev, t_prev = self._snapshot_cpu()
        
        # Shorter sampling interval for more responsive monitoring
        SAMPLE_INTERVAL = 0.1  # 100ms instead of 1ms
        
        # Debug: print first memory reading
        try:
            with self._conn.cursor() as c:
                row_b, col_b = self._fetch_row_col_bytes(c)
                print(f"[DEBUG] Initial memory: ROW={row_b/_MB:.1f}MB, COL={col_b/_MB:.1f}MB")
        except Exception as e:
            print(f"[DEBUG] Failed to get initial memory: {e}")
        
        sample_count = 0
        while not self._stop_evt.is_set():
            time.sleep(SAMPLE_INTERVAL)
            cpu_now, t_now = self._snapshot_cpu()
            dt = t_now - t_prev
            if dt <= 0: continue

            d_row = max(0, cpu_now["ROW"] - cpu_prev["ROW"])
            d_col = max(0, cpu_now["COL"] - cpu_prev["COL"])
            row_pct = 100 * d_row / (_HZ * _NCPU * dt)
            col_pct = 100 * d_col / (_HZ * _NCPU * dt)

            try:
                with self._conn.cursor() as c:
                    row_b, col_b = self._fetch_row_col_bytes(c)
                    
                    # Debug: print significant changes
                    if sample_count % 50 == 0:  # Every 5 seconds
                        print(f"[DEBUG @{t_now-t_prev:.1f}s] Memory: ROW={row_b/_MB:.1f}MB, COL={col_b/_MB:.1f}MB")
                        
            except Exception as e:
                print(f"[WARN] Memory fetch error: {e}")
                row_b = col_b = 0

            self.out.append((
                t_now,
                row_pct, col_pct,
                row_b / _MB,
                col_b / _MB
            ))

            cpu_prev, t_prev = cpu_now, t_now
            sample_count += 1



def _plot_util(samples, fname, title):
    if not samples:                    # ← NEW: 无采样直接返回
        print(f"[WARN] no util samples for {title}")
        return
    t0      = samples[0][0]
    xs      = [s[0]-t0 for s in samples]
    row_cpu = [s[1]     for s in samples]
    col_cpu = [s[2]     for s in samples]
    row_mem = [s[3]     for s in samples]
    col_mem = [s[4]     for s in samples]

    fig, ax1 = plt.subplots()
    ax1.plot(xs, row_cpu, label="ROW cpu%")
    ax1.plot(xs, col_cpu, label="COL cpu%")
    ax1.set_ylabel("cpu %"); ax1.legend(loc="upper left")

    # ── 右侧 mem 轴 ───────────────────────
    ax2 = ax1.twinx()
    ax2.plot(xs, row_mem, "--", label="ROW mem MB")
    ax2.plot(xs, col_mem,  ":", label="COL exec MB")
    ax2.set_ylabel("mem (MB)")

    # 给曲线留 15 % 余量，避免图例/折线挤在顶端
    ymax_mem = max(max(row_mem), max(col_mem), 1)      # 防 0
    ax2.set_ylim(0, ymax_mem * 1.15)

    # 右侧图例
    ax2.legend(loc="upper right")

    plt.title(f"Engine utilisation – {title}")
    plt.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ────────── run one routing mode ──────────
def run_mode(
    tag: str, tasks: List[Tuple[str, str]], arrivals: List[float], args
):
    n = len(tasks)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=68, leave=False)
    bench_start = time.perf_counter()

    lock = Lock()
    threads: List[Thread] = []
    for i, (job, rel) in enumerate(zip(tasks, arrivals)):
        th = Thread(
            target=execute_query,
            args=(
                i,
                job,
                rel,
                ROUTING_MODES[tag],
                args,
                lat,
                bench_start,
                bar,
                lock,
                tag,
            ),
            daemon=True,
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    bar.close()
    return time.perf_counter() - bench_start, lat

# ─────────────────────────────── main ───────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset")
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--timeout", type=int, default=60_000)
    pa.add_argument("--limit", "-n", type=int, default=1000)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--rounds", type=int, default=3)
    pa.add_argument(
        "--scenario",
        choices=["default", "phase_switch", "mixed_mismatch", "resource_spike"],
        default="default",
        help="压测场景",
    )

    # arrival pattern
    grp = pa.add_mutually_exclusive_group()
    grp.add_argument("--poisson_qps", type=float, default=100)
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
    pa.add_argument("--blocker_db", default="tpch_sf100")
    pa.add_argument("--blocker_repeat", type=int, default=5)

    pa.add_argument("--out", default="routing_bench.csv")
    pa.add_argument("--AP_heavy", action="store_true")
    pa.add_argument("--TP_heavy", action="store_true")

    pa.add_argument("--mysqld_pid", type=int, required=True)
    pa.add_argument("--enable_resource_control", action="store_true",
                help="在同一条连接里动态切换 rc_tp / rc_ap")
    args = pa.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # 0. server settings
    apply_server_settings(args.row_conc, args.col_pool)

    # optional resource spike blocker
    if args.scenario == "resource_spike":
        sql_text = (
            "SELECT SUM(l_extendedprice) FROM lineitem "
            "WHERE l_shipdate < '1996-01-01';"
        )
        Thread(
            target=blocker_worker,
            args=(args.blocker_db, sql_text, args.blocker_repeat),
            daemon=True,
        ).start()
        time.sleep(1)

    # 1. build tasks according to scenario
    if args.scenario == "phase_switch":
        tasks = build_phase_switch(data_dir)
    elif args.scenario == "mixed_mismatch":
        tasks = build_mixed_mismatch(data_dir, args)
    else:
        tasks: List[Tuple[str, str]] = []
        datasets = [args.dataset] if args.dataset else ALL_DATASETS
        for ds in datasets:
            qs = load_mismatched_from_csv(
                data_dir / ds / "query_costs.csv", args.TP_heavy, args.AP_heavy
            )
            tasks += [(ds, q) for q in qs]

    if not tasks:
        print("no queries found", file=sys.stderr)
        sys.exit(1)

    random.shuffle(tasks)
    tasks = tasks[: args.limit] if 0 < args.limit < len(tasks) else tasks
    print(
        f"Loaded {len(tasks)} queries "
        f"from {len(set(db for db, _ in tasks))} dataset(s)"
    )

    # 2. arrival list (reuse across rounds)
    if args.burst:
        arrivals = burst_arrivals(
            len(tasks),
            args.burst_qps,
            args.pause_ms,
            args.burst_len,
            args.seed,
        )
    elif args.saw:
        arrivals = sawtooth_arrivals(
            len(tasks),
            args.base_qps,
            args.peak_qps,
            args.period_ms,
            args.seed,
        )
    else:
        qps = args.poisson_qps or 20.0
        if args.scenario == "phase_switch":
            arrivals = sawtooth_arrivals(len(tasks), 50, 500, 10_000, args.seed)
        elif args.scenario == "resource_spike":
            arrivals = burst_arrivals(len(tasks), 300, 50, 100, args.seed)
        else:
            arrivals = poisson_arrivals(len(tasks), 1.0 / qps, args.seed)

    # 3. benchmark rounds
    agg = {m: {"mk": [], "avg": [], "p95": []} for m in ROUTING_MODES}

    for rd in range(1, args.rounds + 1):
        print(f"\n================  Round {rd}/{args.rounds}  ================")
        for tag in ROUTING_MODES:
            # if not tag.startswith("lightgbm"):
            #     continue
            print(f"\n=== {tag} ===")
            samples: List[Tuple] = []
            mon = UtilMonitor(args.mysqld_pid, samples)
            mon.start()

            mk, lats = run_mode(tag, tasks, arrivals, args)

            mon.stop()
            mon.join()
            _plot_util(samples, f"util_{tag}_r{rd}.png", f"{tag} round{rd}")

            ok = [v for v in lats if v not in (None, args.timeout / 1_000)]
            to = [v for v in lats if v == args.timeout / 1_000]
            avg_lat = mean(ok + to) if (ok or to) else float("nan")
            p95_lat = p95(ok + to)

            print(
                f"(seconds) makespan {mk:.2f}  "
                f"avg_lat {avg_lat:.4f}  p95_lat {p95_lat:.4f}  "
                f"ok {len(ok)}  to {len(to)}  "
                f"fail {len(lats) - len(ok) - len(to)}"
            )

            agg[tag]["mk"].append(mk)
            agg[tag]["avg"].append(avg_lat)
            agg[tag]["p95"].append(p95_lat)

        print("\n------------------------------------------------------------")

    # 4. 输出汇总
    print("\n################### 最终汇总 ###################")

    def fmt(lst, prec=2):
        if not lst:
            return "nan"
        m = mean(lst)
        s = stdev(lst) if len(lst) > 1 else 0.0
        return f"{m:.{prec}f} ±{s:.{prec}f}"

    header = ("MODE", "makespan(s)", "avg_latency(s)", "p95_latency(s)")
    print(
        f"{header[0]:<18}{header[1]:>15}"
        f"{header[2]:>18}{header[3]:>18}"
    )
    for tag in (
        "cost_threshold",
        "hybrid_optimizer",
        "lightgbm_static",
        "lightgbm_dynamic",
    ):
        print(
            f"{tag:<18}"
            f"{fmt(agg[tag]['mk'], 2):>15}"
            f"{fmt(agg[tag]['avg'], 4):>18}"
            f"{fmt(agg[tag]['p95'], 4):>18}"
        )

    print(f"\nSaved last‑round latencies to {args.out}")


if __name__ == "__main__":
    main()
