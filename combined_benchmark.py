#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_benchmark.py - Enhanced unified benchmark script for routing modes
Includes: warmup, RL tracking, resource monitoring, multi-n support
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import threading
import collections
import subprocess
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from statistics import mean, stdev
from threading import Event, Lock, Thread
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import pymysql
import pymysql.err

# ===== Configuration =====
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
        "SET GLOBAL max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = OFF",
        "SET fann_model_routing_enabled = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = OFF",
    ],
    "hybrid_optimizer": [
        "SET GLOBAL max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 50000",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled = OFF",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = OFF",
    ],
    "lightgbm_static": [
        "SET GLOBAL max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = OFF",
    ],
    "lightgbm_dynamic": [
        "SET GLOBAL max_user_connections=10001",
        "SET use_imci_engine = ON",
        "SET cost_threshold_for_imci = 1",
        "SET hybrid_opt_dispatch_enabled = ON",
        "SET fann_model_routing_enabled = ON",
        "SET GLOBAL hybrid_opt_fetch_imci_stats_thread_enabled = ON",
        "SET use_mm1_time = ON",  # CRITICAL: This enables adaptive mode!
    ],
}

# Thread-local storage for connection reuse
_thread_locals = threading.local()
_plan_cache: dict[int, str] = {}
_cache_lock = threading.Lock()

# Global constants for monitoring
_HZ = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
_NCPU = os.cpu_count()
_MB = 1024 * 1024

# Thread classification for resource monitoring
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

# ===== Utility Functions =====
def get_conn(db: str):
    """Get thread-local connection"""
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

def p95(vals: List[float]) -> float:
    """Calculate 95th percentile"""
    if not vals:
        return float("nan")
    vals_sorted = sorted(vals)
    k = int(0.95 * (len(vals_sorted) - 1))
    return vals_sorted[k]

def _decide_imci(plan_json: str) -> int:
    """Determine if plan uses IMCI (column) engine"""
    return 0 if '"query_block"' in plan_json else 1

def _explain_and_decide(cur, sql: str) -> str:
    """Use EXPLAIN to determine row/column engine"""
    key = hash(sql)
    with _cache_lock:
        if key in _plan_cache:
            return _plan_cache[key]

    cur.execute(f"EXPLAIN FORMAT='json' {sql}")
    plan_json = cur.fetchone()[0]
    engine = "COL" if _decide_imci(plan_json) else "ROW"

    with _cache_lock:
        _plan_cache[key] = engine
    return engine

def _classify(comm: str) -> str:
    """Classify thread by name"""
    for p in COL_PREFIX:
        if comm.startswith(p):
            return "COL"
    for p in SHR_PREFIX:
        if comm.startswith(p):
            return "SHA"
    return "ROW"

def _read_jiffies(stat_path: str) -> int:
    """Read CPU jiffies from stat file - CORRECTED"""
    with open(stat_path) as f:
        line = f.read()
    
    # Find the end of the comm field (enclosed in parentheses)
    start = line.find('(')
    end = line.rfind(')')
    
    # Split fields after the comm field
    fields = line[end + 2:].split()
    
    # Fields 11 and 12 (0-indexed) are utime and stime
    # But after removing pid and comm, they are at indices 11 and 12
    utime = int(fields[11])
    stime = int(fields[12])
    
    return utime + stime

# ===== Query Loading Functions =====
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
                    continue
                if TP_heavy and row_t >= col_t:
                    continue
                if AP_heavy and col_t >= row_t:
                    continue
                res.append(r["query"].rstrip(";").replace('"', ""))
            except Exception:
                pass
    return res

# ===== Arrival Generators =====
def poisson_arrivals(n: int, mean_sec: float, seed: int) -> List[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_sec)
        out.append(t)
    return out
class ResourceMonitor(Thread):
    """Monitor CPU and memory usage for row/column engines"""
    def __init__(self, pid: int, interval: float = 1.0):  # Increased interval
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.samples = []
        self._stop_evt = Event()
        self._conn = None
        
    def stop(self):
        self._stop_evt.set()
        if self._conn:
            try:
                self._conn.close()
            except:
                pass
    
    def _snapshot_cpu(self):
        """Get CPU usage by thread classification"""
        base = f"/proc/{self.pid}/task"
        acc = collections.Counter(ROW=0, COL=0, SHA=0)
        
        try:
            # Also get total process CPU for validation
            with open(f"/proc/{self.pid}/stat") as f:
                proc_stat = f.read()
            proc_jiffies = self._parse_proc_stat(proc_stat)
            
            for tid in os.listdir(base):
                try:
                    # Read thread name
                    with open(f"{base}/{tid}/comm") as f:
                        comm = f.read().strip()
                    cls = _classify(comm)
                    
                    # Read thread CPU
                    with open(f"{base}/{tid}/stat") as f:
                        stat = f.read()
                    jiffies = self._parse_thread_stat(stat)
                    acc[cls] += jiffies
                except:
                    continue
                    
        except Exception as e:
            print(f"CPU snapshot error: {e}")
            
        return acc, time.time()
    
    def _parse_proc_stat(self, stat_line: str) -> int:
        """Parse /proc/[pid]/stat for total process CPU"""
        # Find the closing paren of the comm field
        end = stat_line.rfind(')')
        fields = stat_line[end + 2:].split()
        
        # utime (field 14) + stime (field 15) + cutime (16) + cstime (17)
        # Indices are 11, 12, 13, 14 after removing first 3 fields
        utime = int(fields[11])
        stime = int(fields[12])
        cutime = int(fields[13])
        cstime = int(fields[14])
        
        return utime + stime + cutime + cstime
    
    def _parse_thread_stat(self, stat_line: str) -> int:
        """Parse /proc/[pid]/task/[tid]/stat for thread CPU"""
        # Similar parsing but for thread
        end = stat_line.rfind(')')
        fields = stat_line[end + 2:].split()
        
        utime = int(fields[11])
        stime = int(fields[12])
        
        return utime + stime
    
    def _fetch_memory_stats(self, cur) -> tuple[int, int]:
        """Fetch memory usage for row and column engines"""
        try:
            # Row (InnoDB) memory - more comprehensive
            queries = [
                ("Innodb_buffer_pool_bytes_data", "bp_data"),
                ("Innodb_buffer_pool_bytes_dirty", "bp_dirty"),
                ("Innodb_mem_adaptive_hash", "ahi"),
                ("Innodb_mem_dictionary", "dict"),
                ("Innodb_mem_total", "total")
            ]
            
            row_bytes = 0
            for var, _ in queries:
                try:
                    cur.execute(f"SHOW GLOBAL STATUS LIKE '{var}'")
                    if cur.rowcount:
                        row_bytes += int(cur.fetchone()[1])
                except:
                    pass
            
            # If we got Innodb_mem_total, use it alone
            cur.execute("SHOW GLOBAL STATUS LIKE 'Innodb_mem_total'")
            if cur.rowcount:
                total = int(cur.fetchone()[1])
                if total > 0:
                    row_bytes = total
            
            # Column (IMCI) memory
            col_bytes = 0
            imci_vars = [
                'imci_lru_cache_usage',
                'imci_execution_memory_usage',
                'imci_mem_pool_size'
            ]
            
            for var in imci_vars:
                try:
                    cur.execute(f"SHOW GLOBAL STATUS LIKE '{var}'")
                    if cur.rowcount:
                        col_bytes += int(cur.fetchone()[1])
                except:
                    pass
            
            return row_bytes, col_bytes
            
        except Exception as e:
            print(f"Memory fetch error: {e}")
            return 0, 0
    
    def run(self):
        """Main monitoring loop"""
        try:
            self._conn = pymysql.connect(
                host=HOST, port=PORT, user=USER, password=PASS,
                db="mysql", autocommit=True
            )
        except Exception as e:
            print(f"Monitor connection failed: {e}")
            return
        
        # Get initial CPU snapshot
        cpu_prev, t_prev = self._snapshot_cpu()
        
        # Get system info
        try:
            with open('/proc/stat') as f:
                for line in f:
                    if line.startswith('cpu '):
                        # Count CPUs by counting cpu0, cpu1, etc
                        ncpus = sum(1 for l in f if l.startswith('cpu') and l[3].isdigit())
                        break
            if ncpus == 0:
                ncpus = os.cpu_count() or 1
        except:
            ncpus = os.cpu_count() or 1
        
        while not self._stop_evt.is_set():
            time.sleep(self.interval)
            
            # CPU metrics
            cpu_now, t_now = self._snapshot_cpu()
            dt = t_now - t_prev
            
            if dt > 0:
                # Calculate CPU usage percentage
                d_row = max(0, cpu_now["ROW"] - cpu_prev["ROW"])
                d_col = max(0, cpu_now["COL"] - cpu_prev["COL"])
                d_sha = max(0, cpu_now["SHA"] - cpu_prev["SHA"])
                
                # Convert jiffies to percentage
                # Total available jiffies = HZ * ncpus * dt
                total_jiffies = _HZ * ncpus * dt
                
                row_cpu_pct = 100.0 * d_row / total_jiffies
                col_cpu_pct = 100.0 * d_col / total_jiffies
                sha_cpu_pct = 100.0 * d_sha / total_jiffies
                
                # Memory metrics
                try:
                    with self._conn.cursor() as cur:
                        row_bytes, col_bytes = self._fetch_memory_stats(cur)
                except:
                    row_bytes = col_bytes = 0
                
                # System-wide CPU check
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                except:
                    cpu_percent = 0
                
                sample = {
                    'timestamp': t_now,
                    'row_cpu_pct': row_cpu_pct,
                    'col_cpu_pct': col_cpu_pct,
                    'sha_cpu_pct': sha_cpu_pct,
                    'total_cpu_pct': row_cpu_pct + col_cpu_pct + sha_cpu_pct,
                    'system_cpu_pct': cpu_percent,
                    'row_mem_mb': row_bytes / _MB,
                    'col_mem_mb': col_bytes / _MB
                }
                
                self.samples.append(sample)
                
                # Debug logging every 10 samples
                if len(self.samples) % 10 == 0:
                    print(f"[Monitor] CPU: ROW={row_cpu_pct:.1f}% COL={col_cpu_pct:.1f}% "
                          f"Total={sample['total_cpu_pct']:.1f}% System={cpu_percent:.1f}%")
                
                cpu_prev, t_prev = cpu_now, t_now

# ===== RL Statistics Extraction =====
def extract_rl_stats(log_path: str = "/home/wuy/mypolardb/db/log/master-error.log") -> Dict:
    """Extract RL learning statistics from MySQL error log"""
    stats = {
        'total_feedback': 0,
        'col_choices': 0,
        'row_choices': 0,
        'avg_reward': 0,
        'final_ema': 0,
        'exploration_ratio': 0
    }
    
    try:
        # Try to read log with sudo if needed
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
        except PermissionError:
            result = subprocess.run(
                ['sudo', 'grep', '[RL] feedback', log_path],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
            else:
                return stats
        
        # Parse RL feedback entries
        rewards = []
        for line in lines:
            if '[RL] feedback' in line:
                stats['total_feedback'] += 1
                if 'col=Y' in line:
                    stats['col_choices'] += 1
                else:
                    stats['row_choices'] += 1
                
                # Extract reward
                import re
                match = re.search(r'Δ=([-\d.]+)', line)
                if match:
                    rewards.append(float(match.group(1)))
                
                # Extract EMA
                match = re.search(r'EMA=([-\d.]+)', line)
                if match:
                    stats['final_ema'] = float(match.group(1))
                
                # Check exploration
                if 'explore_bonus=0.10' in line or 'explore_bonus=0.50' in line:
                    stats['exploration_ratio'] += 1
        
        if rewards:
            stats['avg_reward'] = mean(rewards)
        
        if stats['total_feedback'] > 0:
            stats['exploration_ratio'] /= stats['total_feedback']
            
    except Exception as e:
        print(f"Warning: Could not extract RL stats: {e}")
    
    return stats

# ===== Query Execution =====
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

    # Wait for arrival time
    wait = bench_start + arrival_rel - time.perf_counter()
    if wait > 0:
        time.sleep(wait)

    try:
        conn = get_conn(db)
        cur = conn.cursor()

        # Set session parameters on first use
        if not getattr(_thread_locals, "sess_inited", False):
            for s in sess_sql:
                cur.execute(s)
            _thread_locals.sess_inited = True

        # Resource control if enabled
        if getattr(args, "enable_resource_control", False):
            eng = _explain_and_decide(cur, sql)
            rc = "rc_ap" if eng == "COL" else "rc_tp"
            cid = conn.thread_id()
            cur.execute(f"SET POLAR_RESOURCE_CONTROL {rc} FOR CONNECTION {cid}")

        # Set query timeout
        cur.execute(f"SET max_execution_time = {args.timeout}")

    except Exception as e:
        snippet = sql.replace("\n", " ")[:120]
        print(f"[{tag} #{idx}] {e}\n   SQL: {snippet}", file=sys.stderr)
        with lock:
            lat[idx] = None
            bar.update()
        return

    # Execute and time
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
        cur.close()

# ===== Run Single Mode =====
def run_mode(
    tag: str, 
    tasks: List[Tuple[str, str]], 
    arrivals: List[float], 
    args,
    monitor_resources: bool = True
) -> Tuple[float, List[float], Dict, Optional[List[Dict]]]:
    """Run benchmark for a single routing mode"""
    n = len(tasks)
    lat: List[float | None] = [None] * n
    bar = tqdm(total=n, desc=tag, ncols=68, leave=False)
    bench_start = time.perf_counter()
    
    # Start resource monitor if requested
    monitor = None
    if monitor_resources:
        monitor = ResourceMonitor(args.mysqld_pid)
        monitor.start()

    lock = Lock()
    threads: List[Thread] = []
    for i, (job, rel) in enumerate(zip(tasks, arrivals)):
        th = Thread(
            target=execute_query,
            args=(
                i, job, rel, ROUTING_MODES[tag], args,
                lat, bench_start, bar, lock, tag,
            ),
            daemon=True,
        )
        th.start()
        threads.append(th)
    
    for th in threads:
        th.join()
    bar.close()
    
    # Stop resource monitor
    if monitor:
        monitor.stop()
        monitor.join()
    
    makespan = time.perf_counter() - bench_start
    
    # Calculate statistics
    ok = [v for v in lat if v not in (None, args.timeout / 1000)]
    to = [v for v in lat if v == args.timeout / 1000]
    avg_lat = mean(ok + to) if (ok or to) else float("nan")
    p95_lat = p95(ok + to)
    
    stats = {
        "ok_count": len(ok),
        "timeout_count": len(to),
        "fail_count": len(lat) - len(ok) - len(to),
        "avg_latency": avg_lat,
        "p95_latency": p95_lat,
        "makespan": makespan
    }
    
    # Extract RL stats for dynamic mode
    if tag == "lightgbm_dynamic":
        rl_stats = extract_rl_stats()
        stats['rl_stats'] = rl_stats
    
    resource_samples = monitor.samples if monitor else None
    
    return makespan, lat, stats, resource_samples

# ===== Warmup Function =====
def run_warmup(tasks: List[Tuple[str, str]], args, warmup_size: int = 100):
    """Run warmup phase for adaptive methods"""
    print("\n=== Running warmup phase ===")
    warmup_tasks = tasks[:min(warmup_size, len(tasks))]
    warmup_arrivals = poisson_arrivals(len(warmup_tasks), 1.0 / args.poisson_qps, args.seed)
    
    # Only warmup the adaptive method
    for mode in ["lightgbm_dynamic"]:
        print(f"Warming up {mode}...")
        run_mode(mode, warmup_tasks, warmup_arrivals, args, monitor_resources=False)
    
    print("Warmup completed\n")
    time.sleep(2)


def plot_resources(resource_data: Dict[str, List[Dict]], output_prefix: str):
    """Plot resource usage for all modes with fine-grained subplots"""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 4x2 grid for each mode's CPU and memory
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    modes = ["cost_threshold", "hybrid_optimizer", "lightgbm_static", "lightgbm_dynamic"]
    colors = {
        'ROW': '#1f77b4',  # blue
        'COL': '#ff7f0e',  # orange
        'SHA': '#2ca02c',  # green
        'Total': '#d62728', # red
        'System': '#9467bd' # purple
    }
    
    for i, mode in enumerate(modes):
        if mode not in resource_data or not resource_data[mode]:
            continue
            
        samples = resource_data[mode]
        t0 = samples[0]['timestamp']
        times = [(s['timestamp'] - t0) for s in samples]
        
        # CPU subplot for this mode
        ax_cpu = fig.add_subplot(gs[i, 0])
        
        # Plot CPU usage with different line styles
        row_cpu = [s['row_cpu_pct'] for s in samples]
        col_cpu = [s['col_cpu_pct'] for s in samples]
        sha_cpu = [s.get('sha_cpu_pct', 0) for s in samples]
        total_cpu = [s['total_cpu_pct'] for s in samples]
        
        # Apply smoothing for better visibility
        from scipy.ndimage import uniform_filter1d
        window = min(5, len(times) // 10) if len(times) > 10 else 1
        
        ax_cpu.plot(times, uniform_filter1d(row_cpu, window), 
                   label='ROW', color=colors['ROW'], linewidth=2, alpha=0.8)
        ax_cpu.plot(times, uniform_filter1d(col_cpu, window), 
                   label='COL', color=colors['COL'], linewidth=2, alpha=0.8)
        if any(sha_cpu):
            ax_cpu.plot(times, uniform_filter1d(sha_cpu, window), 
                       label='Shared', color=colors['SHA'], linewidth=1.5, alpha=0.6, linestyle='--')
        ax_cpu.plot(times, uniform_filter1d(total_cpu, window), 
                   label='Total', color=colors['Total'], linewidth=2.5, alpha=0.9, linestyle='-.')
        
        ax_cpu.set_xlabel('Time (s)')
        ax_cpu.set_ylabel('CPU %')
        ax_cpu.set_title(f'{mode} - CPU Usage', fontsize=12, fontweight='bold')
        ax_cpu.legend(loc='upper right', framealpha=0.9)
        ax_cpu.grid(True, alpha=0.3)
        ax_cpu.set_ylim(0, max(max(total_cpu) * 1.1, 10))
        
        # Add annotations for peaks
        peak_idx = total_cpu.index(max(total_cpu))
        if peak_idx < len(times):
            ax_cpu.annotate(f'Peak: {max(total_cpu):.1f}%', 
                           xy=(times[peak_idx], max(total_cpu)),
                           xytext=(times[peak_idx] + 5, max(total_cpu) + 5),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=9)
        
        # Memory subplot for this mode
        ax_mem = fig.add_subplot(gs[i, 1])
        
        row_mem = [s['row_mem_mb'] for s in samples]
        col_mem = [s['col_mem_mb'] for s in samples]
        
        ax_mem.plot(times, uniform_filter1d(row_mem, window), 
                   label='ROW', color=colors['ROW'], linewidth=2, alpha=0.8)
        ax_mem.plot(times, uniform_filter1d(col_mem, window), 
                   label='COL', color=colors['COL'], linewidth=2, alpha=0.8)
        
        # Add total memory line
        total_mem = [r + c for r, c in zip(row_mem, col_mem)]
        ax_mem.plot(times, uniform_filter1d(total_mem, window), 
                   label='Total', color=colors['Total'], linewidth=2.5, alpha=0.9, linestyle='-.')
        
        ax_mem.set_xlabel('Time (s)')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_title(f'{mode} - Memory Usage', fontsize=12, fontweight='bold')
        ax_mem.legend(loc='upper right', framealpha=0.9)
        ax_mem.grid(True, alpha=0.3)
        ax_mem.set_ylim(0, max(max(total_mem) * 1.1, 100) if total_mem else 100)
    
    plt.suptitle(f'Resource Usage Comparison - {output_prefix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_resources_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional comparison plots
    plot_resource_comparison(resource_data, output_prefix)
    plot_resource_timeline(resource_data, output_prefix)

def plot_resource_comparison(resource_data: Dict[str, List[Dict]], output_prefix: str):
    """Create comparison plots showing all modes on same axes"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    mode_styles = {
        "cost_threshold": {'color': '#e74c3c', 'linestyle': '-', 'marker': 'o', 'markersize': 4},
        "hybrid_optimizer": {'color': '#3498db', 'linestyle': '--', 'marker': 's', 'markersize': 4},
        "lightgbm_static": {'color': '#2ecc71', 'linestyle': '-.', 'marker': '^', 'markersize': 4},
        "lightgbm_dynamic": {'color': '#f39c12', 'linestyle': ':', 'marker': 'D', 'markersize': 4}
    }
    
    for mode, samples in resource_data.items():
        if not samples:
            continue
            
        t0 = samples[0]['timestamp']
        times = [(s['timestamp'] - t0) for s in samples]
        style = mode_styles.get(mode, {})
        
        # Subsample for markers to avoid cluttering
        marker_interval = max(1, len(times) // 20)
        
        # ROW CPU comparison
        row_cpu = [s['row_cpu_pct'] for s in samples]
        ax1.plot(times, row_cpu, label=mode, **style, markevery=marker_interval)
        
        # COL CPU comparison
        col_cpu = [s['col_cpu_pct'] for s in samples]
        ax2.plot(times, col_cpu, label=mode, **style, markevery=marker_interval)
        
        # Total CPU comparison
        total_cpu = [s['total_cpu_pct'] for s in samples]
        ax3.plot(times, total_cpu, label=mode, **style, markevery=marker_interval)
        
        # Total Memory comparison
        total_mem = [s['row_mem_mb'] + s['col_mem_mb'] for s in samples]
        ax4.plot(times, total_mem, label=mode, **style, markevery=marker_interval)
    
    # Configure subplots
    for ax, title, ylabel in [
        (ax1, 'ROW Engine CPU Usage', 'CPU %'),
        (ax2, 'COL Engine CPU Usage', 'CPU %'),
        (ax3, 'Total CPU Usage', 'CPU %'),
        (ax4, 'Total Memory Usage', 'Memory (MB)')
    ]:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        if 'CPU' in ylabel:
            ax.set_ylim(0, None)
    
    plt.suptitle('Resource Usage Mode Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_resources_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_resource_timeline(resource_data: Dict[str, List[Dict]], output_prefix: str):
    """Create timeline view with heatmaps for resource usage"""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom colormaps
    cpu_cmap = LinearSegmentedColormap.from_list('cpu', ['white', 'yellow', 'orange', 'red'])
    mem_cmap = LinearSegmentedColormap.from_list('mem', ['white', 'lightblue', 'blue', 'darkblue'])
    
    fig, axes = plt.subplots(4, 2, figsize=(18, 12))
    
    modes = ["cost_threshold", "hybrid_optimizer", "lightgbm_static", "lightgbm_dynamic"]
    
    for i, mode in enumerate(modes):
        if mode not in resource_data or not resource_data[mode]:
            continue
            
        samples = resource_data[mode]
        
        # Create time bins for heatmap (1 second bins)
        t0 = samples[0]['timestamp']
        max_time = samples[-1]['timestamp'] - t0
        time_bins = int(max_time) + 1
        
        # Initialize data arrays
        cpu_data = np.zeros((3, time_bins))  # ROW, COL, Total
        mem_data = np.zeros((2, time_bins))  # ROW, COL
        
        # Fill data arrays
        for sample in samples:
            t_idx = min(int(sample['timestamp'] - t0), time_bins - 1)
            cpu_data[0, t_idx] = max(cpu_data[0, t_idx], sample['row_cpu_pct'])
            cpu_data[1, t_idx] = max(cpu_data[1, t_idx], sample['col_cpu_pct'])
            cpu_data[2, t_idx] = max(cpu_data[2, t_idx], sample['total_cpu_pct'])
            mem_data[0, t_idx] = max(mem_data[0, t_idx], sample['row_mem_mb'])
            mem_data[1, t_idx] = max(mem_data[1, t_idx], sample['col_mem_mb'])
        
        # CPU heatmap
        ax_cpu = axes[i, 0]
        im_cpu = ax_cpu.imshow(cpu_data, aspect='auto', cmap=cpu_cmap, 
                              extent=[0, max_time, 0, 3], origin='lower')
        ax_cpu.set_yticks([0.5, 1.5, 2.5])
        ax_cpu.set_yticklabels(['ROW', 'COL', 'Total'])
        ax_cpu.set_xlabel('Time (s)')
        ax_cpu.set_title(f'{mode} - CPU Timeline', fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar_cpu = plt.colorbar(im_cpu, ax=ax_cpu, fraction=0.046, pad=0.04)
        cbar_cpu.set_label('CPU %', rotation=270, labelpad=15)
        
        # Memory heatmap
        ax_mem = axes[i, 1]
        im_mem = ax_mem.imshow(mem_data, aspect='auto', cmap=mem_cmap,
                              extent=[0, max_time, 0, 2], origin='lower')
        ax_mem.set_yticks([0.5, 1.5])
        ax_mem.set_yticklabels(['ROW', 'COL'])
        ax_mem.set_xlabel('Time (s)')
        ax_mem.set_title(f'{mode} - Memory Timeline', fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar_mem = plt.colorbar(im_mem, ax=ax_mem, fraction=0.046, pad=0.04)
        cbar_mem.set_label('Memory (MB)', rotation=270, labelpad=15)
    
    plt.suptitle('Resource Usage Timeline Heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_resources_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()

# Also update the analyze_and_display_results function to create summary resource plots
def plot_resource_summary(all_results: Dict, output_prefix: str = "summary"):
    """Create summary plots comparing resource efficiency across query counts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    query_counts = sorted(all_results.keys())
    modes = ["cost_threshold", "hybrid_optimizer", "lightgbm_static", "lightgbm_dynamic"]
    mode_colors = {
        "cost_threshold": '#e74c3c',
        "hybrid_optimizer": '#3498db', 
        "lightgbm_static": '#2ecc71',
        "lightgbm_dynamic": '#f39c12'
    }
    
    # Extract metrics for each query count
    metrics = {mode: {'makespan': [], 'avg_lat': [], 'p95_lat': [], 'efficiency': []} 
              for mode in modes}
    
    for n in query_counts:
        if n not in all_results:
            continue
        results_n = all_results[n]
        
        for mode in modes:
            if mode in results_n:
                data = results_n[mode]
                metrics[mode]['makespan'].append(mean(data['makespans']))
                metrics[mode]['avg_lat'].append(mean(data['avg_latencies']))
                metrics[mode]['p95_lat'].append(mean(data['p95_latencies']))
                # Calculate efficiency as queries per second
                efficiency = n / mean(data['makespans']) if data['makespans'] else 0
                metrics[mode]['efficiency'].append(efficiency)
    
    # Plot makespan scaling
    for mode in modes:
        if metrics[mode]['makespan']:
            ax1.plot(query_counts[:len(metrics[mode]['makespan'])], 
                    metrics[mode]['makespan'], 
                    label=mode, color=mode_colors[mode], 
                    marker='o', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Queries')
    ax1.set_ylabel('Makespan (s)')
    ax1.set_title('Makespan Scaling', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot throughput (queries per second)
    for mode in modes:
        if metrics[mode]['efficiency']:
            ax2.plot(query_counts[:len(metrics[mode]['efficiency'])], 
                    metrics[mode]['efficiency'], 
                    label=mode, color=mode_colors[mode], 
                    marker='s', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Queries')
    ax2.set_ylabel('Throughput (queries/s)')
    ax2.set_title('Throughput Scaling', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot average latency
    for mode in modes:
        if metrics[mode]['avg_lat']:
            ax3.plot(query_counts[:len(metrics[mode]['avg_lat'])], 
                    metrics[mode]['avg_lat'], 
                    label=mode, color=mode_colors[mode], 
                    marker='^', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Number of Queries')
    ax3.set_ylabel('Average Latency (s)')
    ax3.set_title('Average Latency Scaling', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot P95 latency
    for mode in modes:
        if metrics[mode]['p95_lat']:
            ax4.plot(query_counts[:len(metrics[mode]['p95_lat'])], 
                    metrics[mode]['p95_lat'], 
                    label=mode, color=mode_colors[mode], 
                    marker='D', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Number of Queries')
    ax4.set_ylabel('P95 Latency (s)')
    ax4.set_title('P95 Latency Scaling', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Scaling Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_performance_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    
# ===== Main Benchmark Function =====
def run_full_benchmark(args):
    """Run complete benchmark for all modes and configurations"""
    
    # Initialize
    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    
    # Load queries
    tasks: List[Tuple[str, str]] = []
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    
    for ds in datasets:
        qs = load_mismatched_from_csv(
            data_dir / ds / "query_costs.csv", 
            args.TP_heavy, 
            args.AP_heavy
        )
        tasks += [(ds, q) for q in qs]
    
    if not tasks:
        print("No queries found", file=sys.stderr)
        return None
    
    random.shuffle(tasks)
    
    # Results storage for all query counts
    all_results = {}
    
    # Run benchmarks for each query count
    for query_count in args.limits:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING WITH {query_count} QUERIES")
        print(f"{'='*60}")
        
        # Select queries for this run
        current_tasks = tasks[:min(query_count, len(tasks))]
        print(f"Using {len(current_tasks)} queries from {len(set(db for db, _ in current_tasks))} dataset(s)")
        
        # Generate arrivals
        qps = args.poisson_qps or 20.0
        arrivals = poisson_arrivals(len(current_tasks), 1.0 / qps, args.seed)
        
        # Run warmup for first query count
        if args.warmup_queries > 0 and query_count == args.limits[0]:
            run_warmup(current_tasks, args, args.warmup_queries)
        
        # Results for this query count
        results_n = {}
        resource_data = {}
        
        # Run benchmarks
        for round_num in range(1, args.rounds + 1):
            print(f"\n--- Round {round_num}/{args.rounds} ---")
            
            for mode in ROUTING_MODES:
                print(f"\nRunning {mode}...")
                makespan, latencies, stats, resources = run_mode(
                    mode, current_tasks, arrivals, args, monitor_resources=True
                )
                
                print(f"  Makespan: {makespan:.2f}s")
                print(f"  Avg latency: {stats['avg_latency']:.4f}s")
                print(f"  P95 latency: {stats['p95_latency']:.4f}s")
                print(f"  OK: {stats['ok_count']}, Timeouts: {stats['timeout_count']}, Failures: {stats['fail_count']}")
                
                if 'rl_stats' in stats:
                    rl = stats['rl_stats']
                    print(f"  RL: {rl['total_feedback']} feedbacks, "
                          f"col_ratio={rl['col_choices']/(rl['total_feedback']+1):.2f}, "
                          f"avg_reward={rl['avg_reward']:.3f}")
                
                # Store results
                if mode not in results_n:
                    results_n[mode] = {
                        "makespans": [],
                        "avg_latencies": [],
                        "p95_latencies": [],
                        "stats": []
                    }
                
                results_n[mode]["makespans"].append(makespan)
                results_n[mode]["avg_latencies"].append(stats['avg_latency'])
                results_n[mode]["p95_latencies"].append(stats['p95_latency'])
                results_n[mode]["stats"].append(stats)
                
                # Store resource data from last round
                if round_num == args.rounds and resources:
                    resource_data[mode] = resources
        
        # Plot resources for this query count
        if resource_data:
            plot_resources(resource_data, f"resources_n{query_count}")
        
        all_results[query_count] = results_n
    
    return all_results

# ===== Results Analysis and Display =====
def analyze_and_display_results(all_results: Dict, args):
    """Analyze and display benchmark results for all query counts"""
    
    # Prepare data for comprehensive table
    table_data = []
    
    for query_count in sorted(all_results.keys()):
        results = all_results[query_count]
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {query_count} QUERIES")
        print(f"{'='*80}")
        
        # Calculate statistics for this query count
        summary = {}
        for mode, data in results.items():
            summary[mode] = {
                "makespan_mean": mean(data["makespans"]),
                "makespan_std": stdev(data["makespans"]) if len(data["makespans"]) > 1 else 0,
                "avg_lat_mean": mean(data["avg_latencies"]),
                "avg_lat_std": stdev(data["avg_latencies"]) if len(data["avg_latencies"]) > 1 else 0,
                "p95_lat_mean": mean(data["p95_latencies"]),
                "p95_lat_std": stdev(data["p95_latencies"]) if len(data["p95_latencies"]) > 1 else 0,
            }
        
        # Print summary for this query count
        print(f"{'Mode':<20} {'Makespan(s)':<20} {'Avg Latency(s)':<20} {'P95 Latency(s)':<20}")
        print("-" * 80)
        
        for mode in ["cost_threshold", "hybrid_optimizer", "lightgbm_static", "lightgbm_dynamic"]:
            if mode in summary:
                s = summary[mode]
                print(f"{mode:<20} "
                      f"{s['makespan_mean']:.2f} ±{s['makespan_std']:.2f}    "
                      f"{s['avg_lat_mean']:.4f} ±{s['avg_lat_std']:.4f}    "
                      f"{s['p95_lat_mean']:.4f} ±{s['p95_lat_std']:.4f}")
                
                # Add to table data
                for metric in ["makespan", "avg_latency", "p95_latency"]:
                    table_data.append({
                        "queries": query_count,
                        "metric": metric,
                        "cost_threshold": f"{summary.get('cost_threshold', {}).get(f'{metric[:-7]}_mean', 0):.4f}",
                        "hybrid_opt": f"{summary.get('hybrid_optimizer', {}).get(f'{metric[:-7]}_mean', 0):.4f}",
                        "lgbm_static": f"{summary.get('lightgbm_static', {}).get(f'{metric[:-7]}_mean', 0):.4f}",
                        "lgbm_adaptive": f"{summary.get('lightgbm_dynamic', {}).get(f'{metric[:-7]}_mean', 0):.4f}"
                    })
        
        # Performance improvements
        if "cost_threshold" in summary:
            baseline = summary["cost_threshold"]
            print("\nImprovements vs cost_threshold:")
            
            for mode in ["hybrid_optimizer", "lightgbm_static", "lightgbm_dynamic"]:
                if mode in summary:
                    s = summary[mode]
                    makespan_imp = (baseline["makespan_mean"] - s["makespan_mean"]) / baseline["makespan_mean"] * 100
                    avg_imp = (baseline["avg_lat_mean"] - s["avg_lat_mean"]) / baseline["avg_lat_mean"] * 100
                    p95_imp = (baseline["p95_lat_mean"] - s["p95_lat_mean"]) / baseline["p95_lat_mean"] * 100
                    
                    print(f"  {mode:<20} Makespan: {makespan_imp:+6.1f}%, "
                          f"Avg: {avg_imp:+6.1f}%, P95: {p95_imp:+6.1f}%")
        
        # Dynamic vs Static comparison
        if "lightgbm_static" in summary and "lightgbm_dynamic" in summary:
            static = summary["lightgbm_static"]
            dynamic = summary["lightgbm_dynamic"]
            
            makespan_imp = (static["makespan_mean"] - dynamic["makespan_mean"]) / static["makespan_mean"] * 100
            avg_imp = (static["avg_lat_mean"] - dynamic["avg_lat_mean"]) / static["avg_lat_mean"] * 100
            p95_imp = (static["p95_lat_mean"] - dynamic["p95_lat_mean"]) / static["p95_lat_mean"] * 100
            
            print(f"\nDynamic vs Static: Makespan: {makespan_imp:+6.1f}%, "
                  f"Avg: {avg_imp:+6.1f}%, P95: {p95_imp:+6.1f}%")
    
    # Save comprehensive results table
    df = pd.DataFrame(table_data)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Print final comparison table
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*100)
    print(f"{'Queries':<10} {'Metric':<15} {'Cost Threshold':<15} {'Hybrid Opt':<15} {'LGBM Static':<15} {'LGBM Adaptive':<15}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['queries']:<10} {row['metric']:<15} {row['cost_threshold']:<15} "
              f"{row['hybrid_opt']:<15} {row['lgbm_static']:<15} {row['lgbm_adaptive']:<15}")
        
        # Add separator after each metric group
        if row['metric'] == 'p95_latency':
            print("-"*100)

    plot_resource_summary(all_results, "summary")

# ===== Main Entry Point =====
def main():
    parser = argparse.ArgumentParser(description="Enhanced combined routing benchmark")
    
    # Data selection
    parser.add_argument("--dataset", help="Specific dataset to use")
    parser.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    parser.add_argument("--AP_heavy", action="store_true")
    parser.add_argument("--TP_heavy", action="store_true")
    
    # Benchmark parameters
    parser.add_argument("--timeout", type=int, default=60_000, help="Query timeout in ms")
    parser.add_argument("-n", "--limits", nargs='+', type=int, default=[100], 
                       help="List of query counts to test")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poisson_qps", type=float, default=50)
    parser.add_argument("--warmup_queries", type=int, default=100, 
                       help="Number of warmup queries (0 to disable)")
    
    # System configuration
    parser.add_argument("--mysqld_pid", type=int, required=True)
    parser.add_argument("--enable_resource_control", action="store_true")
    
    # Output
    parser.add_argument("--output_csv", default="benchmark_results.csv")
    
    args = parser.parse_args()
    
    print(f"Benchmark Configuration:")
    print(f"  Query counts: {args.limits}")
    print(f"  Rounds per count: {args.rounds}")
    print(f"  Warmup queries: {args.warmup_queries}")
    print(f"  QPS: {args.poisson_qps}")
    print(f"  Timeout: {args.timeout}ms")
    
    # Run benchmark
    results = run_full_benchmark(args)
    
    if results:
        analyze_and_display_results(results, args)
    else:
        print("Benchmark failed")
        sys.exit(1)

if __name__ == "__main__":
    main()