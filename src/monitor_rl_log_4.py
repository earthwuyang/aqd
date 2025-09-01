#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_rl_log_4.py — Analyze RL / dispatcher learning progress from MySQL logs

Matches kernel logs like:
  2025-08-08T10:23:20.668492+08:00 ... [RL] feedback col=0 lat=1.0 Δ=-1.662 Δ_EMA=3.232 fp=964b3f
  2025-08-08T10:23:20.673070+08:00 ... PdDispatcher2: cpuR=0.44 memR=0.67 perf=-0.50 rsc=0.24 ω=0.30 fused=-0.28 -> ROW
  2025-08-08T10:23:20.673080+08:00 ... [Hybrid] route=ROW mode=dynamic perf=-0.498 margin=-0.036 u_raw=-1.058 prep_us=453
  2025-08-08T11:57:10.959379+08:00 ... [PdTarget] err=14630.5ms  adj=0.050  lo/hi=[0.53,0.37]  cpuT=0.37  memT=0.43  U=0.00  ψmem=0.00
  ... [LinTS] numeric reset triggered (β=7.10)
"""

import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from typing import List, Dict
import argparse
import sys
import os

# ---------- line parsers ----------

def parse_timestamp(line: str):
    # ISO8601 with optional timezone, e.g. 2025-08-08T10:23:20.667897+08:00
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+(?:[+-]\d{2}:\d{2})?)', line)
    return m.group(1) if m else None

def parse_rl_feedback(line: str) -> Dict:
    """
    [RL] feedback col=0 lat=1.0 Δ=-1.662 Δ_EMA=3.232 fp=964b3f
    (accepts col=0/1 or col=Y/N; lat may be '123.4ms' or plain number)
    """
    if '[RL]' not in line or 'feedback' not in line:
        return {}
    out = {'kind': 'rl_feedback'}
    ts = parse_timestamp(line)
    if ts: out['timestamp'] = ts

    # col flag (0/1 or Y/N)
    cm = re.search(r'col=([01YN])', line)
    if cm:
        token = cm.group(1)
        out['use_col'] = (token in ('1', 'Y'))

    # latency (ms), with or without 'ms' suffix
    lm = re.search(r'lat=([\d.]+)(?:ms)?', line)
    if lm:
        out['latency'] = float(lm.group(1))

    # Δ and Δ_EMA on same line (if present)
    dm = re.search(r'Δ=([-\d.]+)', line)
    if dm:
        out['reward'] = float(dm.group(1))
    em = re.search(r'Δ[_ ]?EMA=([-\d.]+)', line)
    if em:
        out['ema'] = float(em.group(1))

    # fingerprint (optional)
    fp = re.search(r'\bfp=([0-9a-fA-Fx]+)', line)
    if fp:
        out['fp'] = fp.group(1)

    return out

def parse_rl_ema(line: str) -> Dict:
    """Fallback if EMA is logged separately: [RL] Δ=0.123  Δ_EMA=0.456"""
    if '[RL]' not in line or 'Δ_EMA' not in line:
        return {}
    out = {'kind': 'rl_ema'}
    ts = parse_timestamp(line)
    if ts: out['timestamp'] = ts
    em = re.search(r'Δ[_ ]?EMA=([-\d.]+)', line)
    if em: out['ema'] = float(em.group(1))
    dm = re.search(r'Δ=([-\d.]+)', line)
    if dm: out['reward'] = float(dm.group(1))
    return out

def parse_pd_dispatch(line: str) -> Dict:
    """
    PdDispatcher2: cpuR=0.44 memR=0.67 perf=-0.50 rsc=0.24 ω=0.30 fused=-0.28 -> COL|ROW
    """
    if 'PdDispatcher2:' not in line:
        return {}
    rx = re.search(
        r'PdDispatcher2:\s*cpuR=([-\d.]+)\s*memR=([-\d.]+)\s*perf=([-\d.]+)\s*'
        r'rsc=([-\d.]+)\s*(?:ω|omega)=([-\d.]+)\s*fused=([-\d.]+)\s*->\s*(COL|ROW)',
        line
    )
    if not rx:
        return {}
    out = {
        'kind': 'pd',
        'cpuR': float(rx.group(1)),
        'memR': float(rx.group(2)),
        'perf': float(rx.group(3)),
        'res_score': float(rx.group(4)),
        'omega': float(rx.group(5)),
        'fused': float(rx.group(6)),
        'pd_decision': (rx.group(7) == 'COL')
    }
    ts = parse_timestamp(line)
    if ts: out['timestamp'] = ts
    return out

def parse_hybrid_route(line: str) -> Dict:
    """
    [Hybrid] route=ROW mode=dynamic perf=-0.498 margin=-0.036 u_raw=-1.058 prep_us=453
    or older:
    [Hybrid] route=COL mode=static margin=0.123 prep_us=123
    or very old:
    [Hybrid] route=COL  prep_us=123
    """
    if '[Hybrid]' not in line or 'route=' not in line:
        return {}
    # dynamic first
    rx = re.search(
        r'\[Hybrid\]\s*route=(COL|ROW)\s*mode=(\w+)\s*perf=([-\d.]+)\s*'
        r'margin=([-\d.]+)\s*u_raw=([-\d.]+)\s*prep_us=(\d+)',
        line
    )
    if rx:
        out = {
            'kind': 'hybrid',
            'hyb_route_col': (rx.group(1) == 'COL'),
            'mode': rx.group(2),
            'perf': float(rx.group(3)),
            'margin': float(rx.group(4)),
            'u_raw': float(rx.group(5)),
            'prep_us': int(rx.group(6))
        }
    else:
        # static pattern
        rx2 = re.search(
            r'\[Hybrid\]\s*route=(COL|ROW)\s*mode=(\w+)\s*margin=([-\d.]+)\s*prep_us=(\d+)',
            line
        )
        if rx2:
            out = {
                'kind': 'hybrid',
                'hyb_route_col': (rx2.group(1) == 'COL'),
                'mode': rx2.group(2),
                'margin': float(rx2.group(3)),
                'prep_us': int(rx2.group(4))
            }
        else:
            # very old
            rx3 = re.search(r'\[Hybrid\]\s*route=(COL|ROW)\s*prep_us=(\d+)', line)
            if not rx3:
                return {}
            out = {
                'kind': 'hybrid',
                'hyb_route_col': (rx3.group(1) == 'COL'),
                'prep_us': int(rx3.group(2))
            }
    ts = parse_timestamp(line)
    if ts: out['timestamp'] = ts
    return out

def parse_pdtarget(line: str) -> Dict:
    """
    [PdTarget] err=14630.5ms  adj=0.050  lo/hi=[0.53,0.37]  cpuT=0.37  memT=0.43  U=0.00  ψmem=0.00
    """
    if '[PdTarget]' not in line:
        return {}
    rx = re.search(
        r'\[PdTarget\]\s*err=([-\d.]+)ms\s*adj=([-\d.]+)\s*lo/hi=\[([-\d.]+),([-\d.]+)\]\s*'
        r'cpuT=([-\d.]+)\s*memT=([-\d.]+)\s*U=([-\d.]+)\s*(?:ψmem|psimem|psi_mem)=([-\d.]+)',
        line
    )
    if not rx:
        return {}
    out = {
        'kind': 'pdtarget',
        'err_ms': float(rx.group(1)),
        'adj': float(rx.group(2)),
        'lo_row': float(rx.group(3)),
        'hi_row': float(rx.group(4)),
        'cpuT': float(rx.group(5)),
        'memT': float(rx.group(6)),
        'U_cpu': float(rx.group(7)),
        'psi_mem': float(rx.group(8)),
    }
    ts = parse_timestamp(line)
    if ts: out['timestamp'] = ts
    return out

def parse_lints_reset(line: str) -> Dict:
    """[LinTS] numeric reset triggered (β=7.10)"""
    if '[LinTS]' not in line or 'reset' not in line:
        return {}
    rx = re.search(r'\[LinTS\].*?\((?:β|beta)=([-\d.]+)\)', line)
    if not rx:
        return {}
    out = {'kind': 'lints_reset', 'beta': float(rx.group(1))}
    ts = parse_timestamp(line)
    if ts: out['timestamp'] = ts
    return out

# ---------- log reading & stitching ----------

def read_rl_logs(log_path: str) -> List[Dict]:
    """
    Build entries keyed off [RL] feedback lines.
    Augment each with nearby PdDispatcher2 / [Hybrid] / [PdTarget] context.
    """
    entries: List[Dict] = []
    pdtargets: List[Dict] = []
    resets = 0

    # Read lines (try sudo if permission denied)
    try:
        try:
            with open(log_path, 'r', errors='ignore') as f:
                lines = f.readlines()
        except PermissionError:
            print("Using sudo to read log file...")
            result = subprocess.run(
                ['sudo', 'cat', log_path],
                capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                print(f"Error reading log file: {result.stderr}")
                return entries
            lines = result.stdout.splitlines()
    except Exception as e:
        print(f"Error reading logs: {e}")
        return entries

    # Pre-scan for PdTarget and LinTS resets (they're not tied 1:1 to queries)
    for line in lines:
        t = parse_pdtarget(line)
        if t: pdtargets.append(t)
        r = parse_lints_reset(line)
        if r: resets += 1

    # Iterate for RL feedback anchor lines
    for i, line in enumerate(lines):
        fb = parse_rl_feedback(line)
        if not fb:
            continue

        # Backward search for PdDispatcher2 and standalone EMA
        back_N = 60
        ema_val = fb.get('ema')
        pd_ctx = None

        for j in range(i - 1, max(-1, i - back_N), -1):
            l2 = lines[j]
            if ema_val is None:
                ema = parse_rl_ema(l2)
                if ema and 'ema' in ema:
                    ema_val = ema['ema']
            if pd_ctx is None:
                pd = parse_pd_dispatch(l2)
                if pd:
                    pd_ctx = pd
            if ema_val is not None and pd_ctx is not None:
                break

        if ema_val is not None:
            fb['ema'] = ema_val
        if pd_ctx is not None:
            fb.update({
                'cpuR': pd_ctx['cpuR'],
                'memR': pd_ctx['memR'],
                'pd_perf': pd_ctx['perf'],
                'res_score': pd_ctx['res_score'],
                'omega': pd_ctx['omega'],
                'fused': pd_ctx['fused'],
                'pd_decision': pd_ctx['pd_decision']
            })

        # Find nearby [Hybrid] route (both directions)
        near_N = 15
        hyb = None
        # forward window
        for j in range(i, min(len(lines), i + near_N)):
            h = parse_hybrid_route(lines[j])
            if h:
                hyb = h
                break
        # backward if not found
        if not hyb:
            for j in range(i - 1, max(-1, i - near_N), -1):
                h = parse_hybrid_route(lines[j])
                if h:
                    hyb = h
                    break
        if hyb:
            fb.update({
                'hyb_route_col': hyb['hyb_route_col'],
                'hyb_mode': hyb.get('mode'),
                'hyb_perf': hyb.get('perf'),
                'hyb_margin': hyb.get('margin'),
                'hyb_u_raw': hyb.get('u_raw'),
                'prep_us': hyb.get('prep_us')
            })

        entries.append(fb)

    print(f"Parsed {len(entries)} RL feedback entries "
          f"(PdTarget samples: {len(pdtargets)}, LinTS resets: {resets})")
    return entries

# ---------- stats & viz ----------

def compute_statistics(entries: List[Dict], window_size: int = 50) -> Dict:
    stats = {
        'rewards': [],
        'ema': [],
        'col_ratio': [],
        'latencies_row': [],
        'latencies_col': [],
        'p95_row': [],
        'p95_col': [],
        'query_nums': [],
        # optional overlays
        'fused_tanh': [],
        'res_tanh': [],
        'cpuR': [],
        'memR': [],
        'mismatch_pd_vs_rl': 0,
        'mismatch_hyb_vs_rl': 0
    }

    window_row = deque(maxlen=window_size)
    window_col = deque(maxlen=window_size)
    col_window = deque(maxlen=window_size)

    current_ema = 0.0

    for i, e in enumerate(entries):
        stats['query_nums'].append(i)

        # rewards
        stats['rewards'].append(e.get('reward', np.nan))

        # ema (carry-forward)
        if 'ema' in e:
            current_ema = e['ema']
        stats['ema'].append(current_ema)

        # engine ratio (moving window) from RL decision
        col_window.append(1 if e.get('use_col', False) else 0)
        stats['col_ratio'].append(sum(col_window) / len(col_window))

        # latencies
        if 'latency' in e:
            if e.get('use_col', False):
                stats['latencies_col'].append(e['latency'])
                window_col.append(e['latency'])
            else:
                stats['latencies_row'].append(e['latency'])
                window_row.append(e['latency'])

        # p95 per engine (moving)
        if len(window_row) > 0:
            idx = max(0, int(0.95 * (len(window_row) - 1)))
            stats['p95_row'].append(sorted(window_row)[idx])
        else:
            stats['p95_row'].append(stats['p95_row'][-1] if stats['p95_row'] else 0.0)

        if len(window_col) > 0:
            idx = max(0, int(0.95 * (len(window_col) - 1)))
            stats['p95_col'].append(sorted(window_col)[idx])
        else:
            stats['p95_col'].append(stats['p95_col'][-1] if stats['p95_col'] else 0.0)

        # dispatcher overlays (if present)
        fused = e.get('fused', None)
        res = e.get('res_score', None)
        stats['fused_tanh'].append(np.tanh(fused / 2.0) if fused is not None else np.nan)
        stats['res_tanh'].append(np.tanh(res) if res is not None else np.nan)
        stats['cpuR'].append(e.get('cpuR', np.nan))
        stats['memR'].append(e.get('memR', np.nan))

        # mismatches
        if 'pd_decision' in e and 'use_col' in e:
            if bool(e['pd_decision']) != bool(e['use_col']):
                stats['mismatch_pd_vs_rl'] += 1
        if 'hyb_route_col' in e and 'use_col' in e:
            if bool(e['hyb_route_col']) != bool(e['use_col']):
                stats['mismatch_hyb_vs_rl'] += 1

    return stats

def create_visualization(stats: Dict, output_file: str = "rl_analysis.png"):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        pass

    x = stats['query_nums']

    # Panel 1: Reward + (optional) fused/resource overlays
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, stats['rewards'], '-', alpha=0.35, linewidth=0.9, label='Instant Δ')
    ax1.plot(x, stats['ema'], '-', linewidth=2.0, label='EMA Δ')
    if np.isfinite(np.nanmean(stats['fused_tanh'])):
        ax1.plot(x, stats['fused_tanh'], linestyle='--', linewidth=1.0, alpha=0.8, label='tanh(fused/2)')
    if np.isfinite(np.nanmean(stats['res_tanh'])):
        ax1.plot(x, stats['res_tanh'], linestyle=':', linewidth=1.0, alpha=0.8, label='tanh(rsc)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.set_xlabel('Query #'); ax1.set_ylabel('Signal (a.u.)')
    ax1.set_title('Reward / Fused / Resource Signals', fontsize=14, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Panel 2: Column Engine Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, stats['col_ratio'], 'g-', linewidth=2)
    ax2.fill_between(x, 0, stats['col_ratio'], alpha=0.25)
    ax2.set_xlabel('Query #'); ax2.set_ylabel('Ratio')
    ax2.set_title('Column Engine Ratio (window=50)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3)

    # Panel 3: Latency Distribution (Histogram)
    ax3 = fig.add_subplot(gs[1, 0])
    all_latencies = stats['latencies_row'] + stats['latencies_col']
    if all_latencies:
        max_lat = min(max(all_latencies), 7000)
        bins = np.logspace(0, np.log10(max_lat if max_lat > 1.01 else 1.01), 50)
        if stats['latencies_row']:
            ax3.hist(stats['latencies_row'], bins=bins, alpha=0.7,
                     label=f'Row (n={len(stats["latencies_row"])})',
                     edgecolor='black', linewidth=0.4)
        if stats['latencies_col']:
            ax3.hist(stats['latencies_col'], bins=bins, alpha=0.7,
                     label=f'Col (n={len(stats["latencies_col"])})',
                     edgecolor='black', linewidth=0.4)
        ax3.set_xscale('log'); ax3.set_yscale('log')
        ax3.set_xlabel('Latency (ms)'); ax3.set_ylabel('Count')
        ax3.set_title('Latency Distribution', fontsize=14, fontweight='bold')
        ax3.legend(); ax3.grid(True, alpha=0.3, which='both')

    # Panel 4: P95 Latency Estimates
    ax4 = fig.add_subplot(gs[1, 1])
    if stats['p95_row']:
        ax4.plot(x, stats['p95_row'], '-', linewidth=2, label='Row P95')
    if stats['p95_col']:
        ax4.plot(x, stats['p95_col'], '-', linewidth=2, label='Col P95')
    ax4.set_xlabel('Query #'); ax4.set_ylabel('P95 (ms)')
    ax4.set_title('P95 Latency (moving window)', fontsize=14, fontweight='bold')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    total_queries = len(x)
    final_col_ratio = stats['col_ratio'][-1] if stats['col_ratio'] else 0.0
    mism_pd = stats.get('mismatch_pd_vs_rl', 0)
    mism_hyb = stats.get('mismatch_hyb_vs_rl', 0)
    subtitle = f'RL Analysis - {total_queries} queries | Final Col Ratio: {final_col_ratio:.2f}'
    if mism_pd or mism_hyb:
        subtitle += f' | mismatches: pd={mism_pd}, hyb={mism_hyb}'
    fig.suptitle(subtitle, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

    save_metrics_summary(stats, output_file.replace('.png', '_summary.txt'))

def save_metrics_summary(stats: Dict, output_file: str):
    with open(output_file, 'w') as f:
        f.write("RL / Dispatcher Learning Summary\n")
        f.write("=" * 60 + "\n\n")
        total = len(stats['query_nums'])
        f.write(f"Total queries processed: {total}\n")
        if stats['col_ratio']:
            f.write(f"Final column engine ratio: {stats['col_ratio'][-1]:.3f}\n")
        if stats['ema']:
            f.write(f"Final EMA reward: {stats['ema'][-1]:.3f}\n")
        if stats['rewards']:
            f.write(f"Average instant reward: {np.nanmean(stats['rewards']):.3f}\n")

        # Dispatcher overlays
        fused = np.array(stats['fused_tanh'], dtype=float)
        res = np.array(stats['res_tanh'], dtype=float)
        if np.isfinite(np.nanmean(fused)):
            f.write(f"\nDispatcher fused (tanh/2) mean: {np.nanmean(fused):.3f}\n")
        if np.isfinite(np.nanmean(res)):
            f.write(f"Dispatcher rsc (tanh) mean: {np.nanmean(res):.3f}\n")
        f.write(f"Mismatches pd_vs_rl: {stats.get('mismatch_pd_vs_rl', 0)}\n")
        f.write(f"Mismatches hyb_vs_rl: {stats.get('mismatch_hyb_vs_rl', 0)}\n")

        # Latency stats
        if stats['latencies_row']:
            arr = np.array(stats['latencies_row'])
            f.write("\nRow engine statistics:\n")
            f.write(f"  Count: {len(arr)}\n")
            f.write(f"  Mean: {np.mean(arr):.2f} ms\n")
            f.write(f"  P50:  {np.percentile(arr, 50):.2f} ms\n")
            f.write(f"  P95:  {np.percentile(arr, 95):.2f} ms\n")

        if stats['latencies_col']:
            arr = np.array(stats['latencies_col'])
            f.write("\nColumn engine statistics:\n")
            f.write(f"  Count: {len(arr)}\n")
            f.write(f"  Mean: {np.mean(arr):.2f} ms\n")
            f.write(f"  P50:  {np.percentile(arr, 50):.2f} ms\n")
            f.write(f"  P95:  {np.percentile(arr, 95):.2f} ms\n")

    print(f"Summary saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize RL / dispatcher learning from MySQL logs")
    parser.add_argument("--log-path",
                        default="/home/wuy/mypolardb/db/log/master-error.log",
                        help="Path to MySQL error log")
    parser.add_argument("--output", default="rl_analysis.png",
                        help="Output PNG for visualization")
    parser.add_argument("--window", type=int, default=50,
                        help="Window size for moving averages and P95 calculation")
    parser.add_argument("--last-n", type=int, default=None,
                        help="Only analyze last N feedback entries (applied after parsing)")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"Log not found: {args.log_path}")
        sys.exit(1)

    print("Reading RL / dispatcher logs...")
    entries = read_rl_logs(args.log_path)
    if not entries:
        print("No RL feedback entries found!")
        print("\nTips:")
        print("1) Ensure dynamic mode is enabled (use_mm1_time = ON).")
        print("2) Run queries to generate [RL] feedback lines.")
        print("3) Check the log path & permissions.")
        sys.exit(1)

    if args.last_n:
        entries = entries[-args.last_n:]
        print(f"Using last {len(entries)} feedback entries")

    print("Computing statistics...")
    stats = compute_statistics(entries, window_size=args.window)

    print("Creating visualization...")
    create_visualization(stats, args.output)

    print("\nDone!")

if __name__ == "__main__":
    main()
