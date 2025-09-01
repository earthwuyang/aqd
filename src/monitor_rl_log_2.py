#!/usr/bin/env python3
"""
analyze_rl_logs.py - Analyze and visualize RL learning progress from MySQL logs
"""

import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
import sys

def parse_rl_log_entry(line: str) -> Dict:
    """Parse a single RL log entry"""
    entry = {}
    
    # Extract timestamp if present
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', line)
    if timestamp_match:
        entry['timestamp'] = timestamp_match.group(1)
    
    # New format: "[RL] feedback  col=Y  lat=123.4ms  Δ=0.567"
    # Extract column choice
    col_match = re.search(r'col=([YN])', line)
    if col_match:
        entry['use_col'] = col_match.group(1) == 'Y'
    
    # Extract latency (now in format "lat=123.4ms")
    lat_match = re.search(r'lat=([\d.]+)ms', line)
    if lat_match:
        entry['latency'] = float(lat_match.group(1))
    
    # Extract reward (delta) - Unicode character Δ
    delta_match = re.search(r'Δ=([-\d.]+)', line)
    if delta_match:
        entry['reward'] = float(delta_match.group(1))
    
    # Also check for the EMA logs: "[RL] Δ=0.123  Δ_EMA=0.456"
    if 'Δ_EMA=' in line:
        ema_match = re.search(r'Δ_EMA=([-\d.]+)', line)
        if ema_match:
            entry['ema'] = float(ema_match.group(1))
    
    # Extract query hash if present (might not be in new format)
    hash_match = re.search(r'qhash=(\w+)', line)
    if hash_match:
        entry['qhash'] = hash_match.group(1)
    
    # Extract baseline (if available - might not be in new format)
    base_match = re.search(r'base=([\d.]+)ms', line)
    if base_match:
        entry['baseline'] = float(base_match.group(1))
    
    # Extract exploration bonus (if present)
    explore_match = re.search(r'explore_bonus=([\d.]+)', line)
    if explore_match:
        entry['explore_bonus'] = float(explore_match.group(1))
    
    return entry

def read_rl_logs(log_path: str = "/home/wuy/mypolardb/db/log/master-error.log") -> List[Dict]:
    """Read and parse RL log entries"""
    entries = []
    ema_tracker = {}  # Track EMA values by approximate time
    
    try:
        # Try to read log with sudo if needed
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
        except PermissionError:
            print("Using sudo to read log file...")
            # Get both feedback and EMA lines
            result = subprocess.run(
                ['sudo', 'grep', '\[RL\]', log_path],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
            else:
                print(f"Error reading log file: {result.stderr}")
                return entries
        
        # First pass: collect EMA values
        for line in lines:
            if '[RL] Δ=' in line and 'Δ_EMA=' in line:
                # This is an EMA monitoring line
                ema_match = re.search(r'Δ_EMA=([-\d.]+)', line)
                if ema_match:
                    # Store with line number as key for now
                    ema_tracker[len(ema_tracker)] = float(ema_match.group(1))
        
        # Second pass: parse feedback entries
        ema_idx = 0
        for line in lines:
            if '[RL] feedback' in line:
                entry = parse_rl_log_entry(line)
                if entry:
                    # Try to associate with nearest EMA value
                    if ema_idx < len(ema_tracker):
                        entry['ema'] = list(ema_tracker.values())[ema_idx]
                        ema_idx += 1
                    entries.append(entry)
        
        print(f"Parsed {len(entries)} RL feedback entries")
        if ema_tracker:
            print(f"Found {len(ema_tracker)} EMA monitoring entries")
        
    except Exception as e:
        print(f"Error reading logs: {e}")
    
    return entries

def compute_statistics(entries: List[Dict], window_size: int = 50) -> Dict:
    """Compute statistics for visualization"""
    stats = {
        'rewards': [],
        'ema': [],
        'col_ratio': [],
        'latencies_row': [],
        'latencies_col': [],
        'p95_row': [],
        'p95_col': [],
        'query_nums': []
    }
    
    # Moving windows for P95 calculation
    window_row = deque(maxlen=window_size)
    window_col = deque(maxlen=window_size)
    
    # Running count for column ratio
    col_window = deque(maxlen=window_size)
    
    # Track EMA from the separate log lines
    current_ema = 0.0
    
    for i, entry in enumerate(entries):
        stats['query_nums'].append(i)
        
        # Rewards
        if 'reward' in entry:
            stats['rewards'].append(entry['reward'])
        else:
            stats['rewards'].append(0)
        
        # EMA - use the value from entry if present, otherwise keep last known
        if 'ema' in entry:
            current_ema = entry['ema']
            stats['ema'].append(current_ema)
        else:
            stats['ema'].append(current_ema)
        
        # Column ratio (moving average)
        col_window.append(1 if entry.get('use_col', False) else 0)
        stats['col_ratio'].append(sum(col_window) / len(col_window))
        
        # Latencies
        if 'latency' in entry:
            if entry.get('use_col', False):
                stats['latencies_col'].append(entry['latency'])
                window_col.append(entry['latency'])
            else:
                stats['latencies_row'].append(entry['latency'])
                window_row.append(entry['latency'])
        
        # P95 estimates
        if len(window_row) > 0:
            p95_idx = int(0.95 * len(window_row))
            stats['p95_row'].append(sorted(window_row)[p95_idx])
        else:
            stats['p95_row'].append(stats['p95_row'][-1] if stats['p95_row'] else 0)
        
        if len(window_col) > 0:
            p95_idx = int(0.95 * len(window_col))
            stats['p95_col'].append(sorted(window_col)[p95_idx])
        else:
            stats['p95_col'].append(stats['p95_col'][-1] if stats['p95_col'] else 0)
    
    return stats

def create_visualization(stats: Dict, output_file: str = "rl_analysis.png"):
    """Create 4-panel visualization similar to the provided example"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Common styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Panel 1: Reward Signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(stats['query_nums'], stats['rewards'], 'b-', alpha=0.3, linewidth=0.8, label='Instant Δ')
    ax1.plot(stats['query_nums'], stats['ema'], 'r-', linewidth=2, label='EMA Δ')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Query #')
    ax1.set_ylabel('Δ')
    ax1.set_title('Reward Signal', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Column Engine Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(stats['query_nums'], stats['col_ratio'], 'g-', linewidth=2)
    ax2.fill_between(stats['query_nums'], 0, stats['col_ratio'], alpha=0.3, color='lightblue')
    ax2.set_xlabel('Query #')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Column Engine Ratio (window=50)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Latency Distribution (Histogram)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Create bins for histogram
    all_latencies = stats['latencies_row'] + stats['latencies_col']
    if all_latencies:
        max_lat = min(max(all_latencies), 7000)  # Cap at 7000ms for visibility
        bins = np.logspace(0, np.log10(max_lat), 50)
        
        # Plot histograms
        if stats['latencies_row']:
            ax3.hist(stats['latencies_row'], bins=bins, alpha=0.7, label=f'Row (n={len(stats["latencies_row"])})', 
                    color='steelblue', edgecolor='black', linewidth=0.5)
        if stats['latencies_col']:
            ax3.hist(stats['latencies_col'], bins=bins, alpha=0.7, label=f'Col (n={len(stats["latencies_col"])})', 
                    color='orange', edgecolor='black', linewidth=0.5)
        
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Count')
        ax3.set_title('Latency Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
    
    # Panel 4: P95 Latency Estimates
    ax4 = fig.add_subplot(gs[1, 1])
    if stats['p95_row']:
        ax4.plot(stats['query_nums'], stats['p95_row'], 'b-', linewidth=2, label='Row P95')
    if stats['p95_col']:
        ax4.plot(stats['query_nums'], stats['p95_col'], 'r-', linewidth=2, label='Col P95')
    ax4.set_xlabel('Query #')
    ax4.set_ylabel('P95 (ms)')
    ax4.set_title('P95 Latency Estimates', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Set overall title
    total_queries = len(stats['query_nums'])
    col_count = sum(1 for r in stats['col_ratio'] if r > 0.5)
    final_col_ratio = stats['col_ratio'][-1] if stats['col_ratio'] else 0
    
    fig.suptitle(f'RL Learning Analysis - {total_queries} Queries, Final Col Ratio: {final_col_ratio:.2f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Also save individual metrics
    save_metrics_summary(stats, output_file.replace('.png', '_summary.txt'))

def save_metrics_summary(stats: Dict, output_file: str):
    """Save summary statistics to text file"""
    with open(output_file, 'w') as f:
        f.write("RL Learning Summary\n")
        f.write("=" * 50 + "\n\n")
        
        total_queries = len(stats['query_nums'])
        f.write(f"Total queries processed: {total_queries}\n")
        
        if stats['col_ratio']:
            final_ratio = stats['col_ratio'][-1]
            f.write(f"Final column engine ratio: {final_ratio:.3f}\n")
        
        if stats['ema']:
            final_ema = stats['ema'][-1]
            f.write(f"Final EMA reward: {final_ema:.3f}\n")
        
        if stats['rewards']:
            avg_reward = np.mean(stats['rewards'])
            f.write(f"Average instant reward: {avg_reward:.3f}\n")
        
        # Latency statistics
        if stats['latencies_row']:
            f.write(f"\nRow engine statistics:\n")
            f.write(f"  Count: {len(stats['latencies_row'])}\n")
            f.write(f"  Mean: {np.mean(stats['latencies_row']):.2f} ms\n")
            f.write(f"  P50: {np.percentile(stats['latencies_row'], 50):.2f} ms\n")
            f.write(f"  P95: {np.percentile(stats['latencies_row'], 95):.2f} ms\n")
        
        if stats['latencies_col']:
            f.write(f"\nColumn engine statistics:\n")
            f.write(f"  Count: {len(stats['latencies_col'])}\n")
            f.write(f"  Mean: {np.mean(stats['latencies_col']):.2f} ms\n")
            f.write(f"  P50: {np.percentile(stats['latencies_col'], 50):.2f} ms\n")
            f.write(f"  P95: {np.percentile(stats['latencies_col'], 95):.2f} ms\n")
        
        print(f"Summary saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize RL learning from MySQL logs")
    parser.add_argument("--log-path", default="/home/wuy/mypolardb/db/log/master-error.log",
                        help="Path to MySQL error log")
    parser.add_argument("--output", default="rl_analysis.png",
                        help="Output file for visualization")
    parser.add_argument("--window", type=int, default=50,
                        help="Window size for moving averages and P95 calculation")
    parser.add_argument("--last-n", type=int, default=None,
                        help="Only analyze last N entries")
    
    args = parser.parse_args()
    
    # Read and parse logs
    print("Reading RL logs...")
    entries = read_rl_logs(args.log_path)
    
    if not entries:
        print("No RL log entries found!")
        print("\nTips:")
        print("1. Make sure lightgbm_dynamic mode is enabled (use_mm1_time = ON)")
        print("2. Run some queries to generate feedback")
        print("3. Check the log path is correct")
        sys.exit(1)
    
    # Limit to last N entries if requested
    if args.last_n:
        entries = entries[-args.last_n:]
        print(f"Using last {len(entries)} entries")
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(entries, window_size=args.window)
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(stats, args.output)
    
    print("\nDone!")

if __name__ == "__main__":
    main()