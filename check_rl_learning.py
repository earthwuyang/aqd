#!/usr/bin/env python3
"""
Quick script to check if RL learning is happening in the MySQL error log
"""

import re
import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

def check_rl_learning(log_path="/home/wuy/mypolardb/db/log/master-error.log"):
    """Check the RL learning progress from MySQL error log"""
    
    # Handle different log paths
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    # Check if we can read the file directly
    log_file = Path(log_path)
    lines = []
    
    if log_file.exists() and log_file.is_file():
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
        except PermissionError:
            print(f"Permission denied reading {log_path}")
            print("\nTrying with sudo...")
            try:
                result = subprocess.run(
                    ['sudo', 'cat', log_path],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                else:
                    print(f"Sudo failed: {result.stderr}")
            except:
                pass
    
    # If still no lines, try local copy
    if not lines and Path("rl_feedback.log").exists():
        print("Using local rl_feedback.log file...")
        with open("rl_feedback.log", 'r') as f:
            lines = f.readlines()
    
    if not lines:
        print(f"\nCould not read log file. Please run:")
        print(f"  sudo grep '[RL] feedback' {log_path} > rl_feedback.log")
        print(f"  chmod 644 rl_feedback.log")
        print("Then re-run this script.")
        return
    
    # Pattern to match RL feedback lines
    feedback_pattern = re.compile(
        r'\[RL\] feedback #(\d+): col=([YN]) lat=([\d.]+)ms '
        r'Δ=([-\d.]+) EMA=([-\d.]+)±([-\d.]+) STD=([-\d.]+)'
    )
    
    # Storage for metrics
    feedback_nums = []
    choices = []  # 1 for col, 0 for row
    latencies = []
    deltas = []
    emas = []
    stds = []
    
    print(f"Parsing {log_path}...")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = feedback_pattern.search(line)
            if match:
                feedback_num = int(match.group(1))
                is_col = match.group(2) == 'Y'
                lat = float(match.group(3))
                delta = float(match.group(4))
                ema = float(match.group(5))
                std = float(match.group(7))
                
                feedback_nums.append(feedback_num)
                choices.append(1 if is_col else 0)
                latencies.append(lat)
                deltas.append(delta)
                emas.append(ema)
                stds.append(std)
    
    if not feedback_nums:
        print("No RL feedback entries found in the log!")
        return
    
    print(f"Found {len(feedback_nums)} RL feedback entries")
    
    # Calculate statistics
    col_ratio = sum(choices) / len(choices)
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"\nSummary Statistics:")
    print(f"  Total feedbacks: {len(feedback_nums)}")
    print(f"  Column selection ratio: {col_ratio:.2%}")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Final EMA reward: {emas[-1]:.4f}")
    print(f"  Final STD: {stds[-1]:.4f}")
    
    # Check if learning is happening
    window = 50
    early_col_ratio = sum(choices[:window]) / window if len(choices) >= window else col_ratio
    late_col_ratio = sum(choices[-window:]) / len(choices[-window:]) if len(choices) >= window else col_ratio
    
    print(f"\nLearning Progress:")
    print(f"  Early column ratio (first {window}): {early_col_ratio:.2%}")
    print(f"  Late column ratio (last {window}): {late_col_ratio:.2%}")
    print(f"  Change: {(late_col_ratio - early_col_ratio):.2%}")
    
    # Check reward convergence
    if len(emas) > 100:
        early_ema = sum(emas[:50]) / 50
        late_ema = sum(emas[-50:]) / 50
        print(f"\nReward Convergence:")
        print(f"  Early EMA average: {early_ema:.4f}")
        print(f"  Late EMA average: {late_ema:.4f}")
        print(f"  Improvement: {late_ema - early_ema:.4f}")
    
    # Plot the learning curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Reward over time
    ax = axes[0, 0]
    ax.plot(feedback_nums, deltas, 'b-', alpha=0.3, label='Instant Δ')
    ax.plot(feedback_nums, emas, 'r-', lw=2, label='EMA Δ')
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.set_title('Reward Signal Over Time')
    ax.set_xlabel('Feedback #')
    ax.set_ylabel('Reward Δ')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Column selection ratio (moving average)
    ax = axes[0, 1]
    window_size = 50
    col_ratio_ma = []
    for i in range(len(choices)):
        start = max(0, i - window_size + 1)
        window = choices[start:i+1]
        col_ratio_ma.append(sum(window) / len(window))
    
    ax.plot(feedback_nums, col_ratio_ma, 'g-', lw=2)
    ax.fill_between(feedback_nums, 0, col_ratio_ma, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_title(f'Column Engine Selection Ratio (MA={window_size})')
    ax.set_xlabel('Feedback #')
    ax.set_ylabel('Column Ratio')
    ax.grid(alpha=0.3)
    
    # 3. Latency distribution
    ax = axes[1, 0]
    row_lats = [lat for lat, choice in zip(latencies, choices) if choice == 0]
    col_lats = [lat for lat, choice in zip(latencies, choices) if choice == 1]
    
    if row_lats:
        ax.hist(row_lats, bins=30, alpha=0.5, label=f'Row (n={len(row_lats)})')
    if col_lats:
        ax.hist(col_lats, bins=30, alpha=0.5, label=f'Col (n={len(col_lats)})')
    
    ax.set_title('Latency Distribution by Engine')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.legend()
    
    # 4. EMA confidence interval
    ax = axes[1, 1]
    ax.plot(feedback_nums, emas, 'b-', lw=2, label='EMA')
    upper = [e + s for e, s in zip(emas, stds)]
    lower = [e - s for e, s in zip(emas, stds)]
    ax.fill_between(feedback_nums, lower, upper, alpha=0.3, label='±1 STD')
    ax.set_title('EMA Reward with Confidence')
    ax.set_xlabel('Feedback #')
    ax.set_ylabel('EMA Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_learning_analysis.png', dpi=150)
    print(f"\nSaved learning curves to rl_learning_analysis.png")
    plt.close()
    
    # Check for signs of good/bad learning
    print("\n" + "="*60)
    print("LEARNING QUALITY CHECK:")
    print("="*60)
    
    # 1. Is the model exploring enough?
    unique_patterns = len(set(zip(choices[i:i+10] for i in range(0, len(choices)-10, 5))))
    print(f"✓ Exploration diversity: {unique_patterns} unique patterns")
    
    # 2. Is the reward improving?
    if len(emas) > 100:
        trend = (emas[-1] - emas[50]) / abs(emas[50]) if emas[50] != 0 else 0
        status = "✓" if trend > 0 else "✗"
        print(f"{status} Reward trend: {trend:+.2%}")
    
    # 3. Is the variance decreasing?
    if len(stds) > 100:
        var_trend = (stds[-1] - stds[50]) / stds[50] if stds[50] != 0 else 0
        status = "✓" if var_trend < 0 else "✗"
        print(f"{status} Variance trend: {var_trend:+.2%}")
    
    # 4. Are we seeing adaptation?
    adaptation = abs(late_col_ratio - 0.5) > 0.1  # Moved away from 50/50
    status = "✓" if adaptation else "✗"
    print(f"{status} Adaptation from baseline: {abs(late_col_ratio - 0.5):.2%}")

if __name__ == "__main__":
    # Default log path
    log_path = "/home/wuy/mypolardb/db/log/master-error.log"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    check_rl_learning(log_path)