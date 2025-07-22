#!/usr/bin/env python3
"""
监控 lightgbm_dynamic 的学习过程
"""

import re
import time
import subprocess
import matplotlib.pyplot as plt
from collections import deque

LOG_FILE="/home/wuy/simple_row_column_routing/master-error.log"
def monitor_mysql_log(log_file=LOG_FILE):
    """
    实时监控 MySQL 日志中的 RL 相关信息
    """
    
    # 数据存储
    rewards = deque(maxlen=1000)
    reward_emas = deque(maxlen=1000)
    choices = deque(maxlen=1000)
    latencies = {"row": deque(maxlen=1000), "col": deque(maxlen=1000)}
    p95_estimates = {"row": deque(maxlen=1000), "col": deque(maxlen=1000)}
    
    # 正则表达式
    feedback_pattern = re.compile(
        r'\[RL\] feedback #(\d+): col=([YN]) lat=([\d.]+)ms '
        r'Δ=([-\d.]+) EMA=([-\d.]+) STD=([-\d.]+) '
        r'row_p95=([\d.]+) col_p95=([\d.]+)'
    )
    
    linucb_pattern = re.compile(
        r'\[LinUCB\] update=(\d+) mu=([-\d.]+) sigma=([-\d.]+) '
        r'alpha=([-\d.]+) ucb=([-\d.]+)'
    )
    
    # 实时绘图
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    with subprocess.Popen(['tail', '-f', log_file], 
                         stdout=subprocess.PIPE, 
                         universal_newlines=True) as proc:
        
        for line in proc.stdout:
            # 解析反馈信息
            match = feedback_pattern.search(line)
            if match:
                count = int(match.group(1))
                is_col = match.group(2) == 'Y'
                lat = float(match.group(3))
                reward = float(match.group(4))
                ema = float(match.group(5))
                std = float(match.group(6))
                row_p95 = float(match.group(7))
                col_p95 = float(match.group(8))
                
                # 存储数据
                rewards.append(reward)
                reward_emas.append(ema)
                choices.append(1 if is_col else 0)
                
                if is_col:
                    latencies["col"].append(lat)
                else:
                    latencies["row"].append(lat)
                
                p95_estimates["row"].append(row_p95)
                p95_estimates["col"].append(col_p95)
                
                # 每 10 次更新图表
                if count % 10 == 0:
                    update_plots(axes, rewards, reward_emas, choices, 
                               latencies, p95_estimates)
    
def update_plots(axes, rewards, reward_emas, choices, latencies, p95_estimates):
    """
    更新监控图表
    """
    # 清除旧图
    for ax in axes.flat:
        ax.clear()
    
    # 1. 奖励趋势
    ax = axes[0, 0]
    ax.plot(list(rewards), 'b-', alpha=0.3, label='Instant Reward')
    ax.plot(list(reward_emas), 'r-', linewidth=2, label='EMA Reward')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Reward Signal Over Time')
    ax.set_xlabel('Query #')
    ax.set_ylabel('Reward (Δ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 引擎选择比例
    ax = axes[0, 1]
    if len(choices) > 0:
        window = 50
        col_ratio = []
        for i in range(len(choices)):
            start = max(0, i - window)
            window_choices = list(choices)[start:i+1]
            if window_choices:
                col_ratio.append(sum(window_choices) / len(window_choices))
        
        ax.plot(col_ratio, 'g-', linewidth=2)
        ax.fill_between(range(len(col_ratio)), 0, col_ratio, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_title(f'Column Engine Selection Ratio (window={window})')
        ax.set_xlabel('Query #')
        ax.set_ylabel('Column Ratio')
        ax.grid(True, alpha=0.3)
    
    # 3. 延迟对比
    ax = axes[1, 0]
    if latencies["row"]:
        ax.hist(list(latencies["row"]), bins=30, alpha=0.5, 
                label=f'Row (n={len(latencies["row"])})', color='blue')
    if latencies["col"]:
        ax.hist(list(latencies["col"]), bins=30, alpha=0.5, 
                label=f'Col (n={len(latencies["col"])})', color='red')
    ax.set_title('Latency Distribution')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_yscale('log')
    
    # 4. P95 估计值趋势
    ax = axes[1, 1]
    ax.plot(list(p95_estimates["row"]), 'b-', label='Row P95', linewidth=2)
    ax.plot(list(p95_estimates["col"]), 'r-', label='Col P95', linewidth=2)
    ax.set_title('P95 Latency Estimates')
    ax.set_xlabel('Query #')
    ax.set_ylabel('P95 Latency (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.01)

def analyze_learning_curve(log_file):
    """
    分析学习曲线，找出 Dynamic 开始超越 Static 的时间点
    """
    # 解析日志获取累积奖励
    cumulative_rewards = []
    current_sum = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            if '[RL] feedback' in line:
                match = re.search(r'Δ=([-\d.]+)', line)
                if match:
                    reward = float(match.group(1))
                    current_sum += reward
                    cumulative_rewards.append(current_sum)
    
    # 找出开始稳定正收益的点
    window = 50
    for i in range(window, len(cumulative_rewards)):
        recent_avg = (cumulative_rewards[i] - cumulative_rewards[i-window]) / window
        if recent_avg > 0.01:  # 阈值
            print(f"Dynamic 模型在第 {i} 个查询后开始稳定超越 Static")
            break
    
    return cumulative_rewards

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # 离线分析模式
        curve = analyze_learning_curve(LOG_FILE)
        plt.figure(figsize=(10, 6))
        plt.plot(curve)
        plt.title("Cumulative Reward Over Time")
        plt.xlabel("Query #")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.show()
    else:
        # 实时监控模式
        print("Starting real-time monitoring...")
        monitor_mysql_log()