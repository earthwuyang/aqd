#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced benchmark script that works with bench_routing_modes_explain_first.py
"""

import subprocess
import re
import pandas as pd
import time
import os
from pathlib import Path
import numpy as np

# Benchmark configuration
MYSQLD_PID = 17748
TIMEOUT = 1  # minutes
ROW_CPU = 30
COL_CPU = 60
POISSON_QPS = 50

# Workload types
# WORKLOAD_TYPES = {
#     "TP heavy": "--TP_heavy",
#     "AP heavy": "--AP_heavy",
#     "normal load": ""
# }

WORKLOAD_TYPES = {
    "normal load": ""
}

# Query counts with warmup
# QUERY_COUNTS = [200, 400, 600, 800, 1000]
QUERY_COUNTS = [400,  1000]

WARMUP_QUERIES = 100  # Run this many queries before measurement

# Routing modes
ROUTING_MODES = {
    "cost_threshold": "cost threshold",
    "hybrid_optimizer": "hybrid opt",
    "lightgbm_static": "lgbm_static",
    "lightgbm_dynamic": "lgbm_adaptive"
}

# Result storage
results = {}

def parse_output(output):
    """Parse benchmark output with enhanced metrics extraction"""
    metrics = {}
    
    # Find final summary
    summary_start = output.find("################### 最终汇总 ###################")
    if summary_start == -1:
        print("WARNING: Final summary not found")
        return metrics
    
    summary_section = output[summary_start:]
    
    # Parse each mode's results with the actual format from your tool
    for mode_key, mode_display in ROUTING_MODES.items():
        # Pattern matches: mode_name  makespan±std  avg_latency±std  p95_latency±std
        pattern = rf"{mode_key}\s+(\d+\.\d+)\s*±\s*(\d+\.\d+)\s+(\d+\.\d+)\s*±\s*(\d+\.\d+)\s+(\d+\.\d+)\s*±\s*(\d+\.\d+)"
        match = re.search(pattern, summary_section)
        
        if match:
            metrics[mode_display] = {
                "makespan": float(match.group(1)),
                "makespan_std": float(match.group(2)),
                "avg_latency": float(match.group(3)),
                "avg_latency_std": float(match.group(4)),
                "p95_latency": float(match.group(5)),
                "p95_latency_std": float(match.group(6))
            }
        else:
            print(f"WARNING: No results found for mode {mode_key}")
    
    return metrics

def run_warmup(workload_type):
    """Run warmup queries for adaptive methods"""
    cmd = [
        "python", "bench_routing_modes_explain_first.py",
        "--mysqld_pid", str(MYSQLD_PID),
        "--enable_resource_control",
        "--rounds", "1",
        "-n", str(WARMUP_QUERIES),
        "--poisson_qps", str(POISSON_QPS),
        "--timeout", str(TIMEOUT * 60 * 1000),
        "--scenario", "mixed_mismatch",  # Use mixed scenario for warmup
    ]
    
    if WORKLOAD_TYPES[workload_type]:
        cmd.append(WORKLOAD_TYPES[workload_type])
    
    # print(f"\nRunning warmup: {WARMUP_QUERIES} queries for {workload_type}")
    # print("-" * 60)
    
    # try:
    #     result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    #     print("Warmup completed successfully")
        
    #     # Extract learning progress from warmup
    #     rl_lines = [line for line in result.stdout.split('\n') if '[RL]' in line]
    #     if rl_lines:
    #         print(f"Found {len(rl_lines)} RL feedback entries during warmup")
    #         # Show last few entries to verify learning
    #         for line in rl_lines[-5:]:
    #             print(f"  {line}")
        
    #     time.sleep(2)  # Brief pause after warmup
    # except subprocess.CalledProcessError as e:
    #     print(f"Warmup failed: {e}")
    #     print(f"STDERR: {e.stderr}")

def run_benchmark(workload_type, query_count, with_warmup=True):
    """Run single benchmark test with optional warmup"""
    
    # Run warmup only once at the beginning
    if with_warmup and query_count == QUERY_COUNTS[0]:
        run_warmup(workload_type)
    
    cmd = [
        "python", "bench_routing_modes_explain_first.py",
        "--mysqld_pid", str(MYSQLD_PID),
        "--enable_resource_control",
        "--rounds", "3",  # Multiple rounds for better statistics
        "-n", str(query_count),
        "--poisson_qps", str(POISSON_QPS),
        "--timeout", str(TIMEOUT * 60 * 1000),
        "--scenario", "default",  # Use default scenario for main test
    ]
    
    if WORKLOAD_TYPES[workload_type]:
        cmd.append(WORKLOAD_TYPES[workload_type])
    
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Test: {workload_type}, {query_count} queries")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract metrics
        metrics = parse_output(result.stdout)
        
        if not metrics:
            print("Failed to parse output. Last 2000 chars of stdout:")
            print(result.stdout[-2000:])
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])
        
        # Extract additional info from output
        rl_feedback_count = result.stdout.count('[RL] feedback')
        if rl_feedback_count > 0:
            print(f"  RL feedback count: {rl_feedback_count}")
            
            # Try to extract column selection ratio for adaptive method
            col_selections = result.stdout.count('col=Y')
            row_selections = result.stdout.count('col=N') 
            if col_selections + row_selections > 0:
                col_ratio = col_selections / (col_selections + row_selections)
                print(f"  Column selection ratio: {col_ratio:.2f}")
                
                # Add to metrics for lgbm_adaptive
                if "lgbm_adaptive" in metrics:
                    metrics["lgbm_adaptive"]["col_ratio"] = col_ratio
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {}

def analyze_results():
    """Analyze results and compute improvement metrics"""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    for workload in WORKLOAD_TYPES.keys():
        print(f"\n{workload}:")
        print("-"*40)
        
        for query_count in QUERY_COUNTS:
            print(f"\n#queries={query_count}:")
            
            # Get baseline (cost_threshold)
            baseline_key = (workload, query_count, "cost threshold")
            if baseline_key not in results:
                continue
                
            baseline = results[baseline_key]
            
            # Compare each method
            improvements = {}
            for mode in ["hybrid opt", "lgbm_static", "lgbm_adaptive"]:
                key = (workload, query_count, mode)
                if key in results:
                    mode_result = results[key]
                    
                    # Calculate improvements
                    makespan_imp = (baseline["makespan"] - mode_result["makespan"]) / baseline["makespan"] * 100
                    avg_lat_imp = (baseline["avg_latency"] - mode_result["avg_latency"]) / baseline["avg_latency"] * 100
                    p95_lat_imp = (baseline["p95_latency"] - mode_result["p95_latency"]) / baseline["p95_latency"] * 100
                    
                    improvements[mode] = {
                        "makespan": makespan_imp,
                        "avg_latency": avg_lat_imp,
                        "p95_latency": p95_lat_imp
                    }
                    
                    print(f"  {mode:15} - makespan: {makespan_imp:+6.1f}%, "
                          f"avg_lat: {avg_lat_imp:+6.1f}%, p95_lat: {p95_lat_imp:+6.1f}%")
                    
                    # Show column ratio for adaptive
                    if mode == "lgbm_adaptive" and "col_ratio" in mode_result:
                        print(f"    └─ column selection ratio: {mode_result['col_ratio']:.2f}")
            
            # Compare lgbm_static vs lgbm_adaptive
            static_key = (workload, query_count, "lgbm_static")
            adaptive_key = (workload, query_count, "lgbm_adaptive")
            if static_key in results and adaptive_key in results:
                static_res = results[static_key]
                adaptive_res = results[adaptive_key]
                
                adapt_vs_static = {
                    "makespan": (static_res["makespan"] - adaptive_res["makespan"]) / static_res["makespan"] * 100,
                    "avg_latency": (static_res["avg_latency"] - adaptive_res["avg_latency"]) / static_res["avg_latency"] * 100,
                    "p95_latency": (static_res["p95_latency"] - adaptive_res["p95_latency"]) / static_res["p95_latency"] * 100
                }
                
                print(f"\n  Adaptive vs Static improvement:")
                print(f"    makespan: {adapt_vs_static['makespan']:+.1f}%, "
                      f"avg_lat: {adapt_vs_static['avg_latency']:+.1f}%, "
                      f"p95_lat: {adapt_vs_static['p95_latency']:+.1f}%")

def monitor_rl_progress():
    """Parse and display RL learning progress from MySQL error log"""
    log_path = "/home/wuy/mypolardb/db/log/master-error.log"
    
    try:
        # Try different methods to read the log
        rl_lines = []
        
        # Method 1: Try direct read (might work if permissions allow)
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                rl_lines = [l for l in lines if '[RL] feedback' in l]
        except PermissionError:
            # Method 2: Try with sudo
            print("\nAttempting to read log with sudo...")
            try:
                result = subprocess.run(
                    ['sudo', 'grep', '[RL] feedback', log_path],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    rl_lines = result.stdout.strip().split('\n') if result.stdout else []
                else:
                    print(f"Sudo access failed: {result.stderr}")
            except Exception as e:
                print(f"Could not use sudo: {e}")
        
        # Method 3: Ask user to copy the log
        if not rl_lines:
            print("\n" + "="*60)
            print("MANUAL LOG ANALYSIS REQUIRED")
            print("="*60)
            print(f"Could not read {log_path} (permission denied)")
            print("\nPlease run the following commands manually:")
            print(f"  sudo grep '[RL] feedback' {log_path} | tail -100 > rl_feedback.log")
            print(f"  sudo chmod 644 rl_feedback.log")
            print("\nThen re-run this script to analyze the results.")
            
            # Check if user already created the file
            if Path("rl_feedback.log").exists():
                print("\nFound rl_feedback.log, analyzing...")
                with open("rl_feedback.log", 'r') as f:
                    rl_lines = f.readlines()
        
        # Analyze if we have data
        if rl_lines:
            print(f"\nTotal RL feedback entries: {len(rl_lines)}")
            
            # Parse statistics
            col_count = sum(1 for l in rl_lines if 'col=Y' in l)
            row_count = sum(1 for l in rl_lines if 'col=N' in l)
            
            if col_count + row_count > 0:
                col_ratio = col_count / (col_count + row_count)
                print(f"Column selection ratio: {col_ratio:.2%} ({col_count}/{col_count + row_count})")
            
            # Show first and last entries
            if len(rl_lines) > 0:
                print("\nFirst 3 entries:")
                for line in rl_lines[:3]:
                    print(f"  {line.strip()}")
                print("\nLast 3 entries:")
                for line in rl_lines[-3:]:
                    print(f"  {line.strip()}")
            
            # Try to run the visualization script if available
            monitor_script = Path("paste-2.txt")
            if monitor_script.exists() and Path("rl_feedback.log").exists():
                try:
                    monitor_cmd = ["python3", str(monitor_script), "rl_feedback.log", 
                                  "-o", "rl_learning_curve.png"]
                    subprocess.run(monitor_cmd, check=True)
                    print("\nRL learning curve saved to rl_learning_curve.png")
                except Exception as e:
                    print(f"Could not generate visualization: {e}")
                    
    except Exception as e:
        print(f"Error analyzing RL progress: {e}")

def create_results_table():
    """Create and save enhanced results table"""
    table_data = []
    
    for query_count in QUERY_COUNTS:
        for metric in ["makespan", "avg_latency", "p95_latency"]:
            row = {
                "queries": f"#queries={query_count}",
                "metric": metric.replace("_", " ")
            }
            
            for workload in WORKLOAD_TYPES.keys():
                for mode in ROUTING_MODES.values():
                    key = (workload, query_count, mode)
                    if key in results and metric in results[key]:
                        value = results[key][metric]
                        std_key = f"{metric}_std"
                        std = results[key].get(std_key, 0)
                        
                        # Format value
                        if metric == "makespan":
                            row[f"{workload}_{mode}"] = f"{value:.2f}"
                        else:
                            row[f"{workload}_{mode}"] = f"{value:.4f}"
                    else:
                        row[f"{workload}_{mode}"] = "N/A"
            
            table_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(table_data)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")
    
    # Print formatted table
    print("\n" + "="*120)
    print("Benchmark Results Table")
    print("="*120)
    print(f"Configuration: timeout={TIMEOUT}min, row_cpu={ROW_CPU}, col_cpu={COL_CPU}, poisson_qps={POISSON_QPS}")
    print("-"*120)
    
    # Print header
    header = "queries\t\tmetric\t\t"
    for workload in WORKLOAD_TYPES.keys():
        header += f"{workload:^60}\t"
    print(header)
    
    sub_header = "\t\t\t\t"
    for workload in WORKLOAD_TYPES.keys():
        for mode in ROUTING_MODES.values():
            sub_header += f"{mode:^14}"
        sub_header += "\t"
    print(sub_header)
    print("-"*120)
    
    # Print data
    for i, row in df.iterrows():
        line = f"{row['queries']}\t{row['metric']:12}"
        for workload in WORKLOAD_TYPES.keys():
            for mode in ROUTING_MODES.values():
                col_name = f"{workload}_{mode}"
                value = row[col_name]
                line += f"{value:^14}"
            line += "\t"
        print(line)
        
        # Add separator after each query count group
        if (i + 1) % 3 == 0:
            print("-"*120)

def main():
    """Main function with enhanced testing"""
    print("Starting enhanced benchmark tests...")
    print(f"Configuration:")
    print(f"  MySQL PID: {MYSQLD_PID}")
    print(f"  Timeout: {TIMEOUT} minutes")
    print(f"  CPU: row={ROW_CPU}, col={COL_CPU}")
    print(f"  QPS: {POISSON_QPS}")
    print(f"  Warmup queries: {WARMUP_QUERIES}")
    
    # Check if benchmark script exists
    if not Path("bench_routing_modes_explain_first.py").exists():
        print("Error: bench_routing_modes_explain_first.py not found")
        return
    
    total_tests = len(WORKLOAD_TYPES) * len(QUERY_COUNTS)
    current_test = 0
    
    # Run all test combinations
    for workload_type in WORKLOAD_TYPES.keys():
        for i, query_count in enumerate(QUERY_COUNTS):
            current_test += 1
            print(f"\n\nProgress: {current_test}/{total_tests}")
            print("="*60)
            
            # Run benchmark (with warmup for first iteration)
            metrics = run_benchmark(workload_type, query_count, 
                                  with_warmup=(i == 0))
            
            # Store results
            if metrics:
                for mode, mode_metrics in metrics.items():
                    key = (workload_type, query_count, mode)
                    results[key] = mode_metrics
                print(f"Success: {workload_type}, {query_count} queries")
            else:
                print(f"Failed: {workload_type}, {query_count} queries")
            
            # Brief pause between tests
            time.sleep(2)
    
    # Analyze and display results
    analyze_results()
    create_results_table()
    
    # Monitor RL learning progress
    monitor_rl_progress()
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    
    # Print Markdown format table
    print("\n\nMarkdown format table:")
    print("```")
    print("| | | AP heavy | | | |")
    print("|---|---|---|---|---|---|")
    print("| | | cost threshold | hybrid opt | lgbm_static | lgbm_adaptive |")
    
    for query_count in QUERY_COUNTS:
        for metric in ["makespan", "avg_latency", "p95_latency"]:
            line = f"| #queries={query_count} | {metric.replace('_', ' ')} |"
            for workload in WORKLOAD_TYPES.keys():
                for mode in ROUTING_MODES.values():
                    key = (workload, query_count, mode)
                    if key in results and metric in results[key]:
                        value = results[key][metric]
                        if metric == "makespan":
                            line += f" {value:.2f} |"
                        else:
                            line += f" {value:.4f} |"
                    else:
                        line += " N/A |"
            print(line)
    print("```")

if __name__ == "__main__":
    main()