#!/usr/bin/env python3
"""
Analyze and combine all benchmark results (100-3000 queries)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_all_results():
    """Load all benchmark results"""
    all_data = []
    
    # Load low-scale results
    with open('routing_comparison_results.json', 'r') as f:
        data1 = json.load(f)
        all_data.extend(data1)
    
    # Load high-scale results
    with open('high_scale_results.json', 'r') as f:
        data2 = json.load(f)
        all_data.extend(data2)
    
    return all_data

def create_comprehensive_summary(data):
    """Create comprehensive analysis of all results"""
    
    results = []
    for item in data:
        if 'error' not in item:
            results.append({
                'Method': item['routing_method'],
                'Queries': item['num_queries'],
                'Makespan (s)': item['makespan'],
                'Mean Latency (ms)': item['mean_latency'] * 1000,
                'P95 Latency (ms)': item['p95_latency'] * 1000,
                'Throughput (q/s)': item['throughput'],
                'Success Rate (%)': (item['successful'] / item['num_queries']) * 100
            })
    
    df = pd.DataFrame(results)
    
    # Create pivot tables for analysis
    makespan_pivot = df.pivot(index='Queries', columns='Method', values='Makespan (s)')
    throughput_pivot = df.pivot(index='Queries', columns='Method', values='Throughput (q/s)')
    latency_pivot = df.pivot(index='Queries', columns='Method', values='Mean Latency (ms)')
    p95_pivot = df.pivot(index='Queries', columns='Method', values='P95 Latency (ms)')
    
    return df, makespan_pivot, throughput_pivot, latency_pivot, p95_pivot

def print_comprehensive_summary(df, makespan_pivot, throughput_pivot, latency_pivot, p95_pivot):
    """Print comprehensive analysis"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE ROUTING METHOD PERFORMANCE ANALYSIS (100-3000 QUERIES)")
    print("="*100)
    
    print("\n### MAKESPAN COMPARISON (seconds) ###")
    print(makespan_pivot.round(3).to_string())
    
    print("\n### THROUGHPUT COMPARISON (queries/second) ###")
    print(throughput_pivot.round(1).to_string())
    
    print("\n### MEAN LATENCY COMPARISON (milliseconds) ###")
    print(latency_pivot.round(1).to_string())
    
    print("\n### P95 LATENCY COMPARISON (milliseconds) ###")
    print(p95_pivot.round(1).to_string())
    
    # Calculate relative performance to default
    print("\n" + "="*100)
    print("RELATIVE PERFORMANCE VS DEFAULT METHOD")
    print("="*100)
    
    makespan_relative = (makespan_pivot.div(makespan_pivot['default'], axis=0) - 1) * 100
    
    print("\n### Makespan Difference from Default (%) ###")
    print("(Negative = Better, Positive = Worse)")
    for method in ['cost_threshold', 'lightgbm', 'gnn']:
        print(f"\n{method}:")
        for queries in sorted(makespan_relative.index):
            diff = makespan_relative.loc[queries, method]
            sign = "+" if diff > 0 else ""
            print(f"  {queries:4d} queries: {sign}{diff:6.2f}%")
    
    # Best performer at each scale
    print("\n" + "="*100)
    print("BEST PERFORMER BY QUERY SCALE")
    print("="*100)
    
    for queries in sorted(makespan_pivot.index):
        best_makespan = makespan_pivot.loc[queries].idxmin()
        best_throughput = throughput_pivot.loc[queries].idxmax()
        best_latency = latency_pivot.loc[queries].idxmin()
        
        print(f"\n{queries} Queries:")
        print(f"  Best Makespan:    {best_makespan:15s} ({makespan_pivot.loc[queries, best_makespan]:.3f}s)")
        print(f"  Best Throughput:  {best_throughput:15s} ({throughput_pivot.loc[queries, best_throughput]:.1f} q/s)")
        print(f"  Best Latency:     {best_latency:15s} ({latency_pivot.loc[queries, best_latency]:.1f}ms)")
    
    # Scalability analysis
    print("\n" + "="*100)
    print("SCALABILITY ANALYSIS")
    print("="*100)
    
    for method in makespan_pivot.columns:
        print(f"\n{method}:")
        prev_queries = 100
        for queries in sorted(makespan_pivot.index)[1:]:
            scale_factor = queries / prev_queries
            makespan_factor = makespan_pivot.loc[queries, method] / makespan_pivot.loc[prev_queries, method]
            efficiency = scale_factor / makespan_factor
            print(f"  {prev_queries:4d} → {queries:4d}: {scale_factor:.1f}x queries, "
                  f"{makespan_factor:.2f}x time, efficiency: {efficiency:.1%}")
            prev_queries = queries

def create_scalability_charts(df):
    """Create scalability visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    df_pivot = df.pivot(index='Queries', columns='Method', values=['Makespan (s)', 
                                                                     'Throughput (q/s)',
                                                                     'Mean Latency (ms)',
                                                                     'P95 Latency (ms)'])
    
    colors = {
        'default': '#1f77b4',
        'cost_threshold': '#ff7f0e',
        'lightgbm': '#2ca02c', 
        'gnn': '#d62728'
    }
    
    # 1. Makespan scaling
    ax = axes[0, 0]
    for method in ['default', 'cost_threshold', 'lightgbm', 'gnn']:
        ax.plot(df_pivot.index, df_pivot['Makespan (s)'][method], 
                marker='o', label=method, color=colors[method], linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Makespan (seconds)')
    ax.set_title('Makespan Scaling (100-3000 queries)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Throughput scaling
    ax = axes[0, 1]
    for method in ['default', 'cost_threshold', 'lightgbm', 'gnn']:
        ax.plot(df_pivot.index, df_pivot['Throughput (q/s)'][method],
                marker='s', label=method, color=colors[method], linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Throughput (queries/second)')
    ax.set_title('Throughput Scaling (100-3000 queries)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mean Latency scaling
    ax = axes[1, 0]
    for method in ['default', 'cost_threshold', 'lightgbm', 'gnn']:
        ax.plot(df_pivot.index, df_pivot['Mean Latency (ms)'][method],
                marker='^', label=method, color=colors[method], linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Mean Latency Scaling (100-3000 queries)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Bar comparison at highest scale (3000 queries)
    ax = axes[1, 1]
    df_3000 = df[df['Queries'] == 3000].set_index('Method')
    
    x = np.arange(len(df_3000.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_3000['Makespan (s)'], width, 
                   label='Makespan (s)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_3000['Throughput (q/s)']/10, width,
                   label='Throughput/10 (q/s)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Routing Method')
    ax.set_ylabel('Performance Metrics')
    ax.set_title('Performance at 3000 Queries')
    ax.set_xticks(x)
    ax.set_xticklabels(df_3000.index)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*10:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=150, bbox_inches='tight')
    print("\nScalability charts saved to scalability_analysis.png")

def main():
    # Load all results
    data = load_all_results()
    
    # Create comprehensive analysis
    df, makespan_pivot, throughput_pivot, latency_pivot, p95_pivot = create_comprehensive_summary(data)
    
    # Print analysis
    print_comprehensive_summary(df, makespan_pivot, throughput_pivot, latency_pivot, p95_pivot)
    
    # Create visualizations
    create_scalability_charts(df)
    
    # Save combined results
    df.to_csv('all_benchmark_results.csv', index=False)
    print("\nAll results saved to all_benchmark_results.csv")

if __name__ == "__main__":
    main()