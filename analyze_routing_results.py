#!/usr/bin/env python3
"""
Analyze routing benchmark results and create comparison charts
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filepath='routing_comparison_results.json'):
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_comparison_table(data):
    """Create comparison table of all methods"""
    results = []
    
    for item in data:
        if 'error' not in item:
            results.append({
                'Method': item['routing_method'],
                'Queries': item['num_queries'],
                'Makespan (s)': round(item['makespan'], 3),
                'Mean Latency (s)': round(item['mean_latency'], 4),
                'Median Latency (s)': round(item['median_latency'], 4),
                'P95 Latency (s)': round(item['p95_latency'], 4),
                'Throughput (q/s)': round(item['throughput'], 1),
                'Success Rate (%)': round(item['successful'] / item['num_queries'] * 100, 1)
            })
    
    df = pd.DataFrame(results)
    return df

def create_visualizations(df):
    """Create comparison charts"""
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Color mapping for methods
    colors = {
        'default': '#1f77b4',
        'cost_threshold': '#ff7f0e', 
        'lightgbm': '#2ca02c',
        'gnn': '#d62728'
    }
    
    # 1. Makespan comparison
    ax = axes[0, 0]
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        ax.plot(method_data['Queries'], method_data['Makespan (s)'], 
                marker='o', label=method, color=colors.get(method, 'gray'), linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Makespan (seconds)')
    ax.set_title('Makespan Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mean Latency comparison
    ax = axes[0, 1]
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        ax.plot(method_data['Queries'], method_data['Mean Latency (s)'], 
                marker='s', label=method, color=colors.get(method, 'gray'), linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Mean Latency (seconds)')
    ax.set_title('Mean Latency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. P95 Latency comparison
    ax = axes[0, 2]
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        ax.plot(method_data['Queries'], method_data['P95 Latency (s)'], 
                marker='^', label=method, color=colors.get(method, 'gray'), linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('P95 Latency (seconds)')
    ax.set_title('95th Percentile Latency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Throughput comparison
    ax = axes[1, 0]
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        ax.plot(method_data['Queries'], method_data['Throughput (q/s)'], 
                marker='D', label=method, color=colors.get(method, 'gray'), linewidth=2)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Throughput (queries/second)')
    ax.set_title('Throughput Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Bar chart for 1000 queries comparison
    ax = axes[1, 1]
    df_1000 = df[df['Queries'] == 1000]
    x = np.arange(len(df_1000))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_1000['Makespan (s)'], width, label='Makespan (s)', 
                   color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Routing Method')
    ax.set_ylabel('Makespan (seconds)')
    ax.set_title('Makespan at 1000 Queries')
    ax.set_xticks(x)
    ax.set_xticklabels(df_1000['Method'])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Relative performance (normalized to default)
    ax = axes[1, 2]
    df_pivot = df.pivot(index='Queries', columns='Method', values='Makespan (s)')
    df_normalized = df_pivot.div(df_pivot['default'], axis=0)
    
    for method in df_normalized.columns:
        ax.plot(df_normalized.index, df_normalized[method], 
                marker='o', label=method, color=colors.get(method, 'gray'), linewidth=2)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Relative Makespan (vs Default)')
    ax.set_title('Relative Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('routing_comparison_charts.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_summary(df):
    """Print summary statistics"""
    
    print("\n" + "="*80)
    print("ROUTING METHOD COMPARISON SUMMARY")
    print("="*80)
    
    # Group by queries for comparison
    for num_queries in sorted(df['Queries'].unique()):
        print(f"\n--- {num_queries} Queries ---")
        df_q = df[df['Queries'] == num_queries].sort_values('Makespan (s)')
        
        print("\nMakespan Ranking:")
        for idx, row in df_q.iterrows():
            print(f"  {row['Method']:15s}: {row['Makespan (s)']:.3f}s "
                  f"(throughput: {row['Throughput (q/s)']:.1f} q/s)")
        
        # Calculate relative performance
        default_makespan = df_q[df_q['Method'] == 'default']['Makespan (s)'].values[0]
        print("\nRelative to Default:")
        for idx, row in df_q.iterrows():
            if row['Method'] != 'default':
                relative = (row['Makespan (s)'] / default_makespan - 1) * 100
                sign = "+" if relative > 0 else ""
                print(f"  {row['Method']:15s}: {sign}{relative:.1f}%")
    
    # Overall best performers
    print("\n" + "="*80)
    print("BEST PERFORMERS BY METRIC")
    print("="*80)
    
    metrics = ['Makespan (s)', 'Mean Latency (s)', 'P95 Latency (s)', 'Throughput (q/s)']
    
    for num_queries in sorted(df['Queries'].unique()):
        print(f"\n{num_queries} Queries:")
        df_q = df[df['Queries'] == num_queries]
        
        for metric in metrics:
            if metric == 'Throughput (q/s)':
                best = df_q.loc[df_q[metric].idxmax()]
            else:
                best = df_q.loc[df_q[metric].idxmin()]
            print(f"  Best {metric:20s}: {best['Method']:15s} ({best[metric]:.3f})")
    
    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    summary_stats = df.groupby('Method').agg({
        'Makespan (s)': ['mean', 'std'],
        'Mean Latency (s)': ['mean', 'std'],
        'P95 Latency (s)': ['mean', 'std'],
        'Throughput (q/s)': ['mean', 'std']
    }).round(3)
    
    print("\n", summary_stats)

def main():
    # Load results
    data = load_results('routing_comparison_results.json')
    
    # Create comparison table
    df = create_comparison_table(data)
    
    # Save to CSV for further analysis
    df.to_csv('routing_comparison_summary.csv', index=False)
    print("Comparison table saved to routing_comparison_summary.csv")
    
    # Print detailed comparison
    print("\nDETAILED COMPARISON TABLE:")
    print(df.to_string(index=False))
    
    # Print summary
    print_summary(df)
    
    # Create visualizations
    create_visualizations(df)
    print("\nCharts saved to routing_comparison_charts.png")

if __name__ == "__main__":
    main()