import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator

# ======================
# MODERN STYLE CONFIG
# ======================
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.family': 'Arial',
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'legend.fontsize': 10,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'figure.dpi': 300,
    'figure.facecolor': 'white'
})

# ======================
# DATA LOADING
# ======================
df = pd.read_csv('pca_omp_num_thread.csv').sort_values('num_threads')
threads = df['num_threads'].unique()

# ======================
# UTILITY FUNCTIONS
# ======================
def set_thread_xaxis(ax):
    """Configure x-axis for thread counts"""
    ax.set_xticks(threads)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(min(threads)-1, max(threads)+1)

def modern_plot(ax, title, xlabel, ylabel):
    """Apply modern styling to plots"""
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=10)
    ax.grid(True, alpha=0.2)
    ax.spines[['top','right']].set_visible(False)

# ======================
# CORE VISUALIZATIONS
# ======================

def plot_scaling_trend():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['num_threads'], df['total_time'], 
            marker='o', linestyle='-', 
            color='#2c7bb6', label='Total Time')
    
    # Calculate and plot trendline
    z = np.polyfit(df['num_threads'], df['total_time'], 3)
    p = np.poly1d(z)
    ax.plot(df['num_threads'], p(df['num_threads']), 
            '--', color='#d7191c', alpha=0.7, 
            label='Trendline (Cubic Fit)')
    
    set_thread_xaxis(ax)
    modern_plot(ax, 
               'Parallel Scaling Performance', 
               'Number of Threads', 
               'Total Execution Time (ms)')
    ax.legend(framealpha=1)
    plt.savefig('scaling_trend.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_speedup_analysis():
    df['speedup'] = df['total_time'].iloc[0] / df['total_time']
    df['efficiency'] = df['speedup'] / df['num_threads'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['num_threads'], df['speedup'], 
            marker='^', color='#1a9641',
            label=f'Max Speedup: {df["speedup"].max():.2f}x')
    
    # Efficiency on secondary axis
    ax2 = ax.twinx()
    ax2.plot(df['num_threads'], df['efficiency'], 
             marker='s', color='#fdae61',
             label=f'Peak Efficiency: {df["efficiency"].max():.1f}%')
    
    set_thread_xaxis(ax)
    modern_plot(ax, 
               'Parallel Speedup Analysis', 
               'Number of Threads', 
               'Speedup (T₁/Tₙ)')
    ax2.set_ylabel('Efficiency (%)', fontweight='bold', labelpad=10)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, framealpha=1)
    
    plt.savefig('speedup_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_operation_breakdown():
    ops = ['computeCovarianceMatrix_time',
           'computeEigenvectors_time',
           'trainPCAModel_time',
           'convertToFullEigenvectors_time',
           'projectData_time']
    
    colors = sns.color_palette("husl", len(ops))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(df))
    
    for op, color in zip(ops, colors):
        ax.bar(df['num_threads'], df[op], 
               bottom=bottom, width=1.5,
               color=color, edgecolor='white',
               linewidth=0.5, label=op.replace('_time',''))
        bottom += df[op]
    
    set_thread_xaxis(ax)
    modern_plot(ax, 
               'Operation Time Distribution', 
               'Number of Threads', 
               'Time (ms)')
    ax.legend(bbox_to_anchor=(1.05, 1), framealpha=1)
    plt.savefig('operation_breakdown.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_bottleneck_scaling():
    bottlenecks = ['computeCovarianceMatrix_time',
                  'computeEigenvectors_time',
                  'trainPCAModel_time']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for op in bottlenecks:
        ax.plot(df['num_threads'], df[op], 
                marker='o', linestyle='-',
                label=op.replace('_time',''))
    
    set_thread_xaxis(ax)
    modern_plot(ax, 
               'Bottleneck Operation Scaling', 
               'Number of Threads', 
               'Time (ms)')
    ax.legend(framealpha=1)
    plt.savefig('bottleneck_scaling.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_memory_throughput():
    df['throughput'] = (df['computeMean_data_size'] * 4) / (df['computeMean_time'] * 1e6)  # GB/s
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['num_threads'], df['throughput'],
            marker='D', color='#7b3294',
            label='Memory Throughput')
    
    set_thread_xaxis(ax)
    modern_plot(ax, 
               'Memory Bandwidth Utilization', 
               'Number of Threads', 
               'Throughput (GB/s)')
    ax.legend(framealpha=1)
    plt.savefig('memory_throughput.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_recognition_performance():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['num_threads'], df['recognizeFace_time'],
            marker='s', color='#e66101',
            label='Recognition Time')
    
    set_thread_xaxis(ax)
    modern_plot(ax, 
               'Face Recognition Performance', 
               'Number of Threads', 
               'Time (ms)')
    ax.legend(framealpha=1)
    plt.savefig('recognition_performance.png', bbox_inches='tight', dpi=300)
    plt.close()

# ======================
# EXECUTION
# ======================
if __name__ == "__main__":
    plot_scaling_trend()
    plot_speedup_analysis()
    plot_operation_breakdown()
    plot_bottleneck_scaling()
    plot_memory_throughput()
    plot_recognition_performance()
    
    print("""
    Successfully generated modern visualizations:
    1. scaling_trend.png - Parallel scaling with trendline
    2. speedup_analysis.png - Speedup and efficiency
    3. operation_breakdown.png - Time distribution by operation
    4. bottleneck_scaling.png - Key bottleneck operations
    5. memory_throughput.png - Memory bandwidth usage
    6. recognition_performance.png - Recognition latency
    """)