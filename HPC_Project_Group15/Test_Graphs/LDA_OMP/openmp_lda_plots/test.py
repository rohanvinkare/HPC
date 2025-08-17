import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for plots
os.makedirs('openmp_lda_plots', exist_ok=True)

# Load the data
df = pd.read_csv('lda_omp_num_thread.csv')

# Standardize column names
df.columns = [col.replace('::', '_') for col in df.columns]

# Set professional styling
plt.style.use('seaborn')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Color palette
main_color = '#1f77b4'
highlight_color = '#ff7f0e'

# 1. Total Time vs Number of Threads
plt.figure(figsize=(10, 6))
plt.plot(df['num_threads'], df['total_time'], 
         marker='o', color=main_color, linewidth=2)
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Total Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Total Execution Time vs Thread Count\n(400 Images)', fontweight='bold')
plt.xticks(df['num_threads'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/total_time_vs_threads.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Speedup Plot
baseline_time = df[df['num_threads'] == 2]['total_time'].values[0]
df['speedup'] = baseline_time / df['total_time']
df['ideal_speedup'] = df['num_threads'] / 2

plt.figure(figsize=(10, 6))
plt.plot(df['num_threads'], df['speedup'], 
         label='Actual Speedup', marker='o', color=main_color, linewidth=2)
plt.plot(df['num_threads'], df['ideal_speedup'], 
         label='Ideal Speedup', linestyle='--', color='gray', linewidth=2)
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Speedup (x)', fontweight='bold')
plt.title('OpenMP LDA: Speedup vs Thread Count\n(Reference: 2 threads)', fontweight='bold')
plt.xticks(df['num_threads'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/speedup_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Efficiency Plot
df['efficiency'] = (df['speedup'] / df['num_threads']) * 2  # Normalized to 2 threads

plt.figure(figsize=(10, 6))
plt.plot(df['num_threads'], df['efficiency'], 
         marker='o', color=main_color, linewidth=2)
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Parallel Efficiency', fontweight='bold')
plt.title('OpenMP LDA: Parallel Efficiency vs Thread Count', fontweight='bold')
plt.xticks(df['num_threads'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/efficiency_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Time Breakdown for Best Thread Count
best_threads = df.loc[df['total_time'].idxmin(), 'num_threads']
best_row = df[df['num_threads'] == best_threads].iloc[0]

components = [
    'PGMImage_load_time',
    'loadHierarchicalDataset_time',
    'computeMean_time',
    'centerData_time',
    'applyPCA_time',
    'projectData_time',
    'computeWithinClassScatter_time',
    'computeBetweenClassScatter_time',
    'findLDAProjection_time',
    'trainLDAModel_time',
    'distance_computations_time',
    'recognizeFace_time'
]

times = [best_row[comp] for comp in components]
labels = [comp.replace('_time', '').replace('_', ' ') for comp in components]

# Sort by time
sorted_idx = np.argsort(times)[::-1]
times_sorted = [times[i] for i in sorted_idx]
labels_sorted = [labels[i] for i in sorted_idx]

plt.figure(figsize=(12, 6))
plt.barh(labels_sorted, times_sorted, color=main_color, alpha=0.8)
plt.xlabel('Time (ms)', fontweight='bold')
plt.title(f'OpenMP LDA Time Breakdown\n({best_threads} Threads, 400 Images)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig('openmp_lda_plots/time_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Parallel Operations Scaling
parallel_ops = [
    'loadHierarchicalDataset_time',
    'computeMean_time',
    'centerData_time',
    'projectData_time',
    'trainLDAModel_time'
]

plt.figure(figsize=(10, 6))
for op in parallel_ops:
    plt.plot(df['num_threads'], df[op], 
             marker='o', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Parallel Operation Scaling', fontweight='bold')
plt.xticks(df['num_threads'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/parallel_ops_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Compute-Intensive Operations Analysis
compute_ops = [
    'applyPCA_time',
    'projectData_time',
    'trainLDAModel_time'
]

plt.figure(figsize=(10, 6))
for op in compute_ops:
    plt.plot(df['num_threads'], df[op], 
             marker='s', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Compute-Intensive Operations', fontweight='bold')
plt.xticks(df['num_threads'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/compute_intensive_ops.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. LDA-Specific Operations
lda_ops = [
    'computeWithinClassScatter_time',
    'computeBetweenClassScatter_time',
    'findLDAProjection_time'
]

plt.figure(figsize=(10, 6))
for op in lda_ops:
    plt.plot(df['num_threads'], df[op], 
             marker='^', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: LDA-Specific Operations', fontweight='bold')
plt.xticks(df['num_threads'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/lda_specific_ops.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Memory Operations Analysis
mem_ops = [
    'computeMean_time',
    'centerData_time',
    'projectData_time'
]

plt.figure(figsize=(10, 6))
for op in mem_ops:
    plt.plot(df['num_threads'], df[op], 
             marker='D', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Memory-Bound Operations', fontweight='bold')
plt.xticks(df['num_threads'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/memory_ops.png', dpi=300, bbox_inches='tight')
plt.close()

print("All OpenMP LDA performance plots saved to 'openmp_lda_plots' directory")