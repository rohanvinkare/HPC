import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# Create directory for plots
os.makedirs('openmp_lda_plots', exist_ok=True)

# Load the data
df = pd.read_csv('lda_omp_num_images.csv')

# Standardize column names
df.columns = [col.replace('::', '_') for col in df.columns]

# Set professional styling
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Color palette
main_color = '#1f77b4'
highlight_color = '#ff7f0e'

# 1. Total Time vs Number of Images
plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['total_time'], 
         marker='o', color=main_color, linewidth=2)
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Total Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Total Execution Time (8 Threads)', fontweight='bold')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/total_time_vs_images.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Time Breakdown for Largest Dataset
max_images = df['num_images'].max()
max_row = df[df['num_images'] == max_images].iloc[0]

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

times = [max_row[comp] for comp in components]
labels = [comp.replace('_time', '').replace('_', ' ') for comp in components]

# Sort by time
sorted_idx = np.argsort(times)[::-1]
times_sorted = [times[i] for i in sorted_idx]
labels_sorted = [labels[i] for i in sorted_idx]

plt.figure(figsize=(12, 6))
plt.barh(labels_sorted, times_sorted, color=main_color, alpha=0.8)
plt.xlabel('Time (ms)', fontweight='bold')
plt.title(f'OpenMP LDA Time Breakdown ({max_images} Images, 8 Threads)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig('openmp_lda_plots/time_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Compute-Intensive Operations Scaling
compute_ops = [
    'applyPCA_time',
    'projectData_time',
    'trainLDAModel_time'
]

plt.figure(figsize=(10, 6))
for op in compute_ops:
    plt.plot(df['num_images'], df[op], 
             marker='o', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Compute-Intensive Operations Scaling', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('openmp_lda_plots/compute_ops_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Data Size vs Operation Time
plt.figure(figsize=(10, 6))
plt.scatter(df['projectData_data_size']/1e6, df['projectData_time'],
            c=df['num_images'], cmap='viridis', s=100, alpha=0.7)
plt.colorbar(label='Number of Images')
plt.xlabel('Project Data Size (Millions of operations)', fontweight='bold')
plt.ylabel('Execution Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Project Data Performance', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('openmp_lda_plots/data_size_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. LDA-Specific Operations
lda_ops = [
    'computeWithinClassScatter_time',
    'computeBetweenClassScatter_time',
    'findLDAProjection_time'
]

plt.figure(figsize=(10, 6))
for op in lda_ops:
    plt.plot(df['num_images'], df[op], 
             marker='s', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: LDA-Specific Operations', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('openmp_lda_plots/lda_specific_ops.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Memory Operations Analysis
mem_ops = [
    'computeMean_time',
    'centerData_time',
    'projectData_time'
]

plt.figure(figsize=(10, 6))
for op in mem_ops:
    plt.plot(df['num_images'], df[op], 
             marker='D', label=op.replace('_time', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('OpenMP LDA: Memory-Bound Operations', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('openmp_lda_plots/memory_ops.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Throughput Analysis
df['throughput'] = df['num_images'] / (df['total_time'] / 1000)  # images/second

plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['throughput'], 
         marker='o', color=highlight_color, linewidth=2)
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Throughput (images/second)', fontweight='bold')
plt.title('OpenMP LDA: Processing Throughput (8 Threads)', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('openmp_lda_plots/throughput.png', dpi=300, bbox_inches='tight')
plt.close()

print("All OpenMP LDA performance plots saved to 'openmp_lda_plots' directory")