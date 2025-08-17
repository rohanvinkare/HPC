import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# Create directory for plots
os.makedirs('cuda_lda_plots', exist_ok=True)

# Load the data
df = pd.read_csv('lda_cuda.csv')

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
cuda_color = '#2ca02c'  # Green color for CUDA
highlight_color = '#d62728'  # Red for highlighting

# 1. Total Time vs Number of Images
plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['total_time'], 
         marker='o', color=cuda_color, linewidth=2)
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Total Time (ms)', fontweight='bold')
plt.title('CUDA LDA: Total Execution Time', fontweight='bold')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('cuda_lda_plots/total_time_vs_images.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Time Breakdown for Largest Dataset
max_images = df['num_images'].max()
max_row = df[df['num_images'] == max_images].iloc[0]

components = [
    'PGMImage_load_time',
    'loadHierarchicalDataset_time',
    'cudaComputeMean_time',
    'cudaCenterData_time',
    'applyPCA_time',
    'cudaProjectData_time',
    'cudaComputeWithinClassScatter_time',
    'computeBetweenClassScatter_time',
    'findLDAProjection_time',
    'trainLDAModel_time',
    'cudaDistanceComputations_time',
    'recognizeFace_time'
]

times = [max_row[comp] for comp in components]
labels = [comp.replace('_time', '').replace('cuda', '').replace('_', ' ') for comp in components]

# Sort by time
sorted_idx = np.argsort(times)[::-1]
times_sorted = [times[i] for i in sorted_idx]
labels_sorted = [labels[i] for i in sorted_idx]

plt.figure(figsize=(12, 6))
plt.barh(labels_sorted, times_sorted, color=cuda_color, alpha=0.8)
plt.xlabel('Time (ms)', fontweight='bold')
plt.title(f'CUDA LDA Time Breakdown ({max_images} Images)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig('cuda_lda_plots/time_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. CUDA Kernel Performance Analysis
cuda_kernels = [
    'cudaComputeMean_time',
    'cudaCenterData_time',
    'cudaProjectData_time',
    'cudaComputeWithinClassScatter_time',
    'cudaDistanceComputations_time'
]

plt.figure(figsize=(10, 6))
for kernel in cuda_kernels:
    plt.plot(df['num_images'], df[kernel], 
             marker='o', label=kernel.replace('_time', '').replace('cuda', '').replace('_', ' '),
             linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('CUDA LDA: Kernel Performance', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('cuda_lda_plots/kernel_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Memory Operations Analysis
plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['cudaComputeMean_time'], label='Compute Mean', marker='s')
plt.plot(df['num_images'], df['cudaCenterData_time'], label='Center Data', marker='D')
plt.plot(df['num_images'], df['cudaProjectData_time'], label='Project Data', marker='^')
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('CUDA LDA: Memory-Bound Operations', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('cuda_lda_plots/memory_ops.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Data Transfer vs Computation Time
compute_time = df['total_time'] - df['loadHierarchicalDataset_time'] - df['PGMImage_load_time']
plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['total_time'], label='Total Time', color=cuda_color, linewidth=2)
plt.plot(df['num_images'], compute_time, label='Computation Time', linestyle='--', color=highlight_color)
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('CUDA LDA: Computation vs Total Time', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('cuda_lda_plots/computation_vs_total.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Throughput Analysis
df['throughput'] = df['num_images'] / (df['total_time'] / 1000)  # images/second

plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['throughput'], 
         marker='o', color=highlight_color, linewidth=2)
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Throughput (images/second)', fontweight='bold')
plt.title('CUDA LDA: Processing Throughput', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('cuda_lda_plots/throughput.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. LDA-Specific Operations
lda_ops = [
    'cudaComputeWithinClassScatter_time',
    'computeBetweenClassScatter_time',
    'findLDAProjection_time'
]

plt.figure(figsize=(10, 6))
for op in lda_ops:
    clean_name = op.replace('_time', '').replace('cuda', '').replace('_', ' ')
    plt.plot(df['num_images'], df[op], 
             marker='s', label=clean_name,
             linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('CUDA LDA: LDA-Specific Operations', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('cuda_lda_plots/lda_specific_ops.png', dpi=300, bbox_inches='tight')
plt.close()

print("All CUDA LDA performance plots saved to 'cuda_lda_plots' directory")