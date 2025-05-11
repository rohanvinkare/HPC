import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Create directory for plots if it doesn't exist
if not os.path.exists('lda_comparison_plots'):
    os.makedirs('lda_comparison_plots')

# Load the dataset
lda_df = pd.read_csv('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/LDA_SEQ/lda_seq.csv')

# Standardize column names by replacing '::' with '_'
lda_df.columns = [col.replace('::', '_') for col in lda_df.columns]

# Set up professional styling
plt.style.use('seaborn')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Define colors for consistent styling
sequential_color = '#1f77b4'
openmp_color = '#ff7f0e'
cuda_color = '#2ca02c'
lda_color = '#d62728'

# 1. Total Time vs Number of Images
plt.figure(figsize=(10, 6))
plt.plot(lda_df['num_images'], lda_df['total_time'], 
         label='LDA', marker='o', color=lda_color, linewidth=2)
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Total Time (ms)', fontweight='bold')
plt.title('Total Execution Time for LDA Implementation', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/lda_total_time.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Time Breakdown for Largest Dataset (400 images)
max_images = lda_df['num_images'].max()
max_row = lda_df[lda_df['num_images'] == max_images].iloc[0]

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

times = [max_row[component] for component in components]
labels = [comp.replace('_time', '').replace('_', ' ') for comp in components]

# Sort by time (descending) for better visualization
sorted_indices = np.argsort(times)[::-1]
times_sorted = [times[i] for i in sorted_indices]
labels_sorted = [labels[i] for i in sorted_indices]

plt.figure(figsize=(12, 6))
plt.barh(labels_sorted, times_sorted, color=lda_color, alpha=0.8)
plt.xlabel('Time (ms)', fontweight='bold')
plt.title(f'Time Breakdown for LDA ({max_images} Images)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig('lda_comparison_plots/lda_time_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Key Operation Times vs Number of Images
key_operations = [
    'loadHierarchicalDataset_time',
    'applyPCA_time',
    'projectData_time',
    'trainLDAModel_time'
]

plt.figure(figsize=(10, 6))
for op in key_operations:
    plt.plot(lda_df['num_images'], lda_df[op], 
             label=op.replace('_time', '').replace('_', ' '), 
             marker='o', linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('Key Operation Times for LDA', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/lda_key_operations.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Data Size Scaling
data_size_metrics = [
    'computeMean_data_size',
    'centerData_data_size',
    'projectData_data_size',
    'trainLDAModel_data_size'
]

plt.figure(figsize=(10, 6))
for metric in data_size_metrics:
    plt.plot(lda_df['num_images'], lda_df[metric], 
             label=metric.replace('_data_size', '').replace('_', ' '),
             marker='s', linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Data Size (elements/operations)', fontweight='bold')
plt.title('Data Size Scaling in LDA Implementation', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/lda_data_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Compute-Intensive Operations Analysis
compute_ops = [
    'applyPCA_time',
    'projectData_time',
    'trainLDAModel_time'
]

plt.figure(figsize=(10, 6))
for op in compute_ops:
    plt.plot(lda_df['num_images'], lda_df[op], 
             label=op.replace('_time', '').replace('_', ' '),
             marker='^', linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('Compute-Intensive Operations in LDA', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/lda_compute_operations.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Memory Usage vs Performance
fig, ax1 = plt.subplots(figsize=(10, 6))

color = lda_color
ax1.set_xlabel('Number of Images', fontweight='bold')
ax1.set_ylabel('Time (ms)', color=color, fontweight='bold')
ax1.plot(lda_df['num_images'], lda_df['total_time'], 
         color=color, marker='o', label='Total Time')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#9467bd'
ax2.set_ylabel('Memory Usage (MB)', color=color, fontweight='bold')
total_memory = lda_df['projectData_data_size'] / (1024 * 1024)  # Convert to MB
ax2.plot(lda_df['num_images'], total_memory, 
         color=color, marker='s', label='Memory Usage')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('LDA Performance vs Memory Usage', fontweight='bold')
fig.tight_layout()
plt.savefig('lda_comparison_plots/lda_memory_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. LDA-Specific Operations
lda_specific_ops = [
    'computeWithinClassScatter_time',
    'computeBetweenClassScatter_time',
    'findLDAProjection_time'
]

plt.figure(figsize=(10, 6))
for op in lda_specific_ops:
    plt.plot(lda_df['num_images'], lda_df[op], 
             label=op.replace('_time', '').replace('_', ' '),
             marker='D', linewidth=2)
    
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('LDA-Specific Operation Times', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/lda_specific_operations.png', dpi=300, bbox_inches='tight')
plt.close()

print("All LDA performance plots have been saved in the 'lda_comparison_plots' directory.")