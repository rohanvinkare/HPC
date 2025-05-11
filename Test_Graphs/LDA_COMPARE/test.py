import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# Create directory for plots
os.makedirs('lda_comparison_plots', exist_ok=True)

# Load and standardize all datasets
def load_and_standardize(filename):
    df = pd.read_csv(filename)
    # Standardize column names
    df.columns = [col.replace('::', '_')
                  .replace('cuda', '')
                  .replace('CenterData', 'centerData')
                  .replace('Compute', 'compute')
                  .replace('ProjectData', 'projectData')
                  .strip() for col in df.columns]
    return df

# Load all datasets
seq_df = load_and_standardize('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Results/lda_seq.csv')
omp_thread_df = load_and_standardize('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Results/lda_omp_num_thread.csv')
omp_img_df = load_and_standardize('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Results/lda_omp_num_images.csv')
cuda_df = load_and_standardize('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Results/lda_cuda.csv')

# Set professional styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Color palette
colors = {
    'Sequential': '#1f77b4',
    'OpenMP (8 threads)': '#ff7f0e',
    'OpenMP (best threads)': '#d62728',
    'CUDA': '#2ca02c'
}

# Helper function to safely get component times
def get_component_time(row, component):
    variations = [
        component,
        component.replace('compute', 'cudaCompute'),
        component.replace('centerData', 'cudaCenterData'),
        component.replace('projectData', 'cudaProjectData')
    ]
    for var in variations:
        try:
            return row[var]
        except KeyError:
            continue
    return np.nan

# 1. Total Time Comparison (Varying Images)
plt.figure(figsize=(12, 6))
plt.plot(seq_df['num_images'], seq_df['total_time'], 
         label='Sequential', marker='o', color=colors['Sequential'], linewidth=2)
plt.plot(omp_img_df['num_images'], omp_img_df['total_time'], 
         label='OpenMP (8 threads)', marker='s', color=colors['OpenMP (8 threads)'], linewidth=2)
plt.plot(cuda_df['num_images'], cuda_df['total_time'], 
         label='CUDA', marker='^', color=colors['CUDA'], linewidth=2)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Total Time (ms)', fontweight='bold')
plt.title('LDA Implementation Comparison: Total Execution Time', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('lda_comparison_plots/total_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Speedup Comparison (vs Sequential)
plt.figure(figsize=(12, 6))
# Calculate speedups
omp_img_df['speedup'] = seq_df['total_time'] / omp_img_df['total_time']
cuda_df['speedup'] = seq_df['total_time'] / cuda_df['total_time']

plt.plot(omp_img_df['num_images'], omp_img_df['speedup'], 
         label='OpenMP (8 threads)', marker='s', color=colors['OpenMP (8 threads)'], linewidth=2)
plt.plot(cuda_df['num_images'], cuda_df['speedup'], 
         label='CUDA', marker='^', color=colors['CUDA'], linewidth=2)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Speedup (x) vs Sequential', fontweight='bold')
plt.title('LDA Implementation Speedup Comparison', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('lda_comparison_plots/speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Thread Scaling Analysis (400 images)
best_threads = omp_thread_df.loc[omp_thread_df['total_time'].idxmin(), 'num_threads']
plt.figure(figsize=(12, 6))
plt.plot(omp_thread_df['num_threads'], omp_thread_df['total_time'], 
         marker='o', color=colors['OpenMP (best threads)'], linewidth=2)
plt.axhline(y=seq_df[seq_df['num_images'] == 400]['total_time'].values[0], 
            color=colors['Sequential'], linestyle='--', label='Sequential')
plt.axhline(y=cuda_df[cuda_df['num_images'] == 400]['total_time'].values[0], 
            color=colors['CUDA'], linestyle='--', label='CUDA')

plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Total Time (ms)', fontweight='bold')
plt.title(f'OpenMP Thread Scaling (400 Images)\nBest: {best_threads} threads', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/thread_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Time Breakdown Comparison (400 images)
components = [
    'loadHierarchicalDataset_time',
    'computeMean_time',
    'centerData_time',
    'applyPCA_time',
    'projectData_time',
    'trainLDAModel_time',
    'recognizeFace_time'
]

labels = [comp.replace('_time', '').replace('_', ' ') for comp in components]
x = np.arange(len(labels))
width = 0.25

seq_400 = seq_df[seq_df['num_images'] == 400].iloc[0]
omp_400 = omp_thread_df[(omp_thread_df['num_threads'] == best_threads) & 
                        (omp_thread_df['num_images'] == 400)].iloc[0]
cuda_400 = cuda_df[cuda_df['num_images'] == 400].iloc[0]

plt.figure(figsize=(14, 6))
plt.bar(x - width, [get_component_time(seq_400, comp) for comp in components], 
        width, label='Sequential', color=colors['Sequential'])
plt.bar(x, [get_component_time(omp_400, comp) for comp in components], 
        width, label=f'OpenMP ({best_threads} threads)', color=colors['OpenMP (best threads)'])
plt.bar(x + width, [get_component_time(cuda_400, comp) for comp in components], 
        width, label='CUDA', color=colors['CUDA'])

plt.xlabel('Components', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('LDA Component Time Comparison (400 Images)', fontweight='bold')
plt.xticks(x, labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('lda_comparison_plots/component_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Compute-Intensive Operations Comparison
compute_ops = [
    'applyPCA_time',
    'projectData_time',
    'trainLDAModel_time'
]

plt.figure(figsize=(12, 6))
for op in compute_ops:
    plt.plot(seq_df['num_images'], seq_df[op], 
             label=f'Seq: {op.replace("_time", "")}', 
             color=colors['Sequential'], linestyle='-', marker='o', markersize=4)
    plt.plot(omp_img_df['num_images'], omp_img_df[op], 
             label=f'OMP: {op.replace("_time", "")}', 
             color=colors['OpenMP (8 threads)'], linestyle='-', marker='s', markersize=4)
    plt.plot(cuda_df['num_images'], [get_component_time(row, op) for _, row in cuda_df.iterrows()], 
             label=f'CUDA: {op.replace("_time", "")}', 
             color=colors['CUDA'], linestyle='-', marker='^', markersize=4)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('Compute-Intensive Operations Comparison', fontweight='bold')
plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.15), loc='center')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
plt.savefig('lda_comparison_plots/compute_ops_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Throughput Comparison
plt.figure(figsize=(12, 6))
for name, df, color in [
    ('Sequential', seq_df, colors['Sequential']),
    ('OpenMP (8 threads)', omp_img_df, colors['OpenMP (8 threads)']),
    ('CUDA', cuda_df, colors['CUDA'])
]:
    throughput = df['num_images'] / (df['total_time'] / 1000)  # images/second
    plt.plot(df['num_images'], throughput, 
             label=name, marker='o', color=color, linewidth=2)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Throughput (images/second)', fontweight='bold')
plt.title('LDA Implementation Throughput Comparison', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.savefig('lda_comparison_plots/throughput_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Memory Operations Comparison
mem_ops = [
    'computeMean_time',
    'centerData_time',
    'projectData_time'
]

plt.figure(figsize=(12, 6))
for op in mem_ops:
    plt.plot(seq_df['num_images'], seq_df[op], 
             label=f'Seq: {op.replace("_time", "")}', 
             color=colors['Sequential'], linestyle='-', marker='o', markersize=4)
    plt.plot(omp_img_df['num_images'], omp_img_df[op], 
             label=f'OMP: {op.replace("_time", "")}', 
             color=colors['OpenMP (8 threads)'], linestyle='-', marker='s', markersize=4)
    plt.plot(cuda_df['num_images'], [get_component_time(row, op) for _, row in cuda_df.iterrows()], 
             label=f'CUDA: {op.replace("_time", "")}', 
             color=colors['CUDA'], linestyle='-', marker='^', markersize=4)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('Memory Operations Comparison', fontweight='bold')
plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.15), loc='center')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
plt.savefig('lda_comparison_plots/memory_ops_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Strong Scaling Efficiency
seq_time_400 = seq_df[seq_df['num_images'] == 400]['total_time'].values[0]
omp_thread_df['efficiency'] = seq_time_400 / (omp_thread_df['total_time'] * omp_thread_df['num_threads'])

plt.figure(figsize=(10, 6))
plt.plot(omp_thread_df['num_threads'], omp_thread_df['efficiency'], 
         marker='o', color=colors['OpenMP (best threads)'], linewidth=2)
plt.axhline(y=1, color='gray', linestyle='--', label='Ideal')
plt.xlabel('Number of Threads', fontweight='bold')
plt.ylabel('Parallel Efficiency', fontweight='bold')
plt.title('OpenMP Strong Scaling Efficiency (400 Images)', fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('lda_comparison_plots/strong_scaling_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

print("All LDA comparison plots saved to 'lda_comparison_plots' directory")