import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import os

# Set global style parameters
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 12,
    'figure.titlesize': 14,
    'axes.titlesize': 13,
    'axes.labelweight': 'bold',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

# Load data
df = pd.read_csv('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/PCA_OMP/num_images/pca_omp_num_image.csv')

# Create output directory if it doesn't exist
output_dir = '/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/PCA_OMP/num_images'
os.makedirs(output_dir, exist_ok=True)

# 1. Total Time vs Number of Images
plt.figure(figsize=(10, 6))
plt.plot(df['number_of_images'], df['total_time']/1000, 
         'bo-', linewidth=2, markersize=8, markerfacecolor='white')
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Total Time (seconds)', fontweight='bold')
plt.title('PCA Total Execution Time\n(Parallel Implementation)', pad=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/pca_omp_total_time.png')
plt.close()

# 2. Time Breakdown by Operation (Stacked Area)
ops = ['computeMean_time', 'centerData_time', 'computeCovarianceMatrix_time',
       'computeEigenvectors_time', 'convertToFullEigenvectors_time',
       'projectData_time', 'trainPCAModel_time']
op_labels = [op.replace('_time', '') for op in ops]
colors = plt.cm.tab20(np.linspace(0, 1, len(ops)))

plt.figure(figsize=(12, 7))
plt.stackplot(df['number_of_images'], 
             [df[op] for op in ops],
             labels=op_labels,
             colors=colors,
             alpha=0.8,
             edgecolor='black')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('PCA Operation Time Breakdown (OMP)', pad=20)
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig(f'{output_dir}/pca_omp_time_breakdown.png', bbox_inches='tight')
plt.close()

# 3. Time Complexity Analysis (Log-Log)
plt.figure(figsize=(10, 6))
plt.loglog(df['number_of_images'], df['total_time'], 
          'ro-', linewidth=2, markersize=8, label='Actual Performance')
plt.loglog(df['number_of_images'], 5*df['number_of_images']**2, 
          'b--', linewidth=2, label='O(n²) Reference')
plt.loglog(df['number_of_images'], 0.1*df['number_of_images']**3, 
          'g:', linewidth=2, label='O(n³) Reference')

plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.minorticks_off()

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('PCA Time Complexity Analysis (OMP)', pad=20)
plt.legend()
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.savefig(f'{output_dir}/pca_omp_complexity_analysis.png')
plt.close()

# 4. Bottleneck Operations Growth
bottlenecks = {
    'computeCovarianceMatrix_time': 'Covariance Matrix',
    'trainPCAModel_time': 'Train Model',
    'computeEigenvectors_time': 'Eigenvector Computation'
}

plt.figure(figsize=(10, 6))
for col, label in bottlenecks.items():
    plt.plot(df['number_of_images'], df[col], 
             'o-', linewidth=2, markersize=8, label=label)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('PCA Bottleneck Operations Growth (OMP)', pad=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{output_dir}/pca_omp_bottlenecks.png')
plt.close()

# 5. Memory Usage vs Time
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['computeMean_data_size']/1e6, df['total_time']/1000,
                c=df['number_of_images'], s=150, cmap='viridis',
                edgecolor='black', linewidth=0.5)

cbar = plt.colorbar(sc)
cbar.set_label('Number of Images', fontweight='bold')

plt.xlabel('Data Size (Millions of elements)', fontweight='bold')
plt.ylabel('Total Time (seconds)', fontweight='bold')
plt.title('Memory Usage vs Execution Time (OMP)', pad=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{output_dir}/pca_omp_memory_vs_time.png')
plt.close()

# 6. Per-Image Processing Time
plt.figure(figsize=(10, 6))
plt.plot(df['number_of_images'], df['total_time']/df['number_of_images'], 
        'go-', linewidth=2, markersize=8, markerfacecolor='white')

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time per Image (ms)', fontweight='bold')
plt.title('PCA Efficiency: Time per Image (OMP)', pad=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{output_dir}/pca_omp_time_per_image.png')
plt.close()

# 7. Parallel Efficiency
df['time_per_element'] = df['total_time'] / df['computeMean_data_size']
df['ideal_scaling'] = df['time_per_element'].iloc[0] * df['computeMean_data_size']

plt.figure(figsize=(10, 6))
plt.plot(df['number_of_images'], df['total_time']/1000, 
        'bo-', label='Actual', linewidth=2, markersize=8)
plt.plot(df['number_of_images'], df['ideal_scaling']/1000, 
        'r--', label='Ideal', linewidth=2)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (seconds)', fontweight='bold')
plt.title('PCA Parallel Efficiency (OMP)', pad=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{output_dir}/pca_omp_parallel_efficiency.png')
plt.close()

print("All graphs generated successfully in:", output_dir)
print("- pca_omp_total_time.png")
print("- pca_omp_time_breakdown.png")
print("- pca_omp_complexity_analysis.png")
print("- pca_omp_bottlenecks.png")
print("- pca_omp_memory_vs_time.png")
print("- pca_omp_time_per_image.png")
print("- pca_omp_parallel_efficiency.png")