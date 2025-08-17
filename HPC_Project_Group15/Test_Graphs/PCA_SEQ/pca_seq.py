import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter

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
df = pd.read_csv('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/PCA_SEQ/pca_seq.csv')

# 1. Total Time vs Number of Images
plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['total_time']/1000, 
         'bo-', linewidth=2, markersize=8, markerfacecolor='white')
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Total Time (seconds)', fontweight='bold')
plt.title('PCA Total Execution Time\n(Sequential Implementation)', pad=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('pca_total_time.png')
plt.close()

# 2. Time Breakdown by Operation (Stacked Area)
ops = ['computeMean_time', 'centerData_time', 'computeCovarianceMatrix_time',
       'computeEigenvectors_time', 'convertToFullEigenvectors_time',
       'projectData_time', 'trainPCAModel_time']
op_labels = [op.replace('_time', '') for op in ops]
colors = plt.cm.tab20(np.linspace(0, 1, len(ops)))

plt.figure(figsize=(12, 7))
plt.stackplot(df['num_images'], 
             [df[op] for op in ops],
             labels=op_labels,
             colors=colors,
             alpha=0.8,
             edgecolor='black')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('PCA Operation Time Breakdown', pad=20)
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig('pca_time_breakdown.png', bbox_inches='tight')
plt.close()

# 3. Time Complexity Analysis (Log-Log)
plt.figure(figsize=(10, 6))
plt.loglog(df['num_images'], df['total_time'], 
          'ro-', linewidth=2, markersize=8, label='Actual Performance')
plt.loglog(df['num_images'], 5*df['num_images']**2, 
          'b--', linewidth=2, label='O(n²) Reference')
plt.loglog(df['num_images'], 0.1*df['num_images']**3, 
          'g:', linewidth=2, label='O(n³) Reference')

plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.minorticks_off()

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('PCA Time Complexity Analysis', pad=20)
plt.legend()
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.savefig('pca_complexity_analysis.png')
plt.close()

# 4. Bottleneck Operations Growth
bottlenecks = {
    'computeCovarianceMatrix_time': 'Covariance Matrix',
    'trainPCAModel_time': 'Train Model',
    'convertToFullEigenvectors_time': 'Eigenvector Conversion'
}

plt.figure(figsize=(10, 6))
for col, label in bottlenecks.items():
    plt.plot(df['num_images'], df[col], 
             'o-', linewidth=2, markersize=8, label=label)

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time (ms)', fontweight='bold')
plt.title('PCA Bottleneck Operations Growth', pad=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('pca_bottlenecks.png')
plt.close()

# 5. Memory Usage vs Time
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['computeMean_data_size']/1e6, df['total_time']/1000,
                c=df['num_images'], s=150, cmap='viridis',
                edgecolor='black', linewidth=0.5)

cbar = plt.colorbar(sc)
cbar.set_label('Number of Images', fontweight='bold')

plt.xlabel('Data Size (Millions of elements)', fontweight='bold')
plt.ylabel('Total Time (seconds)', fontweight='bold')
plt.title('Memory Usage vs Execution Time', pad=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('pca_memory_vs_time.png')
plt.close()

# 6. Per-Image Processing Time
plt.figure(figsize=(10, 6))
plt.plot(df['num_images'], df['total_time']/df['num_images'], 
        'go-', linewidth=2, markersize=8, markerfacecolor='white')

plt.xlabel('Number of Images', fontweight='bold')
plt.ylabel('Time per Image (ms)', fontweight='bold')
plt.title('PCA Efficiency: Time per Image', pad=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('pca_time_per_image.png')
plt.close()

print("All graphs generated successfully:")
print("- pca_total_time.png")
print("- pca_time_breakdown.png")
print("- pca_complexity_analysis.png")
print("- pca_bottlenecks.png")
print("- pca_memory_vs_time.png")
print("- pca_time_per_image.png")