import pandas as pd
import matplotlib.pyplot as plt
import os

# Create directory for plots if it doesn't exist
if not os.path.exists('comparison_plots'):
    os.makedirs('comparison_plots')

# Load the datasets
seq_df = pd.read_csv('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/PCA_SEQ/pca_seq.csv')
omp_df = pd.read_csv('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/PCA_OMP/num_images/pca_omp_num_image.csv')
cuda_df = pd.read_csv('/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Test_Graphs/PCA_CUDA/pca_cuda.csv')

# Standardize column names
seq_df.columns = [col.replace('::', '_').strip() for col in seq_df.columns]
omp_df.columns = [col.strip() for col in omp_df.columns]
cuda_df.columns = [col.strip() for col in cuda_df.columns]

# Rename image count columns to be consistent
seq_df = seq_df.rename(columns={'num_images': 'image_count'})
omp_df = omp_df.rename(columns={'number_of_images': 'image_count'})
cuda_df = cuda_df.rename(columns={'number_of_images': 'image_count'})

# Merge datasets
merged_df = seq_df.merge(omp_df, on='image_count', how='inner', suffixes=('_seq', '_omp'))
merged_df = merged_df.merge(cuda_df, on='image_count', how='inner')
merged_df.rename(columns={'total_time': 'total_time_cuda'}, inplace=True)

# Function to safely get column values with fallbacks
def get_col_value(row, col_name, suffixes):
    for suffix in suffixes:
        try:
            return row[f'{col_name}{suffix}']
        except KeyError:
            continue
    try:
        return row[col_name]
    except KeyError:
        return None

# Plot 1: Total Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['total_time_seq'], label='Sequential', marker='o')
plt.plot(merged_df['image_count'], merged_df['total_time_omp'], label='OpenMP', marker='s')
plt.plot(merged_df['image_count'], merged_df['total_time_cuda'], label='CUDA', marker='^')
plt.xlabel('Number of Images')
plt.ylabel('Total Time (ms)')
plt.title('Total Execution Time Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/total_time_comparison.png')
plt.close()

# Plot 2: Speedup Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['total_time_seq'] / merged_df['total_time_omp'], 
         label='OpenMP Speedup', marker='o')
plt.plot(merged_df['image_count'], merged_df['total_time_seq'] / merged_df['total_time_cuda'], 
         label='CUDA Speedup', marker='s')
plt.xlabel('Number of Images')
plt.ylabel('Speedup (x)')
plt.title('Speedup Compared to Sequential Implementation')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/speedup_comparison.png')
plt.close()

# Plot 3: Time Breakdown for largest dataset
max_images = merged_df['image_count'].max()
max_row = merged_df[merged_df['image_count'] == max_images].iloc[0]

components = [
    'PGMImage_load_time',
    'loadHierarchicalDataset_time',
    'computeMean_time',
    'centerData_time',
    'computeCovarianceMatrix_time',
    'powerIteration_time',
    'computeEigenvectors_time',
    'convertToFullEigenvectors_time',
    'projectData_time',
    'distance_computations_time',
    'recognizeFace_time'
]

# Handle different naming conventions
seq_times = []
omp_times = []
cuda_times = []
labels = []

for component in components:
    # Sequential
    seq_val = get_col_value(max_row, component, ['_seq', ''])
    # OpenMP
    omp_val = get_col_value(max_row, component, ['_omp', ''])
    # CUDA (skip power iteration)
    if component != 'powerIteration_time':
        cuda_val = get_col_value(max_row, component, ['', '_cuda'])
    else:
        cuda_val = None
    
    if seq_val is not None and omp_val is not None and (cuda_val is not None or component == 'powerIteration_time'):
        seq_times.append(seq_val)
        omp_times.append(omp_val)
        if component != 'powerIteration_time':
            cuda_times.append(cuda_val)
        labels.append(component.replace('_time', '').replace('_', ' '))

plt.figure(figsize=(14, 8))
width = 0.25
x = range(len(labels))

plt.bar([i - width for i in x], seq_times, width, label='Sequential')
plt.bar([i for i in x], omp_times, width, label='OpenMP')
plt.bar([i + width for i in x[:len(cuda_times)]], cuda_times, width, label='CUDA')

plt.xlabel('Components')
plt.ylabel('Time (ms)')
plt.title(f'Time Breakdown Comparison for {max_images} Images')
plt.xticks([i for i in x], labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('comparison_plots/time_breakdown_comparison.png')
plt.close()

# Plot 4: Compute Mean Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['computeMean_time_seq'], label='Sequential', marker='o')
plt.plot(merged_df['image_count'], merged_df['computeMean_time_omp'], label='OpenMP', marker='s')
plt.plot(merged_df['image_count'], merged_df['computeMean_time'], label='CUDA', marker='^')
plt.xlabel('Number of Images')
plt.ylabel('Time (ms)')
plt.title('Compute Mean Time Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/compute_mean_comparison.png')
plt.close()

# Plot 5: Covariance Matrix Computation Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['computeCovarianceMatrix_time_seq'], label='Sequential', marker='o')
plt.plot(merged_df['image_count'], merged_df['computeCovarianceMatrix_time_omp'], label='OpenMP', marker='s')
plt.plot(merged_df['image_count'], merged_df['computeCovarianceMatrix_time'], label='CUDA', marker='^')
plt.xlabel('Number of Images')
plt.ylabel('Time (ms)')
plt.title('Covariance Matrix Computation Time Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/covariance_matrix_comparison.png')
plt.close()

# Plot 6: Eigenvectors Computation Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['computeEigenvectors_time_seq'], label='Sequential', marker='o')
plt.plot(merged_df['image_count'], merged_df['computeEigenvectors_time_omp'], label='OpenMP', marker='s')
plt.plot(merged_df['image_count'], merged_df['computeEigenvectors_time'], label='CUDA', marker='^')
plt.xlabel('Number of Images')
plt.ylabel('Time (ms)')
plt.title('Eigenvectors Computation Time Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/eigenvectors_comparison.png')
plt.close()

# Plot 7: Data Projection Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['projectData_time_seq'], label='Sequential', marker='o')
plt.plot(merged_df['image_count'], merged_df['projectData_time_omp'], label='OpenMP', marker='s')
plt.plot(merged_df['image_count'], merged_df['projectData_time'], label='CUDA', marker='^')
plt.xlabel('Number of Images')
plt.ylabel('Time (ms)')
plt.title('Data Projection Time Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/project_data_comparison.png')
plt.close()

# Plot 8: Face Recognition Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(merged_df['image_count'], merged_df['recognizeFace_time_seq'], label='Sequential', marker='o')
plt.plot(merged_df['image_count'], merged_df['recognizeFace_time_omp'], label='OpenMP', marker='s')
plt.plot(merged_df['image_count'], merged_df['recognizeFace_time'], label='CUDA', marker='^')
plt.xlabel('Number of Images')
plt.ylabel('Time (ms)')
plt.title('Face Recognition Time Comparison')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plots/face_recognition_comparison.png')
plt.close()

print("All comparison plots have been saved in the 'comparison_plots' directory.")