# Face Recognition PCA-OMP Project Commands

## Compilation Commands

### Basic Compilation

```bash
g++ -fopenmp -std=c++17 pca_parallel.cpp -o pca_parallel
```

## Running the Program

```bash
# Run the program for training
OMP_NUM_THREADS=8 ./pca_parallel --train /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_400 100 model_output.dat

# Run the program for testing
OMP_NUM_THREADS=4 ./pca_parallel --test model_output.dat /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/att_faces/s4/1.pgm

# For evaluation
OMP_NUM_THREADS=4 ./pca_parallel --eval model_output.dat test_dataset_directory
```
