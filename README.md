# Facial Recognition using PCA and LDA with Parallel Computing

## Project Structure

```
.
├── data/
│   ├── data_40/          # First 40 images of AT&T dataset
│   ├── data_80/          # Next 40 images (total 80)
│   ├── ...               # Subsequent chunks (up to data_400)
├── LDA_CUDA/             # CUDA implementation of LDA
│   ├── commands.md       # Execution commands
│   └── ...               # Source files
├── LDA_OMP/              # OpenMP implementation of LDA
├── LDA_SEQ/              # Sequential LDA
├── PCA_CUDA/             # CUDA implementation of PCA
├── PCA_OMP/              # OpenMP PCA
├── PCA_SEQ/              # Sequential PCA
├── PCA_SEQ_inbuilt/      # PCA using built-in libraries
├── Test_Graphs/          # Performance comparison graphs
├── LDA_COMPARE/          # LDA comparison scripts
├── PCA_Compair/          # PCA comparison scripts
├── Test_Result_CSV/      # All test results in CSV format
└── README.md             # This file
```

## Requirements

- C++17 compiler (g++/clang++)
- OpenMP (for CPU parallelization)
- CUDA Toolkit (for GPU acceleration)
- OpenCV (for image processing)
- Python (for generating graphs, optional)

## Installation

```bash
# Install dependencies on Ubuntu
sudo apt install g++ libopencv-dev python3-pip
pip3 install matplotlib pandas  # For graph generation
```

## How to Run

## Refer to individual `commands.md` for algorithm-specific commands to run

1. **Sequential PCA/LDA**:

   ```bash
   cd PCA_SEQ/
   g++ -o pca_seq pca_seq.cpp `pkg-config --cflags --libs opencv4` -O3
   ./pca_seq ../data/data_40/
   ```

2. **OpenMP PCA/LDA**:

   ```bash
   cd PCA_OMP/
   g++ -o pca_omp pca_omp.cpp `pkg-config --cflags --libs opencv4` -O3 -fopenmp
   ./pca_omp ../data/data_80/ 4  # Use 4 threads
   ```

3. **CUDA PCA/LDA**:
   ```bash
   cd PCA_CUDA/
   nvcc -o pca_cuda pca_cuda.cu `pkg-config --cflags --libs opencv4` -O3
   ./pca_cuda ../data/data_160/
   ```

## Generating Results

1. Run tests from each implementation folder using `commands.md`
2. Collect CSVs in `Test_Result_CSV/`
3. Generate graphs:
   ```bash
   cd Test_Graphs/
   python3 plot_results.py ../Test_Result_CSV/pca_results.csv
   ```

## Expected Outputs

- Recognition accuracy in console
- Eigenfaces/Fisherfaces as PGM files
- Performance metrics in CSV
- Comparative graphs in `Test_Graphs/`

## Notes

- Dataset paths must be relative to executable
- For large datasets (>200 images), use CUDA for best performance
