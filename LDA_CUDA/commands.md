# CUDA-Accelerated LDA Face Recognition System

## Compilation Command
```bash
nvcc -std=c++17 -Xcompiler -fopenmp -arch=sm_70 -Wno-deprecated-gpu-targets face_recognition_lda_cuda.cu -o face_recognition_lda_cuda -ltbb -lstdc++fs

nvcc -std=c++17 --extended-lambda -Xcompiler -fopenmp -arch=sm_70 -Wno-deprecated-gpu-targets face_recognition_lda_cuda.cu -o face_recognition_lda_cuda -ltbb -lstdc++fs
```

## Usage Guide

### Training Mode
```bash
./face_recognition_lda_cuda --train dataset_directory [num_components] [model_output_file]
```
Parameters:
- `dataset_directory`: Directory containing training images
- `num_components`: (Optional) Number of LDA components
- `model_output_file`: (Optional) Output path for trained model

### Testing Mode
```bash
./face_recognition_lda_cuda --test model_file test_image.pgm
```
Parameters:
- `model_file`: Path to trained model file
- `test_image.pgm`: Image file to test

### Evaluation Mode
```bash
./face_recognition_lda_cuda --eval model_file dataset_directory
```
Parameters:
- `model_file`: Path to trained model file
- `dataset_directory`: Directory containing test images

## Example Usage
```bash
# Train the model
./face_recognition_lda_cuda --train /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_400 100 model.dat

# Test a single image
./face_recognition_lda_cuda --test model.dat ./test/face1.pgm

# Evaluate on test dataset
./face_recognition_lda_cuda --eval model.dat ./dataset/test
```
