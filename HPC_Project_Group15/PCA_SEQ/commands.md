# Face Recognition PCA Project Commands

## Compilation Commands

### Basic Compilation

```bash
g++ -std=c++17 face_recognition_pca.cpp -o face_recognition_pca -lstdc++fs
```

## Running the Program

```bash
# Run the program for training
./face_recognition_pca --train /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_40 100 my_face_model.dat

# Run the program for testing
./face_recognition_pca --test my_face_model.dat /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_40/s2/1.pgm
```
