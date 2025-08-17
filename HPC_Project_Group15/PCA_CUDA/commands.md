# Face Recognition PCA-OMP Project Commands

## Compilation Commands

### Basic Compilation

```bash
nvcc face_recognition.cu -o face_recognition -lcublas -lcusolver
```

## Running the Program

```bash
./face_recognition --train /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_40 100 model_file.dat

./face_recognition --test model_file.dat /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/att_faces/s21/1.pgm

./face_recognition --eval model_file.dat test_dataset_directory
```
