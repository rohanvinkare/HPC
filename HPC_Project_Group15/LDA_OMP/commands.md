# Face Recognition LDA

# Compilation
```
g++ -fopenmp -std=c++17 face_recognition_lda.cpp -o face_recognition_lda
```

# Training
```
export OMP_NUM_THREADS=4
./face_recognition_lda --train /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_400
```

# Testing
```
export OMP_NUM_THREADS=4
./face_recognition_lda --test face_recognition_lda.dat /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/att_faces/s4/5.pgm
```
