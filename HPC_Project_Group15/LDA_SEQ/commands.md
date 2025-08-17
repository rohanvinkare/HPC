# Face Recognition PCA Project Commands

## Compilation Commands

### Basic Compilation
```bash
g++ -std=c++17 face_recognition_lda.cpp -o face_recognition_lda -lstdc++fs
```

## Running the Program
```bash
# Run the program for training
./face_recognition_lda --train /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/Data/data_40 100 my_lda_model.dat

# Run the program for testing
./face_recognition_lda --test my_lda_model.dat /home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/att_faces/s2/1.pgm

# Run the program for eval on database
./face_recognition_lda --eval my_lda_model.dat faces_dataset

```
