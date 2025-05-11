
## Compile with g++
```
g++ -std=c++17 -O3 your_file.cpp -o pca_analysis \
$(pkg-config --cflags --libs opencv4) -lstdc++fs
```

## Run the program
```
./pca_analysis
```