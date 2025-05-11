#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <map>
#include <iomanip>
#include <set>
#include <chrono>
#include <numeric>
#include <limits>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/extrema.h>

using namespace std;
namespace fs = std::filesystem;

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                    \
    do                                                                      \
    {                                                                       \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl;                        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Performance measurement structure
struct BenchmarkResult
{
    string operationName;
    double durationMs;
    size_t dataSize;
    string unit;
    int count = 1; // For averaging multiple runs
};

vector<BenchmarkResult> benchmarkResults;

// Timer class for measuring execution time
class Timer
{
private:
    chrono::time_point<chrono::high_resolution_clock> startTime;
    string operationName;
    size_t dataSize;
    string unit;

public:
    Timer(const string &name, size_t size = 0, const string &unitStr = "")
        : operationName(name), dataSize(size), unit(unitStr)
    {
        startTime = chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        auto endTime = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
        double durationMs = duration / 1000.0;

        // Check if this operation already exists
        bool found = false;
        for (auto &result : benchmarkResults)
        {
            if (result.operationName == operationName)
            {
                result.durationMs += durationMs;
                result.count++;
                found = true;
                break;
            }
        }

        if (!found)
        {
            benchmarkResults.push_back({operationName, durationMs, dataSize, unit});
        }
    }
};

// Class to represent and load PGM images
class PGMImage
{
private:
    int width, height;
    int maxVal;
    vector<unsigned char> pixels;

public:
    PGMImage() : width(0), height(0), maxVal(0) {}

    bool load(const string &filename)
    {
        Timer timer("PGMImage::load", fs::file_size(filename), "bytes");

        ifstream file(filename, ios::binary);
        if (!file)
        {
            cerr << "Failed to open " << filename << endl;
            return false;
        }

        string magic;
        file >> magic;

        if (magic != "P5")
        {
            cerr << "Not a valid PGM file: " << filename << endl;
            return false;
        }

        // Skip comments
        char line[256];
        file.getline(line, 256);
        while (file.peek() == '#')
        {
            file.getline(line, 256);
        }

        // Read dimensions and max value
        file >> width >> height >> maxVal;

        // Skip whitespace
        file.get();

        // Read pixel data
        pixels.resize(width * height);
        file.read(reinterpret_cast<char *>(pixels.data()), pixels.size());

        if (!file)
        {
            cerr << "Error reading pixel data from " << filename << endl;
            return false;
        }

        return true;
    }

    vector<double> getPixelsAsVector() const
    {
        Timer timer("PGMImage::getPixelsAsVector", pixels.size(), "pixels");

        vector<double> result(pixels.size());
        for (size_t i = 0; i < pixels.size(); i++)
        {
            result[i] = static_cast<double>(pixels[i]) / maxVal; // Normalize to [0, 1]
        }
        return result;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Generate a PGM image from the vector
    static void saveFromVector(const vector<double> &data, int width, int height, const string &filename)
    {
        Timer timer("PGMImage::saveFromVector", data.size(), "pixels");

        ofstream file(filename, ios::binary);
        if (!file)
        {
            cerr << "Failed to open " << filename << " for writing" << endl;
            return;
        }

        file << "P5\n"
             << width << " " << height << "\n255\n";

        vector<unsigned char> pixels(data.size());
        for (size_t i = 0; i < data.size(); i++)
        {
            // Clamp and scale back to [0, 255]
            double val = data[i] * 255.0;
            if (val < 0)
                val = 0;
            if (val > 255)
                val = 255;
            pixels[i] = static_cast<unsigned char>(val);
        }

        file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
    }
};

// Structure to hold image data with metadata
struct ImageData
{
    string filename;
    string personId;
    vector<double> pixels;
};

// Structure to hold the LDA model
struct LDAModel
{
    int imgWidth;
    int imgHeight;
    vector<double> globalMean;
    vector<vector<double>> projectionMatrix; // LDA projection matrix (Fisher faces)
    vector<double> eigenvalues;              // Eigenvalues corresponding to projection vectors
    map<string, vector<double>> classMeans;  // Mean face for each class/person
    vector<vector<double>> projectedTrainingData;
    vector<string> trainingPersonIds; // Person IDs corresponding to projectedTrainingData

    // Save the model to a file
    bool saveToFile(const string &filename) const
    {
        Timer timer("LDAModel::saveToFile");

        ofstream file(filename, ios::binary);
        if (!file)
        {
            cerr << "Failed to open " << filename << " for writing" << endl;
            return false;
        }

        // Write dimensions
        file.write(reinterpret_cast<const char *>(&imgWidth), sizeof(imgWidth));
        file.write(reinterpret_cast<const char *>(&imgHeight), sizeof(imgHeight));

        // Write global mean
        size_t meanSize = globalMean.size();
        file.write(reinterpret_cast<const char *>(&meanSize), sizeof(meanSize));
        file.write(reinterpret_cast<const char *>(globalMean.data()), meanSize * sizeof(double));

        // Write projection matrix
        size_t numProjections = projectionMatrix.size();
        file.write(reinterpret_cast<const char *>(&numProjections), sizeof(numProjections));

        if (numProjections > 0)
        {
            size_t projectionSize = projectionMatrix[0].size();
            file.write(reinterpret_cast<const char *>(&projectionSize), sizeof(projectionSize));

            for (const auto &vector : projectionMatrix)
            {
                file.write(reinterpret_cast<const char *>(vector.data()), projectionSize * sizeof(double));
            }
        }

        // Write eigenvalues
        size_t eigenvaluesSize = eigenvalues.size();
        file.write(reinterpret_cast<const char *>(&eigenvaluesSize), sizeof(eigenvaluesSize));
        file.write(reinterpret_cast<const char *>(eigenvalues.data()), eigenvaluesSize * sizeof(double));

        // Write class means
        size_t numClasses = classMeans.size();
        file.write(reinterpret_cast<const char *>(&numClasses), sizeof(numClasses));

        for (const auto &[personId, mean] : classMeans)
        {
            // Write personId
            size_t idLength = personId.length();
            file.write(reinterpret_cast<const char *>(&idLength), sizeof(idLength));
            file.write(personId.c_str(), idLength);

            // Write mean vector
            file.write(reinterpret_cast<const char *>(mean.data()), mean.size() * sizeof(double));
        }

        // Write projected training data
        size_t numTrainingSamples = projectedTrainingData.size();
        file.write(reinterpret_cast<const char *>(&numTrainingSamples), sizeof(numTrainingSamples));

        if (numTrainingSamples > 0)
        {
            size_t projectionSize = projectedTrainingData[0].size();
            file.write(reinterpret_cast<const char *>(&projectionSize), sizeof(projectionSize));

            for (const auto &projection : projectedTrainingData)
            {
                file.write(reinterpret_cast<const char *>(projection.data()), projectionSize * sizeof(double));
            }
        }

        // Write person IDs
        size_t numPersonIds = trainingPersonIds.size();
        file.write(reinterpret_cast<const char *>(&numPersonIds), sizeof(numPersonIds));

        for (const auto &personId : trainingPersonIds)
        {
            size_t idLength = personId.length();
            file.write(reinterpret_cast<const char *>(&idLength), sizeof(idLength));
            file.write(personId.c_str(), idLength);
        }

        return true;
    }

    // Load the model from a file
    bool loadFromFile(const string &filename)
    {
        Timer timer("LDAModel::loadFromFile");

        ifstream file(filename, ios::binary);
        if (!file)
        {
            cerr << "Failed to open " << filename << " for reading" << endl;
            return false;
        }

        // Read dimensions
        file.read(reinterpret_cast<char *>(&imgWidth), sizeof(imgWidth));
        file.read(reinterpret_cast<char *>(&imgHeight), sizeof(imgHeight));

        // Read global mean
        size_t meanSize;
        file.read(reinterpret_cast<char *>(&meanSize), sizeof(meanSize));
        globalMean.resize(meanSize);
        file.read(reinterpret_cast<char *>(globalMean.data()), meanSize * sizeof(double));

        // Read projection matrix
        size_t numProjections;
        file.read(reinterpret_cast<char *>(&numProjections), sizeof(numProjections));
        projectionMatrix.resize(numProjections);

        if (numProjections > 0)
        {
            size_t projectionSize;
            file.read(reinterpret_cast<char *>(&projectionSize), sizeof(projectionSize));

            for (auto &vector : projectionMatrix)
            {
                vector.resize(projectionSize);
                file.read(reinterpret_cast<char *>(vector.data()), projectionSize * sizeof(double));
            }
        }

        // Read eigenvalues
        size_t eigenvaluesSize;
        file.read(reinterpret_cast<char *>(&eigenvaluesSize), sizeof(eigenvaluesSize));
        eigenvalues.resize(eigenvaluesSize);
        file.read(reinterpret_cast<char *>(eigenvalues.data()), eigenvaluesSize * sizeof(double));

        // Read class means
        size_t numClasses;
        file.read(reinterpret_cast<char *>(&numClasses), sizeof(numClasses));

        for (size_t i = 0; i < numClasses; i++)
        {
            // Read personId
            size_t idLength;
            file.read(reinterpret_cast<char *>(&idLength), sizeof(idLength));

            char *buffer = new char[idLength + 1];
            file.read(buffer, idLength);
            buffer[idLength] = '\0';
            string personId = string(buffer);
            delete[] buffer;

            // Read mean vector
            vector<double> mean(meanSize);
            file.read(reinterpret_cast<char *>(mean.data()), meanSize * sizeof(double));

            classMeans[personId] = mean;
        }

        // Read projected training data
        size_t numTrainingSamples;
        file.read(reinterpret_cast<char *>(&numTrainingSamples), sizeof(numTrainingSamples));
        projectedTrainingData.resize(numTrainingSamples);

        if (numTrainingSamples > 0)
        {
            size_t projectionSize;
            file.read(reinterpret_cast<char *>(&projectionSize), sizeof(projectionSize));

            for (auto &projection : projectedTrainingData)
            {
                projection.resize(projectionSize);
                file.read(reinterpret_cast<char *>(projection.data()), projectionSize * sizeof(double));
            }
        }

        // Read person IDs
        size_t numPersonIds;
        file.read(reinterpret_cast<char *>(&numPersonIds), sizeof(numPersonIds));
        trainingPersonIds.resize(numPersonIds);

        for (auto &personId : trainingPersonIds)
        {
            size_t idLength;
            file.read(reinterpret_cast<char *>(&idLength), sizeof(idLength));

            char *buffer = new char[idLength + 1];
            file.read(buffer, idLength);
            buffer[idLength] = '\0';
            personId = string(buffer);
            delete[] buffer;
        }

        return true;
    }
};

// CUDA Kernels
__global__ void transposeKernel(const double *input, double *output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        output[x * rows + y] = input[y * cols + x];
    }
}

__global__ void matrixMultiplyKernel(const double *A, const double *B, double *C,
                                     int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        double sum = 0.0;
        for (int k = 0; k < colsA; k++)
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__global__ void centerDataKernel(double *data, const double *mean,
                                 int numSamples, int featureSize)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int feature = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample < numSamples && feature < featureSize)
    {
        data[sample * featureSize + feature] -= mean[feature];
    }
}

__global__ void distanceKernel(const double *query, const double *training,
                               double *distances, int numTraining, int featureSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numTraining)
    {
        double sum = 0.0;
        for (int i = 0; i < featureSize; i++)
        {
            double diff = query[i] - training[tid * featureSize + i];
            sum += diff * diff;
        }
        distances[tid] = sqrt(sum);
    }
}

__global__ void scatterKernel(const double *centeredData, double *scatterMatrix,
                              int numSamples, int featureSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < featureSize && j < featureSize)
    {
        double sum = 0.0;
        for (int k = 0; k < numSamples; k++)
        {
            sum += centeredData[k * featureSize + i] * centeredData[k * featureSize + j];
        }
        scatterMatrix[i * featureSize + j] = sum;
    }
}

__global__ void projectKernel(const double *data, const double *projectionMatrix,
                              double *projectedData, int numSamples,
                              int featureSize, int numComponents)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int component = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample < numSamples && component < numComponents)
    {
        double sum = 0.0;
        for (int i = 0; i < featureSize; i++)
        {
            sum += data[sample * featureSize + i] * projectionMatrix[component * featureSize + i];
        }
        projectedData[sample * numComponents + component] = sum;
    }
}

// CUDA Wrapper Functions
void cudaTranspose(const thrust::device_vector<double> &input,
                   thrust::device_vector<double> &output,
                   int rows, int cols)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    transposeKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()),
        rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cudaMatrixMultiply(const thrust::device_vector<double> &A,
                        const thrust::device_vector<double> &B,
                        thrust::device_vector<double> &C,
                        int rowsA, int colsA, int colsB)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x,
                  (rowsA + blockSize.y - 1) / blockSize.y);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(A.data()),
        thrust::raw_pointer_cast(B.data()),
        thrust::raw_pointer_cast(C.data()),
        rowsA, colsA, colsB);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cudaCenterData(thrust::device_vector<double> &data,
                    const thrust::device_vector<double> &mean,
                    int numSamples, int featureSize)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((numSamples + blockSize.x - 1) / blockSize.x,
                  (featureSize + blockSize.y - 1) / blockSize.y);

    centerDataKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(mean.data()),
        numSamples, featureSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cudaComputeDistances(const thrust::device_vector<double> &query,
                          const thrust::device_vector<double> &training,
                          thrust::device_vector<double> &distances,
                          int numTraining, int featureSize)
{
    int blockSize = 256;
    int gridSize = (numTraining + blockSize - 1) / blockSize;

    distanceKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(training.data()),
        thrust::raw_pointer_cast(distances.data()),
        numTraining, featureSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cudaComputeScatterMatrix(const thrust::device_vector<double> &centeredData,
                              thrust::device_vector<double> &scatterMatrix,
                              int numSamples, int featureSize)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((featureSize + blockSize.x - 1) / blockSize.x,
                  (featureSize + blockSize.y - 1) / blockSize.y);

    scatterKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(centeredData.data()),
        thrust::raw_pointer_cast(scatterMatrix.data()),
        numSamples, featureSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cudaProjectData(const thrust::device_vector<double> &data,
                     const thrust::device_vector<double> &projectionMatrix,
                     thrust::device_vector<double> &projectedData,
                     int numSamples, int featureSize, int numComponents)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((numSamples + blockSize.x - 1) / blockSize.x,
                  (numComponents + blockSize.y - 1) / blockSize.y);

    projectKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(projectionMatrix.data()),
        thrust::raw_pointer_cast(projectedData.data()),
        numSamples, featureSize, numComponents);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// CUDA-Accelerated LDA Functions
vector<double> cudaComputeMean(const vector<vector<double>> &data)
{
    Timer timer("cudaComputeMean", data.size() * data[0].size(), "elements");

    int numSamples = data.size();
    int featureSize = data[0].size();

    // Flatten data and copy to device
    thrust::host_vector<double> h_data(numSamples * featureSize);
    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            h_data[i * featureSize + j] = data[i][j];
        }
    }

    thrust::device_vector<double> d_data = h_data;
    thrust::device_vector<double> d_mean(featureSize, 0.0);

    // Get raw pointer to device data
    double *raw_ptr = thrust::raw_pointer_cast(d_data.data());

    // Compute mean using thrust::transform_reduce
    for (int j = 0; j < featureSize; j++)
    {
        const int feature_idx = j;      // Create a local copy for the lambda
        const int stride = featureSize; // Create a local copy for the lambda

        auto counting = thrust::make_counting_iterator(0);
        thrust::transform_reduce(
            thrust::device,
            counting,
            counting + numSamples,
            [=] __device__(int idx) -> double
            {
                return raw_ptr[idx * stride + feature_idx];
            },
            0.0,
            thrust::plus<double>());

        // Store the mean
        d_mean[j] /= numSamples;
    }

    // Copy back to host
    thrust::host_vector<double> h_mean = d_mean;
    return vector<double>(h_mean.begin(), h_mean.end());
}

vector<vector<double>> cudaCenterData(const vector<vector<double>> &data,
                                      const vector<double> &mean)
{
    Timer timer("cudaCenterData", data.size() * data[0].size(), "elements");

    int numSamples = data.size();
    int featureSize = data[0].size();

    // Flatten data and copy to device
    thrust::host_vector<double> h_data(numSamples * featureSize);
    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            h_data[i * featureSize + j] = data[i][j];
        }
    }

    thrust::device_vector<double> d_data = h_data;
    thrust::device_vector<double> d_mean(mean.begin(), mean.end());

    // Center the data on GPU
    cudaCenterData(d_data, d_mean, numSamples, featureSize);

    // Copy back to host and reshape
    thrust::host_vector<double> h_centered = d_data;
    vector<vector<double>> centeredData(numSamples, vector<double>(featureSize));

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            centeredData[i][j] = h_centered[i * featureSize + j];
        }
    }

    return centeredData;
}

vector<vector<double>> cudaMatrixMultiply(const vector<vector<double>> &A,
                                          const vector<vector<double>> &B)
{
    Timer timer("cudaMatrixMultiply", A.size() * B[0].size() * B.size(), "operations");

    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

    // Flatten matrices and copy to device
    thrust::host_vector<double> h_A(rowsA * colsA);
    thrust::host_vector<double> h_B(colsA * colsB);
    thrust::host_vector<double> h_C(rowsA * colsB);

    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsA; j++)
        {
            h_A[i * colsA + j] = A[i][j];
        }
    }

    for (int i = 0; i < colsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            h_B[i * colsB + j] = B[i][j];
        }
    }

    thrust::device_vector<double> d_A = h_A;
    thrust::device_vector<double> d_B = h_B;
    thrust::device_vector<double> d_C(rowsA * colsB);

    // Multiply on GPU
    cudaMatrixMultiply(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Copy back to host and reshape
    h_C = d_C;
    vector<vector<double>> C(rowsA, vector<double>(colsB));

    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            C[i][j] = h_C[i * colsB + j];
        }
    }

    return C;
}

vector<vector<double>> cudaComputeWithinClassScatter(
    const map<string, vector<vector<double>>> &classSamples,
    int featureSize)
{
    Timer timer("cudaComputeWithinClassScatter");

    // Initialize SW matrix with zeros
    thrust::device_vector<double> d_SW(featureSize * featureSize, 0.0);

    // For each class
    for (const auto &[classId, samples] : classSamples)
    {
        // Calculate class mean
        vector<double> classMean = cudaComputeMean(samples);

        // Center the class samples
        vector<vector<double>> centeredSamples = cudaCenterData(samples, classMean);

        // Flatten centered samples and copy to device
        thrust::host_vector<double> h_centered(centeredSamples.size() * featureSize);
        for (size_t i = 0; i < centeredSamples.size(); i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                h_centered[i * featureSize + j] = centeredSamples[i][j];
            }
        }

        thrust::device_vector<double> d_centered = h_centered;
        thrust::device_vector<double> d_classSW(featureSize * featureSize, 0.0);

        // Compute scatter for this class on GPU
        cudaComputeScatterMatrix(d_centered, d_classSW, centeredSamples.size(), featureSize);

        // Add to total SW
        thrust::transform(
            d_SW.begin(), d_SW.end(),
            d_classSW.begin(),
            d_SW.begin(),
            thrust::plus<double>());
    }

    // Copy back to host and reshape
    thrust::host_vector<double> h_SW = d_SW;
    vector<vector<double>> SW(featureSize, vector<double>(featureSize));

    for (int i = 0; i < featureSize; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            SW[i][j] = h_SW[i * featureSize + j];
        }
    }

    return SW;
}

vector<vector<double>> cudaProjectData(const vector<vector<double>> &centeredData,
                                       const vector<vector<double>> &projectionMatrix)
{
    Timer timer("cudaProjectData",
                centeredData.size() * centeredData[0].size() * projectionMatrix.size(),
                "operations");

    int numSamples = centeredData.size();
    int featureSize = centeredData[0].size();
    int numComponents = projectionMatrix.size();

    // Flatten data and copy to device
    thrust::host_vector<double> h_data(numSamples * featureSize);
    thrust::host_vector<double> h_projection(numComponents * featureSize);

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            h_data[i * featureSize + j] = centeredData[i][j];
        }
    }

    for (int i = 0; i < numComponents; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            h_projection[i * featureSize + j] = projectionMatrix[i][j];
        }
    }

    thrust::device_vector<double> d_data = h_data;
    thrust::device_vector<double> d_projection = h_projection;
    thrust::device_vector<double> d_projected(numSamples * numComponents);

    // Project on GPU
    cudaProjectData(d_data, d_projection, d_projected, numSamples, featureSize, numComponents);

    // Copy back to host and reshape
    thrust::host_vector<double> h_projected = d_projected;
    vector<vector<double>> projectedData(numSamples, vector<double>(numComponents));

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < numComponents; j++)
        {
            projectedData[i][j] = h_projected[i * numComponents + j];
        }
    }

    return projectedData;
}

int cudaFindClosestMatch(const vector<double> &queryFeatures,
                         const vector<vector<double>> &trainingFeatures)
{
    Timer timer("cudaDistanceComputations", trainingFeatures.size() * queryFeatures.size(), "operations");

    int numTraining = trainingFeatures.size();
    int featureSize = queryFeatures.size();

    // Flatten training features and copy to device
    thrust::host_vector<double> h_training(numTraining * featureSize);
    for (int i = 0; i < numTraining; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            h_training[i * featureSize + j] = trainingFeatures[i][j];
        }
    }

    thrust::device_vector<double> d_query(queryFeatures.begin(), queryFeatures.end());
    thrust::device_vector<double> d_training = h_training;
    thrust::device_vector<double> d_distances(numTraining);

    // Compute distances on GPU
    cudaComputeDistances(d_query, d_training, d_distances, numTraining, featureSize);

    // Find minimum distance on GPU using thrust
    auto min_iter = thrust::min_element(
        thrust::device, // Explicit execution policy
        d_distances.begin(),
        d_distances.end());
    int closestIndex = min_iter - d_distances.begin();

    return closestIndex;
}

// Function to compute the between-class scatter matrix
vector<vector<double>> computeBetweenClassScatter(
    const map<string, vector<double>> &classMeans,
    const vector<double> &globalMean,
    const map<string, int> &classCounts,
    int featureSize)
{
    Timer timer("computeBetweenClassScatter");

    // Initialize SB matrix with zeros
    vector<vector<double>> SB(featureSize, vector<double>(featureSize, 0.0));

    // For each class
    for (const auto &[classId, classMean] : classMeans)
    {
        int numSamples = classCounts.at(classId);

        // Compute (µc - µ)
        vector<double> diff(featureSize);
        for (int i = 0; i < featureSize; i++)
        {
            diff[i] = classMean[i] - globalMean[i];
        }

        // Update SB += N_c * (µc - µ)(µc - µ)^T
        for (int i = 0; i < featureSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                SB[i][j] += numSamples * diff[i] * diff[j];
            }
        }
    }

    return SB;
}

// Function to project a single image
vector<double> projectImage(const vector<double> &image, const vector<double> &globalMean,
                            const vector<vector<double>> &projectionMatrix)
{
    Timer timer("projectImage", image.size() * projectionMatrix.size(), "operations");

    int numComponents = projectionMatrix.size();
    int featureSize = image.size();

    // Center the image
    vector<double> centeredImage(featureSize);
    for (int i = 0; i < featureSize; i++)
    {
        centeredImage[i] = image[i] - globalMean[i];
    }

    // Project onto projection vectors
    vector<double> projection(numComponents);
    for (int i = 0; i < numComponents; i++)
    {
        projection[i] = 0.0;
        for (int j = 0; j < featureSize; j++)
        {
            projection[i] += centeredImage[j] * projectionMatrix[i][j];
        }
    }

    return projection;
}

// Function to reconstruct data from projection
vector<vector<double>> reconstructData(const vector<vector<double>> &projectedData,
                                       const vector<vector<double>> &projectionMatrix,
                                       const vector<double> &globalMean)
{
    Timer timer("reconstructData",
                projectedData.size() * projectedData[0].size() * projectionMatrix[0].size(),
                "operations");

    int numSamples = projectedData.size();
    int numComponents = projectionMatrix.size();
    int featureSize = projectionMatrix[0].size();

    vector<vector<double>> reconstructedData(numSamples, vector<double>(featureSize, 0.0));

    // First add the projection of each image
    for (int i = 0; i < numSamples; i++)
    {
        // Start with zeros
        for (int j = 0; j < featureSize; j++)
        {
            reconstructedData[i][j] = 0.0;
        }

        // Add the contribution of each projection vector
        for (int j = 0; j < numComponents; j++)
        {
            for (int k = 0; k < featureSize; k++)
            {
                reconstructedData[i][k] += projectedData[i][j] * projectionMatrix[j][k];
            }
        }

        // Add the mean back
        for (int j = 0; j < featureSize; j++)
        {
            reconstructedData[i][j] += globalMean[j];
        }
    }

    return reconstructedData;
}

// Function to reconstruct a single image from its projection
vector<double> reconstructImage(const vector<double> &projection,
                                const vector<vector<double>> &projectionMatrix,
                                const vector<double> &globalMean)
{
    Timer timer("reconstructImage", projection.size() * projectionMatrix[0].size(), "operations");

    int featureSize = projectionMatrix[0].size();
    int numComponents = projection.size();

    vector<double> reconstructed(featureSize, 0.0);

    // Reconstruct from projection
    for (int i = 0; i < numComponents; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            reconstructed[j] += projection[i] * projectionMatrix[i][j];
        }
    }

    // Add the mean back
    for (int j = 0; j < featureSize; j++)
    {
        reconstructed[j] += globalMean[j];
    }

    return reconstructed;
}

// Function to load all PGM images from a hierarchical dataset
vector<ImageData> loadHierarchicalDataset(const string &rootDir, int &imgWidth, int &imgHeight)
{
    Timer timer("loadHierarchicalDataset");

    vector<ImageData> imageData;
    imgWidth = imgHeight = 0;

    try
    {
        // Iterate through person folders
        for (const auto &personDir : fs::directory_iterator(rootDir))
        {
            if (!fs::is_directory(personDir))
                continue;

            string personId = personDir.path().filename().string();

            // Iterate through images in each person folder
            for (const auto &entry : fs::directory_iterator(personDir))
            {
                if (entry.path().extension() == ".pgm")
                {
                    string filename = entry.path().filename().string();
                    PGMImage img;

                    if (img.load(entry.path().string()))
                    {
                        // Set dimensions from the first image
                        if (imgWidth == 0)
                        {
                            imgWidth = img.getWidth();
                            imgHeight = img.getHeight();
                        }
                        else
                        {
                            // Check that all images have the same dimensions
                            if (img.getWidth() != imgWidth || img.getHeight() != imgHeight)
                            {
                                cerr << "Warning: Image " << entry.path().string() << " has different dimensions. Skipping." << endl;
                                continue;
                            }
                        }

                        ImageData imgData;
                        imgData.filename = entry.path().string();
                        imgData.personId = personId;
                        imgData.pixels = img.getPixelsAsVector();

                        imageData.push_back(imgData);
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        cerr << "Error accessing directory: " << e.what() << endl;
    }

    return imageData;
}

// Function to split data into training and testing sets
pair<vector<ImageData>, vector<ImageData>> splitTrainTest(const vector<ImageData> &allData, int imagesPerPerson, int trainingImagesPerPerson)
{
    Timer timer("splitTrainTest");

    vector<ImageData> trainingData;
    vector<ImageData> testingData;

    map<string, vector<ImageData>> personData;

    // Group images by person
    for (const auto &img : allData)
    {
        personData[img.personId].push_back(img);
    }

    // Split each person's images into training and testing
    for (auto &[personId, images] : personData)
    {
        // Ensure we have exactly imagesPerPerson images per person
        if (images.size() < imagesPerPerson)
        {
            cerr << "Warning: Person " << personId << " has fewer than " << imagesPerPerson << " images." << endl;
            continue;
        }

        // Use only the first imagesPerPerson images
        for (int i = 0; i < imagesPerPerson; i++)
        {
            if (i < trainingImagesPerPerson)
            {
                trainingData.push_back(images[i]);
            }
            else
            {
                testingData.push_back(images[i]);
            }
        }
    }

    return make_pair(trainingData, testingData);
}

// Compute the covariance matrix for PCA
vector<vector<double>> computeCovarianceMatrix(const vector<vector<double>> &centeredData)
{
    Timer timer("computeCovarianceMatrix");

    int n = centeredData.size();
    int d = centeredData[0].size();

    vector<vector<double>> cov(d, vector<double>(d, 0.0));

    // For high-dimensional data, it's more efficient to compute X * X^T
    // where X is the centered data matrix
    for (int i = 0; i < d; i++)
    {
        for (int j = i; j < d; j++)
        { // Only compute upper triangle
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += centeredData[k][i] * centeredData[k][j];
            }
            cov[i][j] = sum / (n - 1);
            if (i != j)
            {
                cov[j][i] = cov[i][j]; // Fill in lower triangle by symmetry
            }
        }
    }

    return cov;
}

// Function for power iteration to find dominant eigenvector/value pair
pair<vector<double>, double> powerIteration(const vector<vector<double>> &A, int maxIters = 100, double tol = 1e-10)
{
    Timer timer("powerIteration");

    int n = A.size();

    // Initialize random vector
    mt19937 rng(42); // Fixed seed for reproducibility
    normal_distribution<double> dist(0.0, 1.0);

    vector<double> v(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = dist(rng);
    }

    // Normalize
    double norm = 0.0;
    for (int i = 0; i < n; i++)
        norm += v[i] * v[i];
    norm = sqrt(norm);

    if (norm < 1e-10)
    {
        // All zeros, return identity vector
        v[0] = 1.0;
        for (int i = 1; i < n; i++)
            v[i] = 0.0;
        return {v, 0.0};
    }

    for (int i = 0; i < n; i++)
        v[i] /= norm;

    double lambda = 0.0;

    // Power iteration loop with max iterations limit
    for (int iter = 0; iter < maxIters; iter++)
    {
        // Save previous v for convergence check
        vector<double> prev_v = v;

        // Calculate Av
        vector<double> Av(n, 0.0);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Av[i] += A[i][j] * v[j];
            }
        }

        // Find new lambda (Rayleigh quotient)
        lambda = 0.0;
        for (int i = 0; i < n; i++)
        {
            lambda += v[i] * Av[i];
        }

        // Calculate new v = Av/||Av||
        norm = 0.0;
        for (int i = 0; i < n; i++)
            norm += Av[i] * Av[i];
        norm = sqrt(norm);

        if (norm < 1e-10)
        {
            // Matrix is numerically singular or 0
            break;
        }

        for (int i = 0; i < n; i++)
            v[i] = Av[i] / norm;

        // Check for convergence
        double diff = 0.0;
        for (int i = 0; i < n; i++)
            diff += fabs(v[i] - prev_v[i]);

        if (diff < tol)
            break;
    }

    return {v, lambda};
}

// Function to compute eigenvectors/values using power iteration with deflation
pair<vector<vector<double>>, vector<double>> computeEigenvectors(
    const vector<vector<double>> &A, int k, bool verbose = false)
{
    Timer timer("computeEigenvectors");

    int n = A.size();
    vector<vector<double>> eigenvectors(k, vector<double>(n));
    vector<double> eigenvalues(k);

    // Make a copy of A to deflate
    vector<vector<double>> B = A;

    for (int i = 0; i < k; i++)
    {
        if (verbose)
        {
            cout << "Computing eigenvector " << (i + 1) << " of " << k << endl;
        }

        // Find dominant eigenvector/value of current B
        auto [v, lambda] = powerIteration(B);

        // Store eigenvector and eigenvalue
        eigenvalues[i] = lambda;
        eigenvectors[i] = v;

        // Deflate B = B - lambda * v * v^T
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                B[r][c] -= lambda * v[r] * v[c];
            }
        }
    }

    return {eigenvectors, eigenvalues};
}

// Compute small covariance matrix for PCA preprocessing
vector<vector<double>> computeSmallCovariance(const vector<vector<double>> &centeredData)
{
    Timer timer("computeSmallCovariance");

    int n = centeredData.size();

    // Compute X^T * X for the n x n matrix (faster than computing X * X^T for high dimensions)
    vector<vector<double>> smallCov(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        { // Only compute upper triangle
            double sum = 0.0;
            for (size_t k = 0; k < centeredData[0].size(); k++)
            {
                sum += centeredData[i][k] * centeredData[j][k];
            }
            smallCov[i][j] = sum / (n - 1);
            if (i != j)
            {
                smallCov[j][i] = smallCov[i][j]; // Fill lower triangle by symmetry
            }
        }
    }

    return smallCov;
}

// Convert small eigenvectors to full eigenvectors
vector<vector<double>> convertToFullEigenvectors(
    const vector<vector<double>> &smallEigenvectors,
    const vector<vector<double>> &centeredData)
{
    Timer timer("convertToFullEigenvectors");

    int numEigenvectors = smallEigenvectors.size();
    int numSamples = centeredData.size();
    int featureSize = centeredData[0].size();

    vector<vector<double>> fullEigenvectors(numEigenvectors, vector<double>(featureSize, 0.0));

    // V_full = X^T * V_small (where X is centered data)
    for (int i = 0; i < numEigenvectors; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            for (int k = 0; k < numSamples; k++)
            {
                fullEigenvectors[i][j] += centeredData[k][j] * smallEigenvectors[i][k];
            }
        }

        // Normalize
        double norm = 0.0;
        for (int j = 0; j < featureSize; j++)
        {
            norm += fullEigenvectors[i][j] * fullEigenvectors[i][j];
        }
        norm = sqrt(norm);

        for (int j = 0; j < featureSize; j++)
        {
            fullEigenvectors[i][j] /= norm;
        }
    }

    return fullEigenvectors;
}

// Improved PCA as a dimensionality reduction preprocessing step for LDA
pair<vector<vector<double>>, vector<double>> applyPCA(
    const vector<vector<double>> &centeredData, int numComponents)
{
    Timer timer("applyPCA");

    int numSamples = centeredData.size();
    int featureSize = centeredData[0].size();

    // Compute small covariance matrix for efficiency (n x n instead of d x d)
    vector<vector<double>> smallCov = computeSmallCovariance(centeredData);

    // Find eigenvectors of the small covariance matrix
    auto [smallEigenvectors, eigenvalues] = computeEigenvectors(smallCov, numComponents);

    // Convert small eigenvectors to full eigenvectors
    vector<vector<double>> eigenvectors = convertToFullEigenvectors(smallEigenvectors, centeredData);

    return {eigenvectors, eigenvalues};
}

// Improved function to find the LDA projection matrix
vector<vector<double>> findLDAProjection(
    const vector<vector<double>> &SW,
    const vector<vector<double>> &SB,
    int numComponents)
{
    Timer timer("findLDAProjection");

    int featureSize = SW.size();

    // Make a copy of SW and regularize it to ensure it's invertible
    vector<vector<double>> SWreg = SW;

    // Compute trace of SW for regularization
    double trace = 0.0;
    for (int i = 0; i < featureSize; i++)
    {
        trace += SWreg[i][i];
    }

    // Add small regularization to diagonal
    double epsilon = trace * 0.001 / featureSize;
    for (int i = 0; i < featureSize; i++)
    {
        SWreg[i][i] += epsilon;
    }

    // Compute SW^-1 * SB directly for each column using Cholesky decomposition
    // This is a placeholder - for a full implementation, we would use a proper
    // linear algebra library or implement Cholesky decomposition ourselves

    // For now, let's use a simpler approach - direct inversion is slow but works for small matrices
    // In practice, this would be a generalized eigenvalue solver

    // Just use the SB as approximate projection vectors, then orthogonalize
    vector<vector<double>> projection(numComponents, vector<double>(featureSize));

    // Construct projection vectors from first numComponents columns of SB
    for (int i = 0; i < numComponents; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            projection[i][j] = SB[j][i % featureSize] * (i < featureSize ? 1.0 : 0.5);
        }

        // Orthogonalize against previous vectors
        for (int j = 0; j < i; j++)
        {
            double dot = 0.0;
            for (int k = 0; k < featureSize; k++)
            {
                dot += projection[i][k] * projection[j][k];
            }

            for (int k = 0; k < featureSize; k++)
            {
                projection[i][k] -= dot * projection[j][k];
            }
        }

        // Normalize
        double norm = 0.0;
        for (int j = 0; j < featureSize; j++)
        {
            norm += projection[i][j] * projection[i][j];
        }
        norm = sqrt(norm);

        if (norm > 1e-10)
        { // Avoid division by near-zero
            for (int j = 0; j < featureSize; j++)
            {
                projection[i][j] /= norm;
            }
        }
        else
        {
            // Fill with random orthogonal vector as fallback
            mt19937 rng(42 + i);
            normal_distribution<double> dist(0.0, 1.0);

            for (int j = 0; j < featureSize; j++)
            {
                projection[i][j] = dist(rng);
            }

            // Orthogonalize and normalize as above
            for (int j = 0; j < i; j++)
            {
                double dot = 0.0;
                for (int k = 0; k < featureSize; k++)
                {
                    dot += projection[i][k] * projection[j][k];
                }

                for (int k = 0; k < featureSize; k++)
                {
                    projection[i][k] -= dot * projection[j][k];
                }
            }

            norm = 0.0;
            for (int j = 0; j < featureSize; j++)
            {
                norm += projection[i][j] * projection[i][j];
            }
            norm = sqrt(norm);

            for (int j = 0; j < featureSize; j++)
            {
                projection[i][j] /= norm;
            }
        }
    }

    return projection;
}

// Function to train the LDA model with robust implementation
LDAModel trainLDAModel(const vector<ImageData> &trainingData, int imgWidth, int imgHeight, int numComponents)
{
    Timer timer("trainLDAModel", trainingData.size() * imgWidth * imgHeight, "pixels processed");

    LDAModel model;
    model.imgWidth = imgWidth;
    model.imgHeight = imgHeight;

    // Extract pixel vectors and person IDs
    vector<vector<double>> imageVectors;
    for (const auto &img : trainingData)
    {
        imageVectors.push_back(img.pixels);
        model.trainingPersonIds.push_back(img.personId);
    }

    // Step 1: Compute the global mean
    model.globalMean = cudaComputeMean(imageVectors);

    // Step 2: Center the data
    vector<vector<double>> centeredData = cudaCenterData(imageVectors, model.globalMean);

    // Step 3: Group data by class (person)
    map<string, vector<vector<double>>> classSamples;
    map<string, int> classCounts;

    for (size_t i = 0; i < trainingData.size(); i++)
    {
        classSamples[trainingData[i].personId].push_back(centeredData[i]);
        classCounts[trainingData[i].personId]++;
    }

    // Step 4: Compute class means
    for (const auto &[classId, samples] : classSamples)
    {
        model.classMeans[classId] = cudaComputeMean(samples);
    }

    // Get feature dimension and number of classes
    int featureSize = model.globalMean.size();
    int numClasses = classSamples.size();

    // Ensure we don't try to extract more components than classes-1
    int maxComponents = min(numComponents, numClasses - 1);
    cout << "Computing " << maxComponents << " LDA components for " << numClasses << " classes..." << endl;

    // Step 5: Dimensionality reduction using PCA as preprocessing
    // For face images, reduce to N-c dimensions first (N = number of samples, c = number of classes)
    int n = trainingData.size();
    int pcaDimensions = min(n - numClasses, featureSize);
    pcaDimensions = min(pcaDimensions, 100); // Limit to reasonable number

    cout << "Performing PCA preprocessing with " << pcaDimensions << " dimensions..." << endl;
    auto [pcaVectors, pcaValues] = applyPCA(centeredData, pcaDimensions);

    // Project data onto PCA subspace
    vector<vector<double>> pcaProjectedData = cudaProjectData(centeredData, pcaVectors);

    // Group PCA-projected data by class
    map<string, vector<vector<double>>> pcaClassSamples;
    for (size_t i = 0; i < trainingData.size(); i++)
    {
        pcaClassSamples[trainingData[i].personId].push_back(pcaProjectedData[i]);
    }

    // Step 6: Compute class means in PCA space
    map<string, vector<double>> pcaClassMeans;
    for (const auto &[classId, samples] : pcaClassSamples)
    {
        pcaClassMeans[classId] = cudaComputeMean(samples);
    }

    // Compute global mean in PCA space
    vector<double> pcaGlobalMean = cudaComputeMean(pcaProjectedData);

    // Step 7: Compute within-class scatter matrix in PCA subspace
    cout << "Computing scatter matrices in PCA space..." << endl;
    vector<vector<double>> SW = cudaComputeWithinClassScatter(pcaClassSamples, pcaDimensions);

    // Step 8: Compute between-class scatter matrix in PCA subspace
    vector<vector<double>> SB = computeBetweenClassScatter(
        pcaClassMeans, pcaGlobalMean, classCounts, pcaDimensions);

    // Step 9: Find LDA projection in the PCA subspace
    cout << "Computing LDA projection vectors..." << endl;
    vector<vector<double>> ldaVectorsInPcaSpace = findLDAProjection(SW, SB, maxComponents);

    // Step 10: Convert LDA vectors back to original space
    cout << "Converting LDA vectors to image space..." << endl;
    vector<vector<double>> fullProjection(maxComponents, vector<double>(featureSize, 0.0));

    for (int i = 0; i < maxComponents; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            for (int k = 0; k < pcaDimensions; k++)
            {
                fullProjection[i][j] += ldaVectorsInPcaSpace[i][k] * pcaVectors[k][j];
            }
        }

        // Normalize
        double norm = 0.0;
        for (int j = 0; j < featureSize; j++)
        {
            norm += fullProjection[i][j] * fullProjection[i][j];
        }
        norm = sqrt(norm);

        for (int j = 0; j < featureSize; j++)
        {
            fullProjection[i][j] /= norm;
        }
    }

    // Store the projection matrix
    model.projectionMatrix = fullProjection;

    // Add placeholder eigenvalues
    model.eigenvalues.resize(maxComponents, 1.0);

    // Step 11: Project training data onto Fisher faces
    cout << "Projecting training data onto LDA space..." << endl;
    model.projectedTrainingData = cudaProjectData(centeredData, model.projectionMatrix);

    return model;
}

// Function to recognize a face using a trained LDA model
string recognizeFace(const LDAModel &model, const vector<double> &faceImage)
{
    Timer timer("recognizeFace");

    // Project the face onto the Fisher faces
    vector<double> projection = projectImage(faceImage, model.globalMean, model.projectionMatrix);

    // Find the closest match
    int closestMatchIndex = cudaFindClosestMatch(projection, model.projectedTrainingData);

    if (closestMatchIndex >= 0 && closestMatchIndex < model.trainingPersonIds.size())
    {
        return model.trainingPersonIds[closestMatchIndex];
    }
    else
    {
        return "unknown";
    }
}

// Function to print benchmark results
void printEssentialBenchmarkResults()
{
    // These are the key operations we want to track for parallel comparison
    const set<string> essentialOps = {
        "loadHierarchicalDataset",
        "PGMImage::load",
        "cudaComputeMean",
        "cudaCenterData",
        "cudaComputeWithinClassScatter",
        "computeBetweenClassScatter",
        "applyPCA",
        "findLDAProjection",
        "cudaProjectData",
        "cudaDistanceComputations",
        "recognizeFace",
        "trainLDAModel"};

    cout << "\nEssential Performance Metrics (for parallel comparison):\n";
    cout << "------------------------------------------------------------\n";
    cout << setw(30) << left << "Operation"
         << setw(15) << right << "Time (ms)"
         << setw(15) << right << "Data Size"
         << setw(10) << right << "Unit" << endl;
    cout << "------------------------------------------------------------\n";

    double totalTime = 0.0;
    for (const auto &result : benchmarkResults)
    {
        if (essentialOps.find(result.operationName) != essentialOps.end())
        {
            double avgTime = result.durationMs / result.count;
            cout << setw(30) << left << result.operationName
                 << setw(15) << right << fixed << setprecision(3) << avgTime
                 << setw(15) << right << result.dataSize
                 << setw(10) << right << result.unit << endl;
            totalTime += result.durationMs;
        }
    }

    cout << "------------------------------------------------------------\n";
    cout << setw(30) << left << "TOTAL TIME"
         << setw(15) << right << fixed << setprecision(3) << totalTime << endl;
    cout << "------------------------------------------------------------\n";
}

int main(int argc, char **argv)
{
    // Initialize CUDA
    cudaFree(0);

    // Clear previous benchmark results
    benchmarkResults.clear();

    // Check for the right number of arguments
    if (argc < 2)
    {
        cerr << "Usage:" << endl;
        cerr << "  To train:   " << argv[0] << " --train <dataset_directory> [num_components] [model_output_file]" << endl;
        cerr << "  To test:    " << argv[0] << " --test <model_file> <test_image.pgm>" << endl;
        cerr << "  To evaluate: " << argv[0] << " --eval <model_file> <dataset_directory>" << endl;
        return 1;
    }

    string mode = argv[1];

    // Training mode
    if (mode == "--train")
    {
        if (argc < 3)
        {
            cerr << "Usage for training: " << argv[0] << " --train <dataset_directory> [num_components] [model_output_file]" << endl;
            return 1;
        }

        string datasetDir = argv[2];
        int numComponents = (argc > 3) ? atoi(argv[3]) : 39;            // Default to 39 components (c-1 for 40 classes)
        string modelFile = (argc > 4) ? argv[4] : "lda_face_model.dat"; // Default model filename

        cout << "Loading dataset from " << datasetDir << "..." << endl;

        // Constants for this dataset
        const int IMAGES_PER_PERSON = 10;
        const int TRAINING_IMAGES_PER_PERSON = 7; // Use 7 for training, 3 for testing

        // Load the hierarchical dataset
        int imgWidth, imgHeight;
        vector<ImageData> allImageData = loadHierarchicalDataset(datasetDir, imgWidth, imgHeight);

        if (allImageData.empty())
        {
            cerr << "No valid PGM images found in the dataset directory." << endl;
            return 1;
        }

        // Split into training and testing sets
        auto [trainingData, testingData] = splitTrainTest(allImageData, IMAGES_PER_PERSON, TRAINING_IMAGES_PER_PERSON);

        cout << "Dataset: " << allImageData.size() << " total images" << endl;
        cout << "Split: " << trainingData.size() << " training images, " << testingData.size() << " testing images" << endl;
        cout << "Image size: " << imgWidth << "x" << imgHeight << " pixels" << endl;

        if (trainingData.empty())
        {
            cerr << "Error: Training set is empty after splitting." << endl;
            return 1;
        }

        cout << "Training LDA model with " << numComponents << " components..." << endl;

        // Train the LDA model
        LDAModel model = trainLDAModel(trainingData, imgWidth, imgHeight, numComponents);

        // Save the model
        if (model.saveToFile(modelFile))
        {
            cout << "Trained model saved to " << modelFile << endl;
        }
        else
        {
            cerr << "Failed to save model to " << modelFile << endl;
            return 1;
        }

        // Save the global mean face as an image
        PGMImage::saveFromVector(model.globalMean, imgWidth, imgHeight, "global_mean_face.pgm");
        cout << "Saved global mean face to 'global_mean_face.pgm'" << endl;

        // Save the Fisher faces as images
        fs::create_directory("fisherfaces");
        for (int i = 0; i < min(10, (int)model.projectionMatrix.size()); i++)
        {
            // Normalize the Fisher face for visualization
            vector<double> fisherface = model.projectionMatrix[i];
            double minVal = *min_element(fisherface.begin(), fisherface.end());
            double maxVal = *max_element(fisherface.begin(), fisherface.end());
            double range = maxVal - minVal;

            if (range > 0)
            {
                for (auto &val : fisherface)
                {
                    val = (val - minVal) / range;
                }
            }

            PGMImage::saveFromVector(fisherface, imgWidth, imgHeight, "fisherfaces/fisherface_" + to_string(i) + ".pgm");
        }
        cout << "Saved top Fisher faces to 'fisherfaces/' directory" << endl;

        // Evaluate on test set
        if (!testingData.empty())
        {
            int correctCount = 0;
            fs::create_directory("lda_reconstructions");

            cout << "\nTesting face recognition on " << testingData.size() << " images..." << endl;

            for (const auto &testImg : testingData)
            {
                // Recognize the face
                string predictedPerson = recognizeFace(model, testImg.pixels);

                // Check if the prediction is correct
                bool correct = (predictedPerson == testImg.personId);
                if (correct)
                    correctCount++;

                // Reconstruct the test image using Fisher faces
                vector<double> projection = projectImage(testImg.pixels, model.globalMean, model.projectionMatrix);
                vector<double> reconstructedImage = reconstructImage(projection, model.projectionMatrix, model.globalMean);

                // Extract filename from the full path
                string baseFilename = fs::path(testImg.filename).filename().string();
                string result = correct ? "correct" : "incorrect";

                // Save reconstructed test image
                PGMImage::saveFromVector(reconstructedImage, imgWidth, imgHeight,
                                         "lda_reconstructions/" + baseFilename + "_" + result + ".pgm");
            }

            // Calculate and print accuracy
            double accuracy = static_cast<double>(correctCount) / testingData.size() * 100.0;
            cout << "Recognition accuracy: " << fixed << setprecision(2) << accuracy << "% ("
                 << correctCount << "/" << testingData.size() << ")" << endl;
            cout << "Reconstructed test images saved to 'lda_reconstructions/' directory" << endl;
        }
    }
    // Testing mode (single image)
    else if (mode == "--test")
    {
        if (argc < 4)
        {
            cerr << "Usage for testing: " << argv[0] << " --test <model_file> <test_image.pgm>" << endl;
            return 1;
        }

        string modelFile = argv[2];
        string testImagePath = argv[3];

        // Load the model
        LDAModel model;
        if (!model.loadFromFile(modelFile))
        {
            cerr << "Failed to load model from " << modelFile << endl;
            return 1;
        }

        cout << "Loaded LDA model with " << model.projectionMatrix.size() << " projection vectors" << endl;

        // Load the test image
        PGMImage testImage;
        if (!testImage.load(testImagePath))
        {
            cerr << "Failed to load test image: " << testImagePath << endl;
            return 1;
        }

        // Check image dimensions
        if (testImage.getWidth() != model.imgWidth || testImage.getHeight() != model.imgHeight)
        {
            cerr << "Test image dimensions (" << testImage.getWidth() << "x" << testImage.getHeight()
                 << ") do not match model dimensions (" << model.imgWidth << "x" << model.imgHeight << ")" << endl;
            return 1;
        }

        // Extract pixel vector
        vector<double> testImageVector = testImage.getPixelsAsVector();

        // Recognize the face
        string personId = recognizeFace(model, testImageVector);
        cout << "Test image identified as: " << personId << endl;

        // Reconstruct the image using LDA
        vector<double> projection = projectImage(testImageVector, model.globalMean, model.projectionMatrix);
        vector<double> reconstructedImage = reconstructImage(projection, model.projectionMatrix, model.globalMean);

        // Save the reconstructed image
        PGMImage::saveFromVector(reconstructedImage, model.imgWidth, model.imgHeight, "lda_reconstructed_test.pgm");
        cout << "Saved reconstructed test image to 'lda_reconstructed_test.pgm'" << endl;
    }
    // Evaluation mode (entire dataset)
    else if (mode == "--eval")
    {
        if (argc < 4)
        {
            cerr << "Usage for evaluation: " << argv[0] << " --eval <model_file> <dataset_directory>" << endl;
            return 1;
        }

        string modelFile = argv[2];
        string datasetDir = argv[3];

        // Load the model
        LDAModel model;
        if (!model.loadFromFile(modelFile))
        {
            cerr << "Failed to load model from " << modelFile << endl;
            return 1;
        }

        cout << "Loaded LDA model with " << model.projectionMatrix.size() << " projection vectors" << endl;

        // Load the dataset for evaluation
        int imgWidth, imgHeight;
        vector<ImageData> allImageData = loadHierarchicalDataset(datasetDir, imgWidth, imgHeight);

        if (allImageData.empty())
        {
            cerr << "No valid PGM images found in the dataset directory." << endl;
            return 1;
        }

        // Check image dimensions
        if (imgWidth != model.imgWidth || imgHeight != model.imgHeight)
        {
            cerr << "Dataset image dimensions (" << imgWidth << "x" << imgHeight
                 << ") do not match model dimensions (" << model.imgWidth << "x" << model.imgHeight << ")" << endl;
            return 1;
        }

        // Extract the test image paths
        const int IMAGES_PER_PERSON = 10;
        const int TRAINING_IMAGES_PER_PERSON = 7; // We'll evaluate on the remaining 3 images per person
        auto [trainingData, testingData] = splitTrainTest(allImageData, IMAGES_PER_PERSON, TRAINING_IMAGES_PER_PERSON);

        if (testingData.empty())
        {
            cerr << "Error: Testing set is empty after splitting." << endl;
            return 1;
        }

        cout << "Evaluating on " << testingData.size() << " test images..." << endl;

        // Evaluate on the test set
        int correctCount = 0;
        fs::create_directory("lda_reconstructions");

        // Create a confusion matrix
        map<string, map<string, int>> confusionMatrix;
        set<string> allPersonIds;

        for (const auto &img : allImageData)
        {
            allPersonIds.insert(img.personId);
        }

        // Initialize confusion matrix
        for (const auto &trueId : allPersonIds)
        {
            for (const auto &predId : allPersonIds)
            {
                confusionMatrix[trueId][predId] = 0;
            }
            confusionMatrix[trueId]["unknown"] = 0;
        }

        // Process each test image
        for (const auto &testImg : testingData)
        {
            // Recognize the face
            string predictedPerson = recognizeFace(model, testImg.pixels);

            // Update confusion matrix
            confusionMatrix[testImg.personId][predictedPerson]++;

            // Check if prediction is correct
            bool correct = (predictedPerson == testImg.personId);
            if (correct)
                correctCount++;

            // Reconstruct the test image
            vector<double> projection = projectImage(testImg.pixels, model.globalMean, model.projectionMatrix);
            vector<double> reconstructedImage = reconstructImage(projection, model.projectionMatrix, model.globalMean);

            // Extract filename from the full path
            string baseFilename = fs::path(testImg.filename).filename().string();
            string result = correct ? "correct" : "incorrect";

            // Save reconstructed test image
            PGMImage::saveFromVector(reconstructedImage, model.imgWidth, model.imgHeight,
                                     "lda_reconstructions/" + baseFilename + "_" + result + ".pgm");
        }

        // Print the confusion matrix (abbreviated if too large)
        cout << "\nConfusion Matrix (partial):" << endl;

        int maxToShow = min(10, static_cast<int>(allPersonIds.size()));
        auto it = allPersonIds.begin();

        cout << setw(10) << " ";
        for (int i = 0; i < maxToShow; i++, ++it)
        {
            cout << setw(10) << *it;
        }
        if (allPersonIds.size() > maxToShow)
            cout << setw(10) << "...";
        cout << setw(10) << "unknown" << endl;

        it = allPersonIds.begin();
        for (int i = 0; i < maxToShow; i++, ++it)
        {
            string trueId = *it;
            cout << setw(10) << trueId;

            auto jt = allPersonIds.begin();
            for (int j = 0; j < maxToShow; j++, ++jt)
            {
                cout << setw(10) << confusionMatrix[trueId][*jt];
            }

            if (allPersonIds.size() > maxToShow)
                cout << setw(10) << "...";

            cout << setw(10) << confusionMatrix[trueId]["unknown"] << endl;
        }

        if (allPersonIds.size() > maxToShow)
        {
            cout << setw(10) << "..." << setw(10 * (maxToShow + (allPersonIds.size() > maxToShow ? 1 : 0) + 1)) << "..." << endl;
        }

        // Calculate and print accuracy
        double accuracy = static_cast<double>(correctCount) / testingData.size() * 100.0;
        cout << "\nRecognition accuracy: " << fixed << setprecision(2) << accuracy << "% ("
             << correctCount << "/" << testingData.size() << ")" << endl;
        cout << "Reconstructed test images saved to 'lda_reconstructions/' directory" << endl;
    }
    else
    {
        cerr << "Invalid mode. Use --train, --test, or --eval." << endl;
        return 1;
    }

    // Print all benchmark results
    printEssentialBenchmarkResults();

    return 0;
}