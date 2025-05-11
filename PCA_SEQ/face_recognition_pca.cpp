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

using namespace std;
namespace fs = std::filesystem;

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

// Structure to hold the PCA model
struct PCAModel
{
    int imgWidth;
    int imgHeight;
    vector<double> meanFace;
    vector<vector<double>> eigenfaces;
    vector<double> eigenvalues;
    vector<vector<double>> projectedTrainingData;
    vector<string> trainingPersonIds; // Person IDs corresponding to projectedTrainingData

    // Save the model to a file
    bool saveToFile(const string &filename) const
    {
        Timer timer("PCAModel::saveToFile");

        ofstream file(filename, ios::binary);
        if (!file)
        {
            cerr << "Failed to open " << filename << " for writing" << endl;
            return false;
        }

        // Write dimensions
        file.write(reinterpret_cast<const char *>(&imgWidth), sizeof(imgWidth));
        file.write(reinterpret_cast<const char *>(&imgHeight), sizeof(imgHeight));

        // Write size of meanFace
        size_t meanFaceSize = meanFace.size();
        file.write(reinterpret_cast<const char *>(&meanFaceSize), sizeof(meanFaceSize));

        // Write meanFace vector
        file.write(reinterpret_cast<const char *>(meanFace.data()), meanFaceSize * sizeof(double));

        // Write size of eigenfaces
        size_t numEigenfaces = eigenfaces.size();
        file.write(reinterpret_cast<const char *>(&numEigenfaces), sizeof(numEigenfaces));

        if (numEigenfaces > 0)
        {
            size_t eigenfaceSize = eigenfaces[0].size();
            file.write(reinterpret_cast<const char *>(&eigenfaceSize), sizeof(eigenfaceSize));

            // Write eigenfaces
            for (const auto &eigenface : eigenfaces)
            {
                file.write(reinterpret_cast<const char *>(eigenface.data()), eigenfaceSize * sizeof(double));
            }
        }

        // Write eigenvalues
        size_t eigenvaluesSize = eigenvalues.size();
        file.write(reinterpret_cast<const char *>(&eigenvaluesSize), sizeof(eigenvaluesSize));
        file.write(reinterpret_cast<const char *>(eigenvalues.data()), eigenvaluesSize * sizeof(double));

        // Write size of projectedTrainingData
        size_t numTrainingSamples = projectedTrainingData.size();
        file.write(reinterpret_cast<const char *>(&numTrainingSamples), sizeof(numTrainingSamples));

        if (numTrainingSamples > 0)
        {
            size_t projectionSize = projectedTrainingData[0].size();
            file.write(reinterpret_cast<const char *>(&projectionSize), sizeof(projectionSize));

            // Write projectedTrainingData
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
        Timer timer("PCAModel::loadFromFile");

        ifstream file(filename, ios::binary);
        if (!file)
        {
            cerr << "Failed to open " << filename << " for reading" << endl;
            return false;
        }

        // Read dimensions
        file.read(reinterpret_cast<char *>(&imgWidth), sizeof(imgWidth));
        file.read(reinterpret_cast<char *>(&imgHeight), sizeof(imgHeight));

        // Read meanFace
        size_t meanFaceSize;
        file.read(reinterpret_cast<char *>(&meanFaceSize), sizeof(meanFaceSize));

        meanFace.resize(meanFaceSize);
        file.read(reinterpret_cast<char *>(meanFace.data()), meanFaceSize * sizeof(double));

        // Read eigenfaces
        size_t numEigenfaces;
        file.read(reinterpret_cast<char *>(&numEigenfaces), sizeof(numEigenfaces));

        eigenfaces.resize(numEigenfaces);
        if (numEigenfaces > 0)
        {
            size_t eigenfaceSize;
            file.read(reinterpret_cast<char *>(&eigenfaceSize), sizeof(eigenfaceSize));

            for (auto &eigenface : eigenfaces)
            {
                eigenface.resize(eigenfaceSize);
                file.read(reinterpret_cast<char *>(eigenface.data()), eigenfaceSize * sizeof(double));
            }
        }

        // Read eigenvalues
        size_t eigenvaluesSize;
        file.read(reinterpret_cast<char *>(&eigenvaluesSize), sizeof(eigenvaluesSize));

        eigenvalues.resize(eigenvaluesSize);
        file.read(reinterpret_cast<char *>(eigenvalues.data()), eigenvaluesSize * sizeof(double));

        // Read projectedTrainingData
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

// Function to compute the mean vector
vector<double> computeMean(const vector<vector<double>> &data)
{
    Timer timer("computeMean", data.size() * data[0].size(), "elements");

    int numSamples = data.size();
    int featureSize = data[0].size();
    vector<double> mean(featureSize, 0.0);

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            mean[j] += data[i][j];
        }
    }

    for (int j = 0; j < featureSize; j++)
    {
        mean[j] /= numSamples;
    }

    return mean;
}

// Function to center the data (subtract mean)
vector<vector<double>> centerData(const vector<vector<double>> &data, const vector<double> &mean)
{
    Timer timer("centerData", data.size() * data[0].size(), "elements");

    int numSamples = data.size();
    int featureSize = data[0].size();
    vector<vector<double>> centeredData(numSamples, vector<double>(featureSize, 0.0));

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            centeredData[i][j] = data[i][j] - mean[j];
        }
    }

    return centeredData;
}

// Function to compute the covariance matrix using matrix multiplication
vector<vector<double>> computeCovarianceMatrix(const vector<vector<double>> &centeredData)
{
    Timer timer("computeCovarianceMatrix",
                centeredData.size() * centeredData.size(),
                "matrix elements");

    int numSamples = centeredData.size();
    int featureSize = centeredData[0].size();

    // For face images, we have more features than samples
    // We'll use the trick to compute a smaller matrix (numSamples x numSamples)
    vector<vector<double>> smallCovMatrix(numSamples, vector<double>(numSamples, 0.0));

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < numSamples; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < featureSize; k++)
            {
                sum += centeredData[i][k] * centeredData[j][k];
            }
            smallCovMatrix[i][j] = sum / (numSamples - 1);
        }
    }

    return smallCovMatrix;
}

// Function to perform power iteration to find the largest eigenvalue and eigenvector
void powerIteration(const vector<vector<double>> &matrix, double &eigenvalue, vector<double> &eigenvector, int maxIters = 100, double tolerance = 1e-10)
{
    Timer timer("powerIteration", matrix.size() * matrix.size(), "matrix elements");

    int n = matrix.size();

    // Start with a random vector
    eigenvector.resize(n);
    for (int i = 0; i < n; i++)
    {
        eigenvector[i] = rand() / (RAND_MAX + 1.0);
    }

    // Normalize the initial vector
    double norm = 0.0;
    for (int i = 0; i < n; i++)
    {
        norm += eigenvector[i] * eigenvector[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < n; i++)
    {
        eigenvector[i] /= norm;
    }

    double prevEigenvalue = 0.0;
    eigenvalue = 0.0;

    for (int iter = 0; iter < maxIters; iter++)
    {
        // Save the previous eigenvalue
        prevEigenvalue = eigenvalue;

        // Multiply matrix and vector
        vector<double> result(n, 0.0);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i] += matrix[i][j] * eigenvector[j];
            }
        }

        // Compute the Rayleigh quotient (approximation of eigenvalue)
        eigenvalue = 0.0;
        for (int i = 0; i < n; i++)
        {
            eigenvalue += eigenvector[i] * result[i];
        }

        // Normalize the result vector
        norm = 0.0;
        for (int i = 0; i < n; i++)
        {
            norm += result[i] * result[i];
        }
        norm = sqrt(norm);

        // Update eigenvector
        for (int i = 0; i < n; i++)
        {
            eigenvector[i] = result[i] / norm;
        }

        // Check for convergence
        if (fabs(eigenvalue - prevEigenvalue) < tolerance)
        {
            break;
        }
    }
}

// Function to compute top eigenvalues and eigenvectors
void computeEigenvectors(const vector<vector<double>> &matrix, int numComponents, vector<double> &eigenvalues, vector<vector<double>> &eigenvectors)
{
    Timer timer("computeEigenvectors",
                matrix.size() * matrix.size() * numComponents,
                "operations");

    int n = matrix.size();
    eigenvalues.resize(numComponents);
    eigenvectors.resize(numComponents, vector<double>(n));

    vector<vector<double>> remainingMatrix = matrix;

    for (int i = 0; i < numComponents; i++)
    {
        vector<double> eigenvector;
        double eigenvalue;

        powerIteration(remainingMatrix, eigenvalue, eigenvector);

        eigenvalues[i] = eigenvalue;
        eigenvectors[i] = eigenvector;

        // Deflate the matrix to find the next eigenvector
        if (i < numComponents - 1)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    remainingMatrix[j][k] -= eigenvalue * eigenvector[j] * eigenvector[k];
                }
            }
        }
    }
}

// Function to convert small eigenvectors to full eigenvectors
vector<vector<double>> convertToFullEigenvectors(const vector<vector<double>> &centeredData, const vector<vector<double>> &smallEigenvectors)
{
    Timer timer("convertToFullEigenvectors",
                centeredData.size() * centeredData[0].size() * smallEigenvectors.size(),
                "operations");

    int numSamples = centeredData.size();
    int featureSize = centeredData[0].size();
    int numEigenvectors = smallEigenvectors.size();

    vector<vector<double>> fullEigenvectors(numEigenvectors, vector<double>(featureSize, 0.0));

    for (int i = 0; i < numEigenvectors; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            for (int k = 0; k < numSamples; k++)
            {
                fullEigenvectors[i][j] += smallEigenvectors[i][k] * centeredData[k][j];
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

// Function to project data onto eigenvectors
vector<vector<double>> projectData(const vector<vector<double>> &centeredData, const vector<vector<double>> &eigenvectors)
{
    Timer timer("projectData",
                centeredData.size() * centeredData[0].size() * eigenvectors.size(),
                "operations");

    int numSamples = centeredData.size();
    int numEigenvectors = eigenvectors.size();
    int featureSize = centeredData[0].size();

    vector<vector<double>> projectedData(numSamples, vector<double>(numEigenvectors, 0.0));

    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < numEigenvectors; j++)
        {
            for (int k = 0; k < featureSize; k++)
            {
                projectedData[i][j] += centeredData[i][k] * eigenvectors[j][k];
            }
        }
    }

    return projectedData;
}

// Function to project a single image onto the eigenfaces
vector<double> projectImage(const vector<double> &image, const vector<double> &meanFace, const vector<vector<double>> &eigenfaces)
{
    Timer timer("projectImage",
                image.size() * eigenfaces.size(),
                "operations");

    int numEigenvectors = eigenfaces.size();
    int featureSize = image.size();

    // Center the image
    vector<double> centeredImage(featureSize);
    for (int i = 0; i < featureSize; i++)
    {
        centeredImage[i] = image[i] - meanFace[i];
    }

    // Project onto eigenfaces
    vector<double> projection(numEigenvectors);
    for (int i = 0; i < numEigenvectors; i++)
    {
        projection[i] = 0.0;
        for (int j = 0; j < featureSize; j++)
        {
            projection[i] += centeredImage[j] * eigenfaces[i][j];
        }
    }

    return projection;
}

// Function to reconstruct data from projection
vector<vector<double>> reconstructData(const vector<vector<double>> &projectedData, const vector<vector<double>> &eigenvectors, const vector<double> &mean)
{
    Timer timer("reconstructData",
                projectedData.size() * projectedData[0].size() * eigenvectors[0].size(),
                "operations");

    int numSamples = projectedData.size();
    int numEigenvectors = eigenvectors.size();
    int featureSize = eigenvectors[0].size();

    vector<vector<double>> reconstructedData(numSamples, vector<double>(featureSize, 0.0));

    // First add the projection of each image
    for (int i = 0; i < numSamples; i++)
    {
        // Start with zeros
        for (int j = 0; j < featureSize; j++)
        {
            reconstructedData[i][j] = 0.0;
        }

        // Add the contribution of each eigenvector
        for (int j = 0; j < numEigenvectors; j++)
        {
            for (int k = 0; k < featureSize; k++)
            {
                reconstructedData[i][k] += projectedData[i][j] * eigenvectors[j][k];
            }
        }

        // Add the mean back
        for (int j = 0; j < featureSize; j++)
        {
            reconstructedData[i][j] += mean[j];
        }
    }

    return reconstructedData;
}

// Function to reconstruct a single image from its projection
vector<double> reconstructImage(const vector<double> &projection, const vector<vector<double>> &eigenfaces, const vector<double> &meanFace)
{
    Timer timer("reconstructImage",
                projection.size() * eigenfaces[0].size(),
                "operations");

    int featureSize = eigenfaces[0].size();
    int numComponents = projection.size();

    vector<double> reconstructed(featureSize, 0.0);

    // Reconstruct from projection
    for (int i = 0; i < numComponents; i++)
    {
        for (int j = 0; j < featureSize; j++)
        {
            reconstructed[j] += projection[i] * eigenfaces[i][j];
        }
    }

    // Add the mean back
    for (int j = 0; j < featureSize; j++)
    {
        reconstructed[j] += meanFace[j];
    }

    return reconstructed;
}

// Modified computeDistance without timing
double computeDistance(const vector<double> &vec1, const vector<double> &vec2)
{
    assert(vec1.size() == vec2.size());
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Modified findClosestMatch with aggregated timing
int findClosestMatch(const vector<double> &queryFeatures, const vector<vector<double>> &trainingFeatures)
{
    Timer timer("distance_computations",
                trainingFeatures.size() * queryFeatures.size(),
                "operations");

    int numTraining = trainingFeatures.size();
    int closestIndex = -1;
    double minDistance = numeric_limits<double>::max();

    for (int i = 0; i < numTraining; i++)
    {
        double distance = computeDistance(queryFeatures, trainingFeatures[i]);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestIndex = i;
        }
    }

    return closestIndex;
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

// Function to train the PCA model
PCAModel trainPCAModel(const vector<ImageData> &trainingData, int imgWidth, int imgHeight, int numComponents)
{
    Timer timer("trainPCAModel",
                trainingData.size() * imgWidth * imgHeight,
                "pixels processed");

    PCAModel model;
    model.imgWidth = imgWidth;
    model.imgHeight = imgHeight;

    // Extract pixel vectors and person IDs
    vector<vector<double>> imageVectors;
    for (const auto &img : trainingData)
    {
        imageVectors.push_back(img.pixels);
        model.trainingPersonIds.push_back(img.personId);
    }

    // Step 1: Compute the mean vector (mean face)
    model.meanFace = computeMean(imageVectors);

    // Step 2: Center the data
    vector<vector<double>> centeredData = centerData(imageVectors, model.meanFace);

    // Step 3: Compute the small covariance matrix (numSamples x numSamples)
    vector<vector<double>> smallCovMatrix = computeCovarianceMatrix(centeredData);

    // Step 4: Compute eigenvectors of the small covariance matrix
    vector<vector<double>> smallEigenvectors;
    int actualComponents = min(numComponents, (int)smallCovMatrix.size());
    computeEigenvectors(smallCovMatrix, actualComponents, model.eigenvalues, smallEigenvectors);

    // Step 5: Convert to full eigenvectors (eigenfaces)
    model.eigenfaces = convertToFullEigenvectors(centeredData, smallEigenvectors);

    // Step 6: Project training data onto eigenfaces
    model.projectedTrainingData = projectData(centeredData, model.eigenfaces);

    return model;
}

// Function to recognize a face using a trained model
string recognizeFace(const PCAModel &model, const vector<double> &faceImage)
{
    Timer timer("recognizeFace");

    // Project the face onto the eigenfaces
    vector<double> projection = projectImage(faceImage, model.meanFace, model.eigenfaces);

    // Find the closest match
    int closestMatchIndex = findClosestMatch(projection, model.projectedTrainingData);

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
// Function to print only essential benchmark results
void printEssentialBenchmarkResults()
{
    // These are the key operations we want to track for parallel comparison
    const set<string> essentialOps = {
        "loadHierarchicalDataset",
        "PGMImage::load",
        "computeMean",
        "centerData",
        "computeCovarianceMatrix",
        "powerIteration",
        "computeEigenvectors",
        "convertToFullEigenvectors",
        "projectData",
        "distance_computations",
        "recognizeFace",
        "trainPCAModel"};

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
        int numComponents = (argc > 3) ? atoi(argv[3]) : 100;       // Default to 100 components
        string modelFile = (argc > 4) ? argv[4] : "face_model.dat"; // Default model filename

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

        cout << "Training PCA model with " << numComponents << " components..." << endl;

        // Train the PCA model
        PCAModel model = trainPCAModel(trainingData, imgWidth, imgHeight, numComponents);

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

        // Save the mean face as an image
        PGMImage::saveFromVector(model.meanFace, imgWidth, imgHeight, "mean_face.pgm");
        cout << "Saved mean face to 'mean_face.pgm'" << endl;

        // Save the eigenfaces as images
        fs::create_directory("eigenfaces");
        for (int i = 0; i < min(10, (int)model.eigenfaces.size()); i++)
        {
            PGMImage::saveFromVector(model.eigenfaces[i], imgWidth, imgHeight, "eigenfaces/eigenface_" + to_string(i) + ".pgm");
        }
        cout << "Saved top 10 eigenfaces to 'eigenfaces/' directory" << endl;

        // Evaluate on test set
        if (!testingData.empty())
        {
            int correctCount = 0;
            fs::create_directory("reconstructions");

            cout << "\nTesting face recognition on " << testingData.size() << " images..." << endl;

            for (const auto &testImg : testingData)
            {
                // Recognize the face
                string predictedPerson = recognizeFace(model, testImg.pixels);

                // Check if the prediction is correct
                bool correct = (predictedPerson == testImg.personId);
                if (correct)
                    correctCount++;

                // Reconstruct the test image using eigenfaces
                vector<double> projection = projectImage(testImg.pixels, model.meanFace, model.eigenfaces);
                vector<double> reconstructedImage = reconstructImage(projection, model.eigenfaces, model.meanFace);

                // Extract filename from the full path
                string baseFilename = fs::path(testImg.filename).filename().string();
                string result = correct ? "correct" : "incorrect";

                // Save reconstructed test image
                PGMImage::saveFromVector(reconstructedImage, imgWidth, imgHeight,
                                         "reconstructions/" + baseFilename + "_" + result + ".pgm");
            }

            // Calculate and print accuracy
            double accuracy = static_cast<double>(correctCount) / testingData.size() * 100.0;
            cout << "Recognition accuracy: " << fixed << setprecision(2) << accuracy << "% ("
                 << correctCount << "/" << testingData.size() << ")" << endl;
            cout << "Reconstructed test images saved to 'reconstructions/' directory" << endl;
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
        PCAModel model;
        if (!model.loadFromFile(modelFile))
        {
            cerr << "Failed to load model from " << modelFile << endl;
            return 1;
        }

        cout << "Loaded PCA model with " << model.eigenfaces.size() << " eigenfaces" << endl;

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

        // Reconstruct the image
        vector<double> projection = projectImage(testImageVector, model.meanFace, model.eigenfaces);
        vector<double> reconstructedImage = reconstructImage(projection, model.eigenfaces, model.meanFace);

        // Save the reconstructed image
        PGMImage::saveFromVector(reconstructedImage, model.imgWidth, model.imgHeight, "reconstructed_test.pgm");
        cout << "Saved reconstructed test image to 'reconstructed_test.pgm'" << endl;
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
        PCAModel model;
        if (!model.loadFromFile(modelFile))
        {
            cerr << "Failed to load model from " << modelFile << endl;
            return 1;
        }

        cout << "Loaded PCA model with " << model.eigenfaces.size() << " eigenfaces" << endl;

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
        fs::create_directory("reconstructions");

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
            vector<double> projection = projectImage(testImg.pixels, model.meanFace, model.eigenfaces);
            vector<double> reconstructedImage = reconstructImage(projection, model.eigenfaces, model.meanFace);

            // Extract filename from the full path
            string baseFilename = fs::path(testImg.filename).filename().string();
            string result = correct ? "correct" : "incorrect";

            // Save reconstructed test image
            PGMImage::saveFromVector(reconstructedImage, model.imgWidth, model.imgHeight,
                                     "reconstructions/" + baseFilename + "_" + result + ".pgm");
        }

        // Print the confusion matrix
        cout << "\nConfusion Matrix:" << endl;
        cout << setw(10) << " ";
        for (const auto &predId : allPersonIds)
        {
            cout << setw(10) << predId;
        }
        cout << setw(10) << "unknown" << endl;

        for (const auto &trueId : allPersonIds)
        {
            cout << setw(10) << trueId;
            for (const auto &predId : allPersonIds)
            {
                cout << setw(10) << confusionMatrix[trueId][predId];
            }
            cout << setw(10) << confusionMatrix[trueId]["unknown"] << endl;
        }

        // Calculate and print accuracy
        double accuracy = static_cast<double>(correctCount) / testingData.size() * 100.0;
        cout << "\nRecognition accuracy: " << fixed << setprecision(2) << accuracy << "% ("
             << correctCount << "/" << testingData.size() << ")" << endl;
        cout << "Reconstructed test images saved to 'reconstructions/' directory" << endl;
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