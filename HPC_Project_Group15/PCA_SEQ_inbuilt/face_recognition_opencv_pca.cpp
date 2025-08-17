#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <numeric>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

const int IMG_WIDTH = 92;
const int IMG_HEIGHT = 112;
const int NUM_COMPONENTS = 100;

// Load all images into a matrix
void loadImages(const string &datasetPath, Mat &dataMatrix)
{
    vector<string> imagePaths;

    for (const auto &personDir : fs::directory_iterator(datasetPath))
    {
        for (const auto &imgPath : fs::directory_iterator(personDir.path()))
        {
            imagePaths.push_back(imgPath.path().string());
        }
    }

    sort(imagePaths.begin(), imagePaths.end());

    int numImages = imagePaths.size();
    dataMatrix = Mat(numImages, IMG_WIDTH * IMG_HEIGHT, CV_32F);

    for (int i = 0; i < numImages; i++)
    {
        cout << "[INFO] Loading image " << i + 1 << "/" << numImages << ": " << imagePaths[i] << endl;

        Mat img = imread(imagePaths[i], IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cerr << "Could not load image: " << imagePaths[i] << endl;
            exit(EXIT_FAILURE);
        }
        resize(img, img, Size(IMG_WIDTH, IMG_HEIGHT));
        img.convertTo(img, CV_32F, 1.0 / 255); // normalize [0,1]
        img = img.reshape(1, 1);               // flatten to 1 row
        img.copyTo(dataMatrix.row(i));
    }
}

// Compute Mean Squared Reconstruction Error
float computeReconstructionError(const Mat &original, const Mat &reconstructed)
{
    float totalError = 0.0f;
    for (int i = 0; i < original.rows; ++i)
    {
        float mse = norm(original.row(i), reconstructed.row(i), NORM_L2SQR);
        totalError += mse;
    }
    return totalError / original.rows;
}

int main()
{
    string datasetPath = "/home/rohan-vinkare/Collage/1_Sem_6/HPC/hpc_project/HPC-5/att_faces"; // <-- Update path if needed
    Mat data;

    cout << "[INFO] Loading images from dataset..." << endl;
    loadImages(datasetPath, data);

    cout << "[INFO] Performing PCA with " << NUM_COMPONENTS << " components..." << endl;

    auto start = chrono::high_resolution_clock::now();

    PCA pca(data, Mat(), PCA::DATA_AS_ROW, NUM_COMPONENTS);
    Mat projected = pca.project(data);
    Mat reconstructed = pca.backProject(projected);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    float mse = computeReconstructionError(data, reconstructed);

    Mat eigenvalues = pca.eigenvalues;
    float totalVariance = sum(eigenvalues)[0];
    float retainedVariance = sum(eigenvalues.rowRange(0, NUM_COMPONENTS))[0];
    float varianceRatio = retainedVariance / totalVariance;

    cout << "\n========== PCA Performance Metrics ==========" << endl;
    cout << "Execution Time         : " << elapsed.count() << " seconds" << endl;
    cout << "Reconstruction MSE     : " << mse << endl;
    cout << "Explained Variance (%) : " << varianceRatio * 100 << "%" << endl;
    cout << "=============================================" << endl;

    return 0;
}
