#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

void standardize(cv::Mat &data)
{
    for (int col = 0; col < data.cols; ++col)
    {
        cv::Scalar mean, stddev;

        // Compute mean and standard deviation for each column (pixels)
        cv::meanStdDev(data.col(col), mean, stddev);

        if (stddev[0] > 0) // Prevent division by zero
        {
            /*
            Standardize each element in the column.
            The formula for standardization is: Z = (X - mean) / stddev
            where X is the original value, mean is the average of the feature,
            and stddev is the standard deviation of the feature.
            */
            for (int row = 0; row < data.rows; ++row)
            {
                data.at<float>(row, col) = (data.at<float>(row, col) - mean[0]) / stddev[0];
            }
        }
    }
}

// Data structure to store datasets
struct Data
{
    cv::Ptr<cv::ml::TrainData> fullData;  // Holds the complete dataset
    cv::Ptr<cv::ml::TrainData> trainData; // Holds the training dataset
    cv::Ptr<cv::ml::TrainData> testData;  // Holds the test dataset
};

cv::Ptr<cv::ml::TrainData> filterData(const cv::Ptr<cv::ml::TrainData> &originalData, int startRow, int endRow)
{
    // Get the total number of rows in the original data
    int totalRows = originalData->getSamples().rows;
    startRow = std::max(0, startRow);     // Ensure startRow is not less than 0
    endRow = std::min(endRow, totalRows); // Ensure endRow does not exceed totalRows

    // Get the samples and responses within the specified range
    cv::Mat samples = originalData->getSamples().rowRange(startRow, endRow);     // Get the samples
    cv::Mat responses = originalData->getResponses().rowRange(startRow, endRow); // Get the responses

    // Apply the mask: Find rows where the response is 7 or 8
    cv::Mat mask = (responses == 7) | (responses == 8);

    // Create empty matrices for storing filtered data
    cv::Mat filteredSamples, filteredResponses;

    // Loop through the mask and filter samples and responses
    for (size_t i = 0; i < mask.total(); ++i)
    {
        if (mask.at<uint8_t>(i))
        {
            // Check if the mask at position i is true
            filteredSamples.push_back(samples.row(i));
            filteredResponses.push_back(responses.row(i));
        }
    }

    // Create and return a new TrainData instance from the filtered samples and responses
    return cv::ml::TrainData::create(filteredSamples, cv::ml::ROW_SAMPLE, filteredResponses);
}

int main(int argc, char *argv[])
{
    /*
    There was an issue when loading the mnist_train.csv file, so I used the mnist_test.csv file instead.
    I was able to load a subset of the size 5000 of the mnist_test.csv file so I think it is because of the
    size of the file.
    */
    // Load the data
    Data data;
    data.fullData = cv::ml::TrainData::loadFromCSV("./mnist_test.csv", 0, 0, 1); // First col is the target as a float

    // Apply filterData function to create training data from the first 1000 rows and exclude responses 7 and 8
    data.trainData = filterData(data.fullData, 0, 1000);

    // Similarly, create test data from the next 5000 rows, excluding responses 7 and 8
    data.testData = filterData(data.fullData, 1000, 6000);

    // Get the samples and responses
    cv::Mat trainSamples = data.trainData->getTrainSamples();  // Get design matrix
    cv::Mat trainTarget = data.trainData->getTrainResponses(); // Get target values
    cv::Mat testSamples = data.testData->getTrainSamples();    // Get design matrix
    cv::Mat testTarget = data.testData->getTrainResponses();   // Get target values

    // Sanity check if the datas rows and collumns are reasonable
    std::cout << "Train Samples: " << trainSamples.rows << "x" << trainSamples.cols << std::endl;
    std::cout << "Train Target: " << trainTarget.rows << "x" << trainTarget.cols << std::endl;
    std::cout << "Test Samples: " << testSamples.rows << "x" << testSamples.cols << std::endl;
    std::cout << "Test Target: " << testTarget.rows << "x" << testTarget.cols << std::endl;

    // Sanity check if the data is filtered correctly
    for (int i = 0; i < trainTarget.rows; ++i)
    {
        float value = trainTarget.at<float>(i, 0);
        std::cout << "Test Target [" << i << "]: " << value << std::endl;
    }

    // Standardize the data to have zero mean and a standard deviation of 1
    standardize(trainSamples);
    standardize(testSamples);

    // Sanity check if the data is standardized correctly
    /*
    The observed mean is close to zero which is expected after standardization but the standard deviation is at around 0.8 which is not 1 as expected. This could mean
    that the standardization formula was not applied correctly or the data is not gaussian distributed. I assume the standardization was done correctly. Help :(
    */
    cv::Scalar trainMean, trainStddev, testMean, testStddev;
    cv::meanStdDev(trainSamples, trainMean, trainStddev);                                                  // Calculate mean and standard deviation for the training samples
    cv::meanStdDev(testSamples, testMean, testStddev);                                                     // Calculate mean and standard deviation for the test samples
    std::cout << "Training Data: Mean = " << trainMean[0] << ", Stddev = " << trainStddev[0] << std::endl; // Print out the mean and standard deviation for training set
    std::cout << "Testing Data: Mean = " << testMean[0] << ", Stddev = " << testStddev[0] << std::endl;    // Print out the mean and standard deviation for test set

    // Perform PCA analysis on the training samples
    int PCADim = 2;
    cv::PCA pca2DTrainSamples(trainSamples, cv::Mat(), 0, PCADim);
    cv::PCA pca2DTestSamples(testSamples, cv::Mat(), 0, PCADim);

    PCADim = 3;
    cv::PCA pca3DTrainSamples(trainSamples, cv::Mat(), 0, PCADim);
    cv::PCA pca3DTestSamples(testSamples, cv::Mat(), 0, PCADim);

    // Project the training samples onto the PCA space
    cv::Mat projected2DTrainSamples = pca2DTrainSamples.project(trainSamples);
    cv::Mat projected3DTrainSamples = pca3DTrainSamples.project(trainSamples);

    // Visualize 2D PCA Projection
    int width2D = 600, height2D = 600;
    cv::Mat visualization2D = cv::Mat::zeros(height2D, width2D, CV_8UC3);

    double minValX2D, maxValX2D, minValY2D, maxValY2D;
    cv::minMaxLoc(projected2DTrainSamples.col(0), &minValX2D, &maxValX2D);
    cv::minMaxLoc(projected2DTrainSamples.col(1), &minValY2D, &maxValY2D);

    for (int i = 0; i < projected2DTrainSamples.rows; i++)
    {
        float x = (projected2DTrainSamples.at<float>(i, 0) - minValX2D) / (maxValX2D - minValX2D) * (width2D - 40) + 20;
        float y = (projected2DTrainSamples.at<float>(i, 1) - minValY2D) / (maxValY2D - minValY2D) * (height2D - 40) + 20;

        int label = static_cast<int>(trainTarget.at<float>(i, 0));
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);

        cv::circle(visualization2D, cv::Point(static_cast<int>(x), static_cast<int>(height2D - y)), 3, color, CV_FILLED);
    }

    cv::imshow("2D PCA Projection", visualization2D);

    // Visualize 3D PCA Projection (Simulated in 2D)
    int width3D = 600, height3D = 600;
    cv::Mat visualization3D = cv::Mat::zeros(height3D, width3D, CV_8UC3);

    double minValX3D, maxValX3D, minValY3D, maxValY3D, minValZ3D, maxValZ3D;
    cv::minMaxLoc(projected3DTrainSamples.col(0), &minValX3D, &maxValX3D);
    cv::minMaxLoc(projected3DTrainSamples.col(1), &minValY3D, &maxValY3D);
    cv::minMaxLoc(projected3DTrainSamples.col(2), &minValZ3D, &maxValZ3D);

    for (int i = 0; i < projected3DTrainSamples.rows; i++)
    {
        float x = (projected3DTrainSamples.at<float>(i, 0) - minValX3D) / (maxValX3D - minValX3D) * (width3D - 40) + 20;
        float y = (projected3DTrainSamples.at<float>(i, 1) - minValY3D) / (maxValY3D - minValY3D) * (height3D - 40) + 20;
        float z = (projected3DTrainSamples.at<float>(i, 2) - minValZ3D) / (maxValZ3D - minValZ3D); // Used for depth effect

        int label = static_cast<int>(trainTarget.at<float>(i, 0));
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
        int radius = 3 + static_cast<int>(z * 10); // Simulate depth by changing the radius based on z value

        cv::circle(visualization3D, cv::Point(static_cast<int>(x), static_cast<int>(height3D - y)), radius, color, CV_FILLED);
    }

    cv::imshow("Simulated 3D PCA Projection", visualization3D);

    cv::PCA pcaFull(trainSamples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    
    std::ofstream outFile("eigenvalues.csv");
    for (int i = 0; i < pcaFull.eigenvalues.rows; i++)
    {
        outFile << i + 1 << "," << pcaFull.eigenvalues.at<float>(i, 0) << std::endl;
    }
    outFile.close();

    cv::waitKey(0);
    return 0;
}