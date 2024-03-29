#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

// Data structure to store the training and test data
struct Data
{
    // Training and test data
    cv::Ptr<cv::ml::TrainData> fullData;
    cv::Ptr<cv::ml::TrainData> trainData;
    cv::Ptr<cv::ml::TrainData> testData;
};

cv::Ptr<cv::ml::TrainData> filterData(const cv::Ptr<cv::ml::TrainData> &originalData, int startRow, int endRow)
{
    // Get the total number of rows in the original data
    int totalRows = originalData->getSamples().rows;
    startRow = std::max(0, startRow);     // Ensure startRow is not less than 0
    endRow = std::min(endRow, totalRows); // Ensure endRow does not exceed totalRows

    // Get the samples and responses within the specified range
    cv::Mat samples = originalData->getSamples().rowRange(startRow, endRow);
    cv::Mat responses = originalData->getResponses().rowRange(startRow, endRow);

    // Apply the mask: Find rows where the response is not 7 or 8
    cv::Mat mask = (responses == 7) | (responses == 8);

    // Create empty matrices for storing filtered data
    cv::Mat filteredSamples, filteredResponses;

    // Loop through the mask and filter samples and responses
    for (size_t i = 0; i < mask.total(); ++i)
    {
        if (mask.at<uint8_t>(i))
        { // Check if the mask at position i is true
            filteredSamples.push_back(samples.row(i));
            filteredResponses.push_back(responses.row(i));
        }
    }

    // Create and return a new TrainData instance from the filtered samples and responses
    return cv::ml::TrainData::create(filteredSamples, cv::ml::ROW_SAMPLE, filteredResponses);
}

int main(int argc, char *argv[])
{
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

        return 0;
}