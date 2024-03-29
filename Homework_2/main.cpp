#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

void standardize(cv::Mat &data)
{
    for (size_t col = 0; col < data.cols; ++col)
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
            for (size_t row = 0; row < data.rows; ++row)
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

    return 0;
}