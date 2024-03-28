#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

// Data structure to store the training and test data
struct Data
{
    // Training and test data
    cv::Ptr<cv::ml::TrainData> trainData;
    cv::Ptr<cv::ml::TrainData> testData;
};

cv::Ptr<cv::ml::TrainData> filterData(const cv::Ptr<cv::ml::TrainData> &originalData, int maxRows)
{
    // Get the samples and responses
    cv::Mat samples = originalData->getSamples();
    cv::Mat responses = originalData->getResponses();

    // Limit the dataset to the first 'maxRows' rows
    samples = samples.rowRange(0, std::min(maxRows, samples.rows));
    responses = responses.rowRange(0, std::min(maxRows, responses.rows));

    // Find the rows where the response is not 7 or 8 (a little OpenCV magic here :D)
    cv::Mat mask = (responses != 7) & (responses != 8); // Matrix with the same size as responses, 1 if the condition is true, 0 otherwise

    // Create empty matrices for storing filtered data.
    cv::Mat filteredSamples, filteredResponses;

    // Loop through each element in the mask matrix. The 'total()' function returns the count of all elements.
    for (size_t i = 0; i < mask.total(); ++i)
    {
        // If the current mask element is true (i.e., not zero), proceed to filter.
        if (mask.at<uchar>(i)) // 'at<uchar>(i)' accesses the i-th element as an unsigned char.
        {
            // Append the i-th row of 'samples' to 'filteredSamples'. This row matches the condition.
            filteredSamples.push_back(samples.row(i));
            // Do the same for 'responses', keeping samples and their responses aligned.
            filteredResponses.push_back(responses.row(i));
        }
    }

    // Create a new TrainData instance from the filtered samples and responses
    return cv::ml::TrainData::create(filteredSamples, cv::ml::ROW_SAMPLE, filteredResponses);
}

int main(int argc, char *argv[])
{
    // Load the data
    Data data;
    data.trainData = cv::ml::TrainData::loadFromCSV("./mnist_train.csv", 0, 0, 1); // First col is the target as a float
    data.testData = cv::ml::TrainData::loadFromCSV("./mnist_test.csv", 0, 0, 1);   // First col is the target as a float

    // Filter the data
    // data.trainData = filterData(data.trainData, 1000); // First 1000, excluding 7 and 8
    // data.testData = filterData(data.testData, 5000);   // First 5000, excluding 7 and 8

    // Get the samples and responses
    // cv::Mat trainSamples = data.trainData->getTrainSamples();  // Get design matrix
    // cv::Mat trainTarget = data.trainData->getTrainResponses(); // Get target values
    cv::Mat testSamples = data.testData->getTrainSamples();    // Get design matrix
    cv::Mat testTarget = data.testData->getTrainResponses();   // Get target values

    // Print the size of the data for a sanity check
    // std::cout << "Train Samples: " << trainSamples.rows << "x" << trainSamples.cols << std::endl;
    // std::cout << "Train Target: " << trainTarget.rows << "x" << trainTarget.cols << std::endl;
    std::cout << "Test Samples: " << testSamples.rows << "x" << testSamples.cols << std::endl;
    std::cout << "Test Target: " << testTarget.rows << "x" << testTarget.cols << std::endl;

    return 0;
}