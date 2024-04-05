#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

class LogisticRegression
{
public:
    LogisticRegression(int numFeatures)
    {
        this->weights_ = cv::Mat::zeros(numFeatures + 1, 1, CV_32F);
    }

    ~LogisticRegression()
    {
    }

    cv::Mat weights_;      // Weight vector, including bias weight
    cv::Mat predictions_;  // To store predictions after calling `predict`
    cv::Mat dataWithBias_; // To store the data matrix with the bias term

    cv::Mat dataPreprocessor(const cv::Mat &data)
    {
        cv::Mat dataWithBias;
        cv::hconcat(cv::Mat::ones(data.rows, 1, data.type()), data, dataWithBias); // Concatenate the column of ones to the original data matrix
        return dataWithBias;
    }

    void train(int epochs, cv::Mat &trainData, cv::Mat &trainLabels, cv::Mat &testData, cv::Mat &testLabels)
    {
        for (int i = 0; i < epochs; ++i)
        {
            // Preprocess training data locally
            cv::Mat trainDataWithBias = dataPreprocessor(trainData);

            // Obtain predictions for the training data
            cv::Mat trainPredictions = predict(trainDataWithBias); // Assuming predict now uses data with bias directly

            // Construct the diagonal weight matrix W for IRLS
            cv::Mat W = cv::Mat::zeros(trainData.rows, trainData.rows, CV_32F);
            for (int j = 0; j < trainData.rows; ++j)
            {
                float p = trainPredictions.at<float>(j, 0);
                W.at<float>(j, j) = p * (1 - p);
            }

            // Calculate Hessian and gradient for weight update
            cv::Mat Hessian = trainDataWithBias.t() * W * trainDataWithBias;
            cv::Mat gradient = trainDataWithBias.t() * (trainPredictions - trainLabels);
            cv::Mat HessianInv;
            cv::invert(Hessian, HessianInv, cv::DECOMP_SVD); // Using SVD for inversion for better numerical stability
            weights_ -= HessianInv * gradient;

            // Calculate and print accuracy on the test data for monitoring
            cv::Mat testDataWithBias = dataPreprocessor(testData); // Preprocess test data
            cv::Mat testPredictions = predict(testDataWithBias);   // Predict on test data
            cv::Mat predictedTestLabels;
            cv::threshold(testPredictions, predictedTestLabels, 0.5, 1, cv::THRESH_BINARY); // Convert probabilities to binary predictions

            float testAccuracy = cv::countNonZero(predictedTestLabels == testLabels) / static_cast<float>(testLabels.rows);
            std::cout << "Epoch: " << i << " Test Accuracy: " << testAccuracy << std::endl;
        }
    }

    cv::Mat predict(const cv::Mat &dataWithBias)
    {
        return sigmoid(dataWithBias, this->weights_);
    }

    cv::Mat sigmoid(const cv::Mat &dataWithBias, const cv::Mat &weights)
    {
        cv::Mat exponendOfWeightedSums;
        cv::Mat weightedSums = dataWithBias * weights;
        cv::exp(-weightedSums, exponendOfWeightedSums);
        return 1.0 / (1.0 + exponendOfWeightedSums);
    }
};

cv::Mat convertLabels(const cv::Mat &originalLabels)
{
    cv::Mat binaryLabels = originalLabels.clone();
    for (int i = 0; i < binaryLabels.rows; ++i)
    {
        if (binaryLabels.at<float>(i, 0) == 7)
        {
            binaryLabels.at<float>(i, 0) = 0.0; // Map 7 to 0
        }
        else if (binaryLabels.at<float>(i, 0) == 8)
        {
            binaryLabels.at<float>(i, 0) = 1.0; // Map 8 to 1
        }
    }
    return binaryLabels;
}

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

    // Print the number of 7 and 8 in the training data
    cv::Mat trainResponses = data.trainData->getResponses();
    int count7 = cv::countNonZero(trainResponses == 7);
    int count8 = cv::countNonZero(trainResponses == 8);
    std::cout << "Number of 7s in the training data: " << count7 << std::endl;
    std::cout << "Number of 8s in the training data: " << count8 << std::endl;

    // Similarly, create test data from the next 5000 rows, excluding responses 7 and 8
    data.testData = filterData(data.fullData, 1000, 6000);
    // Print the number of 7 and 8 in the test data
    cv::Mat testResponses = data.testData->getResponses();
    count7 = cv::countNonZero(testResponses == 7);
    count8 = cv::countNonZero(testResponses == 8);
    std::cout << "Number of 7s in the test data: " << count7 << std::endl;
    std::cout << "Number of 8s in the test data: " << count8 << std::endl;

    // Get the samples and responses
    cv::Mat trainSamples = data.trainData->getTrainSamples();  // Get design matrix
    cv::Mat trainTarget = data.trainData->getTrainResponses(); // Get target values
    cv::Mat testSamples = data.testData->getTrainSamples();    // Get design matrix
    cv::Mat testTarget = data.testData->getTrainResponses();   // Get target values

    // Standardize the data to have zero mean and a standard deviation of 1
    standardize(trainSamples);
    standardize(testSamples);

    // Sanity check if the data is standardized correctly
    /*
    The observed mean is close to zero which is expected after standardization but the standard deviation is at around 0.8 which is not 1 as expected. This could mean
    that the standardization formula was not applied correctly or the data is not gaussian distributed. I assume the standardization was done correctly. Help :(
    */
    cv::Scalar trainMean, trainStddev, testMean, testStddev;
    cv::meanStdDev(trainSamples, trainMean, trainStddev); // Calculate mean and standard deviation for the training samples
    cv::meanStdDev(testSamples, testMean, testStddev);    // Calculate mean and standard deviation for the test samples

    std::cout << "Train Mean: " << trainMean[0] << " Train Stddev: " << trainStddev[0] << std::endl;
    std::cout << "Test Mean: " << testMean[0] << " Test Stddev: " << testStddev[0] << std::endl;

    cv::PCA pcaTrainAllDimensions(trainSamples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::PCA pcaTestAllDimensions(testSamples, cv::Mat(), cv::PCA::DATA_AS_ROW);

    std::ofstream outFile("eigenvalues_train.csv");
    for (int i = 0; i < pcaTrainAllDimensions.eigenvalues.rows; i++)
    {
        outFile << i + 1 << "," << pcaTrainAllDimensions.eigenvalues.at<float>(i, 0) << std::endl;
    }
    outFile.close();

    outFile.open("eigenvalues_test.csv");
    for (int i = 0; i < pcaTestAllDimensions.eigenvalues.rows; i++)
    {
        outFile << i + 1 << "," << pcaTestAllDimensions.eigenvalues.at<float>(i, 0) << std::endl;
    }
    outFile.close();

    cv::PCA pcaTrain85D(trainSamples, cv::Mat(), cv::PCA::DATA_AS_ROW, 85);
    cv::PCA pcaTest85D(testSamples, cv::Mat(), cv::PCA::DATA_AS_ROW, 85);

    cv::Mat projected85DTrainSamples = pcaTrain85D.project(trainSamples);
    cv::Mat projected85DTestSamples = pcaTest85D.project(testSamples);

    LogisticRegression model85D(85);

    // Map 7 to 0 and 8 to 1
    cv::Mat trainLabelsBinary = convertLabels(trainTarget); // Convert the training labels to binary
    cv::Mat testLabelsBinary = convertLabels(testTarget);   // Convert the test labels to binary

    model85D.train(18, projected85DTrainSamples, trainLabelsBinary, projected85DTestSamples, testLabelsBinary);

    cv::waitKey(0);

    return 0;
}