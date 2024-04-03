#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

class LogisticRegression
{
public:
    LogisticRegression(int numFeatures)
    {
        // Initialize weights based on the number of PCA components/features + 1 for bias
        this->weights_ = cv::Mat::zeros(numFeatures + 1, 1, CV_32F);
        // std::cout << "Logistic Regression Object Created with " << numFeatures << " features" << std::endl;
    }

    ~LogisticRegression()
    {
        // std::cout << "Logistic Regression Object Destroyed" << std::endl;
    }

    cv::Mat weights_;      // Weight vector, including bias weight
    cv::Mat predictions_;  // To store predictions after calling `predict`
    cv::Mat dataWithBias_; // To store the data matrix with the bias term

    void dataPreprocessor(cv::Mat &data)
    {
        // std::cout << "Preprocessing Data" << std::endl;
        // Ensure data includes the bias term; prepend a column of ones
        cv::hconcat(cv::Mat::ones(data.rows, 1, data.type()), data, this->dataWithBias_); // Concatenate the column of ones to the original data matrix
    }

    void sigmoid(const cv::Mat &dataWithBias, const cv::Mat &weights)
    {
        cv::Mat exponendOfWeightedSums;                            // To store the exponential of the weighted sums
        cv::Mat weightedSums = -dataWithBias * weights;            // Perform matrix-vector multiplication to get weighted sums
        cv::exp(weightedSums, exponendOfWeightedSums);             // Apply the exponential function element-wise
        this->predictions_ = 1.0 / (1.0 + exponendOfWeightedSums); // Apply the sigmoid function element-wise

        // Sanity Check x)
        // std::cout << "Predictions: " << this->predictions_.rows << "x" << this->predictions_.cols << std::endl;
        // std::cout << "weightedSums: " << weightedSums.rows << "x" << weightedSums.cols << std::endl;
        // std::cout << "exponendOfWeightedSums: " << exponendOfWeightedSums.rows << "x" << exponendOfWeightedSums.cols << std::endl;
        // std::cout << "dataWithBias: " << dataWithBias.rows << "x" << dataWithBias.cols << std::endl;
        // std::cout << "weights: " << weights.rows << "x" << weights.cols << std::endl;
    }

    void train(int epochs, cv::Mat &data, cv::Mat &labels)
    {
        // std::cout << "Training Logistic Regression Model" << std::endl;
        // std::cout << "Number of Epochs: " << epochs << std::endl;

        dataPreprocessor(data);                                   // Preprocess the data
        cv::Mat W = cv::Mat::zeros(data.rows, data.rows, CV_32F); // Initialize W as a diagonal matrix
        for (int i = 0; i < epochs; ++i)
        {
            predict(data); // Obtain current predictions to fill 'predictions_'

            // Construct the diagonal weight matrix W for IRLS
            for (int j = 0; j < data.rows; ++j)
            {
                float p = predictions_.at<float>(j, 0);
                W.at<float>(j, j) = p * (1 - p);
            }

            cv::Mat Hessian = dataWithBias_.t() * W * dataWithBias_;
            cv::Mat gradient = dataWithBias_.t() * (predictions_ - labels);
            cv::Mat HessianInv;
            cv::invert(Hessian, HessianInv, cv::DECOMP_SVD);
            weights_ -= HessianInv * gradient;

            cv::Mat predictedLabels;
            predict(data);
            cv::threshold(predictions_, predictedLabels, 0.5, 1, cv::THRESH_BINARY);
            float accuracy = cv::countNonZero(predictedLabels == labels) / static_cast<float>(labels.rows);

            std::cout << "Epoch " << i + 1 << ": Accuracy = " << accuracy << std::endl;

            // Logging dimensions for debugging
            // std::cout << "Epoch " << i + 1 << ": Weights updated." << std::endl;
        }
    }

    // FIXTHIS START WRITING TRAINING FUNCTION
    void predict(cv::Mat &data)
    {
        // std::cout << "Predicting using Logistic Regression Model" << std::endl;
        // Ensure data includes the bias term; prepend a column of ones
        dataPreprocessor(data);                       // Preprocess the data
        sigmoid(this->dataWithBias_, this->weights_); // Compute the sigmoid function
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

    // Similarly, create test data from the next 5000 rows, excluding responses 7 and 8
    data.testData = filterData(data.fullData, 1000, 6000);

    // Get the samples and responses
    cv::Mat trainSamples = data.trainData->getTrainSamples();  // Get design matrix
    cv::Mat trainTarget = data.trainData->getTrainResponses(); // Get target values
    cv::Mat testSamples = data.testData->getTrainSamples();    // Get design matrix
    cv::Mat testTarget = data.testData->getTrainResponses();   // Get target values

    // Sanity check if the datas rows and collumns are reasonable
    // std::cout << "Train Samples: " << trainSamples.rows << "x" << trainSamples.cols << std::endl;
    // std::cout << "Train Target: " << trainTarget.rows << "x" << trainTarget.cols << std::endl;
    // std::cout << "Test Samples: " << testSamples.rows << "x" << testSamples.cols << std::endl;
    // std::cout << "Test Target: " << testTarget.rows << "x" << testTarget.cols << std::endl;

    // Sanity check if the data is filtered correctly
    // for (int i = 0; i < trainTarget.rows; ++i)
    // {
    //     float value = trainTarget.at<float>(i, 0);
    //     std::cout << "Test Target [" << i << "]: " << value << std::endl;
    // }

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
    // std::cout << "Training Data: Mean = " << trainMean[0] << ", Stddev = " << trainStddev[0] << std::endl; // Print out the mean and standard deviation for training set
    // std::cout << "Testing Data: Mean = " << testMean[0] << ", Stddev = " << testStddev[0] << std::endl;    // Print out the mean and standard deviation for test set

    cv::PCA pcaTrain1D(trainSamples, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    cv::PCA pcaTest1D(testSamples, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);

    cv::Mat projected1DTrainSamples = pcaTrain1D.project(trainSamples);
    cv::Mat projected1DTestSamples = pcaTest1D.project(testSamples);

    // Visualize 1D PCA Projection
    int width1D = 600, height1D = 600;
    cv::Mat TrainVisualization1D = cv::Mat::zeros(height1D, width1D, CV_8UC3);
    cv::Mat TestVisualization1D = cv::Mat::zeros(height1D, width1D, CV_8UC3);

    // Find the minimum and maximum values in the projected 1D data
    double minVal1D, maxVal1D;
    cv::minMaxLoc(projected1DTrainSamples, &minVal1D, &maxVal1D);
    cv::minMaxLoc(projected1DTestSamples, &minVal1D, &maxVal1D);

    // Visualize the projected 1D training data
    for (int i = 0; i < projected1DTrainSamples.rows; i++)
    {
        float x = (projected1DTrainSamples.at<float>(i, 0) - minVal1D) / (maxVal1D - minVal1D) * (width1D - 40) + 20; // Normalize and scale the x value

        int label = static_cast<int>(trainTarget.at<float>(i, 0));                     // Get the label of the sample
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0); // Set the color based on the label

        cv::circle(TrainVisualization1D, cv::Point(static_cast<int>(x), height1D / 2), 3, color, CV_FILLED); // Draw a circle at the projected point
    }

    // Visualize the projected 1D test data
    for (int i = 0; i < projected1DTestSamples.rows; i++)
    {
        float x = (projected1DTestSamples.at<float>(i, 0) - minVal1D) / (maxVal1D - minVal1D) * (width1D - 40) + 20; // Normalize and scale the x value

        int label = static_cast<int>(testTarget.at<float>(i, 0));                      // Get the label of the sample
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0); // Set the color based on the label

        cv::circle(TestVisualization1D, cv::Point(static_cast<int>(x), height1D / 2), 3, color, CV_FILLED); // Draw a circle at the projected point
    }

    cv::imshow("1D PCA Projection Train", TrainVisualization1D); // Display the 1D PCA projection
    cv::imshow("1D PCA Projection Test", TestVisualization1D);   // Display the 1D PCA projection

    // Perform PCA analysis on the training samples
    cv::PCA pca2DTrainSamples(trainSamples, cv::Mat(), 0, 2);
    cv::PCA pca2DTestSamples(testSamples, cv::Mat(), 0, 2);

    cv::PCA pca3DTrainSamples(trainSamples, cv::Mat(), 0, 3);
    cv::PCA pca3DTestSamples(testSamples, cv::Mat(), 0, 3);

    // Project the training samples onto the PCA space
    cv::Mat projected2DTrainSamples = pca2DTrainSamples.project(trainSamples);
    cv::Mat projected2DTestSamples = pca2DTestSamples.project(testSamples);
    cv::Mat projected3DTrainSamples = pca3DTrainSamples.project(trainSamples);
    cv::Mat projected3DTestSamples = pca3DTestSamples.project(testSamples);

    // Visualize 2D PCA Projection
    int width2D = 600, height2D = 600;
    cv::Mat TrainVisualization2D = cv::Mat::zeros(height2D, width2D, CV_8UC3);
    cv::Mat TestVisualization2D = cv::Mat::zeros(height2D, width2D, CV_8UC3);

    // Find the minimum and maximum values in the projected 2D data
    double minValX2D, maxValX2D, minValY2D, maxValY2D;
    cv::minMaxLoc(projected2DTrainSamples.col(0), &minValX2D, &maxValX2D);
    cv::minMaxLoc(projected2DTrainSamples.col(1), &minValY2D, &maxValY2D);
    cv::minMaxLoc(projected2DTestSamples.col(0), &minValX2D, &maxValX2D);
    cv::minMaxLoc(projected2DTestSamples.col(1), &minValY2D, &maxValY2D);

    // Visualize the projected 2D training data
    for (int i = 0; i < projected2DTrainSamples.rows; i++)
    {
        float x = (projected2DTrainSamples.at<float>(i, 0) - minValX2D) / (maxValX2D - minValX2D) * (width2D - 40) + 20;  // Normalize and scale the x value
        float y = (projected2DTrainSamples.at<float>(i, 1) - minValY2D) / (maxValY2D - minValY2D) * (height2D - 40) + 20; // Normalize and scale the y value

        int label = static_cast<int>(trainTarget.at<float>(i, 0));                     // Get the label of the sample
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0); // Set the color based on the label

        cv::circle(TrainVisualization2D, cv::Point(static_cast<int>(x), static_cast<int>(height2D - y)), 3, color, CV_FILLED); // Draw a circle at the projected point
    }

    // Visualize the projected 2D test data
    for (int i = 0; i < projected2DTestSamples.rows; i++)
    {
        float x = (projected2DTestSamples.at<float>(i, 0) - minValX2D) / (maxValX2D - minValX2D) * (width2D - 40) + 20;  // Normalize and scale the x value
        float y = (projected2DTestSamples.at<float>(i, 1) - minValY2D) / (maxValY2D - minValY2D) * (height2D - 40) + 20; // Normalize and scale the y value

        int label = static_cast<int>(testTarget.at<float>(i, 0));                      // Get the label of the sample
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0); // Set the color based on the label

        cv::circle(TestVisualization2D, cv::Point(static_cast<int>(x), static_cast<int>(height2D - y)), 3, color, CV_FILLED); // Draw a circle at the projected point
    }

    cv::imshow("2D PCA Projection Train", TrainVisualization2D); // Display the 2D PCA projection
    cv::imshow("2D PCA Projection Test", TestVisualization2D);   // Display the 2D PCA projection

    // Visualize 3D PCA Projection (Simulated in 2D)
    int width3D = 600, height3D = 600;
    cv::Mat TrainVisualization3D = cv::Mat::zeros(height3D, width3D, CV_8UC3);
    cv::Mat TestVisualization3D = cv::Mat::zeros(height3D, width3D, CV_8UC3);

    // Find the minimum and maximum values in the projected 3D data
    double minValX3D, maxValX3D, minValY3D, maxValY3D, minValZ3D, maxValZ3D;
    cv::minMaxLoc(projected3DTrainSamples.col(0), &minValX3D, &maxValX3D); // Find the minimum and maximum values in the x column
    cv::minMaxLoc(projected3DTrainSamples.col(1), &minValY3D, &maxValY3D); // Find the minimum and maximum values in the y column
    cv::minMaxLoc(projected3DTrainSamples.col(2), &minValZ3D, &maxValZ3D); // Find the minimum and maximum values in the z column
    cv::minMaxLoc(projected3DTestSamples.col(0), &minValX3D, &maxValX3D);  // Find the minimum and maximum values in the x column
    cv::minMaxLoc(projected3DTestSamples.col(1), &minValY3D, &maxValY3D);  // Find the minimum and maximum values in the y column
    cv::minMaxLoc(projected3DTestSamples.col(2), &minValZ3D, &maxValZ3D);  // Find the minimum and maximum values in the z column

    // Visualize the projected 3D training data
    for (int i = 0; i < projected3DTrainSamples.rows; i++)
    {
        float x = (projected3DTrainSamples.at<float>(i, 0) - minValX3D) / (maxValX3D - minValX3D) * (width3D - 40) + 20;  // Normalize and scale the x value
        float y = (projected3DTrainSamples.at<float>(i, 1) - minValY3D) / (maxValY3D - minValY3D) * (height3D - 40) + 20; // Normalize and scale the y value
        float z = (projected3DTrainSamples.at<float>(i, 2) - minValZ3D) / (maxValZ3D - minValZ3D);                        // Used for depth effect

        int label = static_cast<int>(trainTarget.at<float>(i, 0));                     // Get the label of the sample
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0); // Set the color based on the label
        int radius = 3 + static_cast<int>(z * 10);                                     // Simulate depth by changing the radius based on z value

        cv::circle(TrainVisualization3D, cv::Point(static_cast<int>(x), static_cast<int>(height3D - y)), radius, color, CV_FILLED); // Draw a circle at the projected point
    }

    for (int i = 0; i < projected3DTestSamples.rows; i++)
    {
        float x = (projected3DTestSamples.at<float>(i, 0) - minValX3D) / (maxValX3D - minValX3D) * (width3D - 40) + 20;  // Normalize and scale the x value
        float y = (projected3DTestSamples.at<float>(i, 1) - minValY3D) / (maxValY3D - minValY3D) * (height3D - 40) + 20; // Normalize and scale the y value
        float z = (projected3DTestSamples.at<float>(i, 2) - minValZ3D) / (maxValZ3D - minValZ3D);                        // Used for depth effect

        int label = static_cast<int>(testTarget.at<float>(i, 0));                      // Get the label of the sample
        cv::Scalar color = label == 7 ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0); // Set the color based on the label
        int radius = 3 + static_cast<int>(z * 10);                                     // Simulate depth by changing the radius based on z value

        cv::circle(TestVisualization3D, cv::Point(static_cast<int>(x), static_cast<int>(height3D - y)), radius, color, CV_FILLED); // Draw a circle at the projected point
    }

    cv::imshow("Simulated 3D PCA Projection Train", TrainVisualization3D);
    cv::imshow("Simulated 3D PCA Projection Test", TestVisualization3D);

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

    // Sanity check if the projected data is reasonable
    // std::cout << "Projected 2D Train Samples dimensions: " << projected2DTrainSamples.rows << "x" << projected2DTrainSamples.cols << std::endl;
    // std::cout << "Projected 2D Test Samples dimensions: " << projected2DTestSamples.rows << "x" << projected2DTestSamples.cols << std::endl;
    // std::cout << "Projected 3D Train Samples dimensions: " << projected3DTrainSamples.rows << "x" << projected3DTrainSamples.cols << std::endl;
    // std::cout << "Projected 3D Test Samples dimensions: " << projected3DTestSamples.rows << "x" << projected3DTestSamples.cols << std::endl;

    LogisticRegression model2D(2);
    model2D.predict(projected2DTestSamples);

    // Printing out the first few labels for a sanity check
    // std::cout << "First few labels in the training dataset:" << std::endl;
    // for (int i = 0; i < std::min(trainTarget.rows, 5); ++i)
    // {
    //     float label = trainTarget.at<float>(i, 0);
    //     std::cout << "Label [" << i << "]: " << label << std::endl;
    // }

    // std::cout << "\nFirst few labels in the test dataset:" << std::endl;
    // for (int i = 0; i < std::min(testTarget.rows, 5); ++i)
    // {
    //     float label = testTarget.at<float>(i, 0);
    //     std::cout << "Label [" << i << "]: " << label << std::endl;
    // }

    // Map 7 to 0 and 8 to 1
    cv::Mat trainLabelsBinary = convertLabels(trainTarget); // Convert the training labels to binary
    cv::Mat testLabelsBinary = convertLabels(testTarget);   // Convert the test labels to binary

    // Print out the first few labels for a sanity check
    // std::cout << "First few binary labels in the training dataset:" << std::endl;
    // for (int i = 0; i < std::min(trainLabelsBinary.rows, 5); ++i)
    // {
    //     float label = trainLabelsBinary.at<float>(i, 0);
    //     std::cout << "Label [" << i << "]: " << label << std::endl;
    // }

    // std::cout << "\nFirst few binary labels in the test dataset:" << std::endl;
    // for (int i = 0; i < std::min(testLabelsBinary.rows, 5); ++i)
    // {
    //     float label = testLabelsBinary.at<float>(i, 0);
    //     std::cout << "Label [" << i << "]: " << label << std::endl;
    // }

    model2D.train(5, projected2DTestSamples, testLabelsBinary);
    model2D.predict(projected2DTrainSamples);

    // Accuracy calculation
    cv::Mat predictedLabels;
    cv::threshold(model2D.predictions_, predictedLabels, 0.5, 1, cv::THRESH_BINARY);
    float accuracy = cv::countNonZero(predictedLabels == trainLabelsBinary) / static_cast<float>(trainLabelsBinary.rows);
    std::cout << "Accuracy: " << accuracy << std::endl;

    cv::waitKey(0);

    // FIX MAJOR ISSUE WITH TRAINING
    return 0;
}