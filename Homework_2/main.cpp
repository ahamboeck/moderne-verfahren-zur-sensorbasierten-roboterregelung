#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

/**
 * Logistic Regression class for binary classification.
 *
 * This class implements logistic regression, a popular algorithm for binary classification.
 * It uses the sigmoid function to compute the probability that each instance belongs to the positive class.
 * The class provides methods for training the model, making predictions, and evaluating the model's performance.
 */
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

    /**
     * Preprocesses the given data matrix by adding a column of ones to the original data matrix.
     * This is done by concatenating a matrix of ones with the original data matrix horizontally.
     *
     * @param data The input data matrix.
     * @return The preprocessed data matrix with a column of ones added.
     */
    cv::Mat dataPreprocessor(const cv::Mat &data)
    {
        cv::Mat dataWithBias;
        cv::hconcat(cv::Mat::ones(data.rows, 1, data.type()), data, dataWithBias); // Concatenate the column of ones to the original data matrix
        return dataWithBias;
    }

    /**
     * Trains a model using the specified number of epochs and training data.
     *
     * @param epochs The number of training epochs.
     * @param trainData The training data matrix.
     * @param trainLabels The training labels matrix.
     * @param testData The test data matrix.
     * @param testLabels The test labels matrix.
     */
    void train(int epochs, cv::Mat &trainData, cv::Mat &trainLabels, cv::Mat &testData, cv::Mat &testLabels)
    {
        for (int i = 0; i < epochs; ++i)
        {
            // Preprocess training data to include a bias term or other preprocessing steps
            cv::Mat trainDataWithBias = dataPreprocessor(trainData);

            // Obtain predictions for the training data using the current model weights
            // 'predict' function likely applies a sigmoid function on linear combinations of input features and weights
            // to output probabilities (p) for each instance being in the positive class
            cv::Mat trainPredictions = predict(trainDataWithBias);

            // Construct the diagonal weight matrix W for use in the IRLS algorithm
            // This matrix is used to weight instances differently based on the current model predictions
            // It's diagonal because each instance's weight is independent and affects only that instance's contribution
            cv::Mat W = cv::Mat::zeros(trainData.rows, trainData.rows, CV_32F);
            for (int j = 0; j < trainData.rows; ++j)
            {
                float p = trainPredictions.at<float>(j, 0); // p is the predicted probability for instance j
                W.at<float>(j, j) = p * (1 - p);            // Weight calculation, derivative of the logistic function, representing the variance of the prediction
            }

            // Calculate the Hessian matrix and the gradient vector for the weight update step
            // These calculations are central to the Newton-Raphson method used in IRLS for finding weight updates
            cv::Mat Hessian = trainDataWithBias.t() * W * trainDataWithBias;             // Hessian matrix calculation
            cv::Mat gradient = trainDataWithBias.t() * (trainPredictions - trainLabels); // Gradient calculation, difference between predictions and actual labels

            cv::Mat HessianInv;
            // Invert the Hessian matrix for the update calculation
            // DECOMP_SVD is used for inversion to ensure better numerical stability, especially important if Hessian is near-singular
            cv::invert(Hessian, HessianInv, cv::DECOMP_SVD);

            // Update the model weights using the inverted Hessian and the gradient
            // This step is the core of the Newton-Raphson update in the IRLS method
            weights_ -= HessianInv * gradient;

            // For monitoring purposes, calculate and print the accuracy on the test data
            // Preprocess the test data similarly to the training data
            cv::Mat testDataWithBias = dataPreprocessor(testData);

            // Obtain predictions for the test data using the updated model weights
            cv::Mat testPredictions = predict(testDataWithBias);

            // Convert predicted probabilities into binary predictions based on a 0.5 threshold
            // Predictions above this threshold are considered positive (1), and below are negative (0)
            cv::Mat predictedTestLabels;
            cv::threshold(testPredictions, predictedTestLabels, 0.5, 1, cv::THRESH_BINARY);

            // Calculate the test accuracy as the proportion of correctly predicted labels
            float testAccuracy = cv::countNonZero(predictedTestLabels == testLabels) / static_cast<float>(testLabels.rows);

            // Output the current epoch's test accuracy to monitor the model's performance over iterations
<<<<<<< HEAD
            std::cout << "Epoch: " << i << " Test Accuracy: " << testAccuracy << std::endl;
=======
            std::cout << "Iteration: " << i << " Test Accuracy: " << testAccuracy << std::endl;
>>>>>>> 80623c2add7a1b5973ad037ff12f1a70e0a6e57a
        }
    }

    /**
     * Predicts the probability that each instance belongs to the positive class.
     *
     * @param dataWithBias The input data with bias.
     * @return The computed probabilities as a cv::Mat object.
     */
    cv::Mat predict(const cv::Mat &dataWithBias)
    {
        // Call the sigmoid function with the current data and model weights
        // This computes the probability that each instance belongs to the positive class
        return sigmoid(dataWithBias, this->weights_);
    }

    /**
     * Computes the sigmoid function for each element of the input matrix.
     *
     * The sigmoid function is defined as: 1 / (1 + exp(-z)), where z is the input value.
     * This function calculates the sigmoid function for each element of the input matrix.
     *
     * @param dataWithBias The input matrix with bias term.
     * @param weights The weight matrix.
     * @return A matrix where each element is the sigmoid of the corresponding weighted sum.
     */
    cv::Mat sigmoid(const cv::Mat &dataWithBias, const cv::Mat &weights)
    {
        cv::Mat weightedSums = dataWithBias * weights; // Compute weighted sums of the features (linear combination)

        cv::Mat exponendOfWeightedSums;
        // Apply the exponential function to the negative of the weighted sums
        // This is part of the sigmoid function's formula, where we calculate exp(-z) for each z in weighted sums
        cv::exp(-weightedSums, exponendOfWeightedSums);

        // Compute the sigmoid function: 1 / (1 + exp(-z)) for each z
        // This results in a matrix where each element is the sigmoid of the corresponding weighted sum
        return 1.0 / (1.0 + exponendOfWeightedSums);
    }
};

/**
 * Converts the labels in the given cv::Mat object to binary labels.
 *
 * @param originalLabels The original labels to be converted.
 * @return The converted binary labels.
 */
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

/**
 * Standardizes the given matrix of data.
 *
 * @param data The matrix of data to be standardized.
 */
void standardize(cv::Mat &data)
{
    // Ensure the data is in floating-point format to prevent integer truncation during division
    data.convertTo(data, CV_32F);

    // Calculate the global mean and standard deviation across all pixels and images
    cv::Scalar mean, stddev;
    cv::meanStdDev(data, mean, stddev);

    // Check for a non-zero standard deviation to avoid division by zero
    if (stddev[0] != 0)
    {
        // Standardize the data
        data = (data - mean[0]) / stddev[0];
    }
}

struct Data
/**
 * @brief Holds the complete dataset, training dataset, and test dataset.
 */
{
    cv::Ptr<cv::ml::TrainData> fullData;  // Holds the complete dataset
    cv::Ptr<cv::ml::TrainData> trainData; // Holds the training dataset
    cv::Ptr<cv::ml::TrainData> testData;  // Holds the test dataset
};

/**
 * @brief A smart pointer to an instance of cv::ml::TrainData.
 *
 * This smart pointer manages the lifetime of the cv::ml::TrainData object and provides
 * convenient access to its member functions and data. It ensures that the object is
 * properly deallocated when it is no longer needed.
 */
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

    // Print the number of 7 and 8 in the training data for a sanity check
    cv::Mat trainResponses = data.trainData->getResponses();
    // int count7 = cv::countNonZero(trainResponses == 7);
    // int count8 = cv::countNonZero(trainResponses == 8);
    // std::cout << "Number of 7s in the training data: " << count7 << std::endl;
    // std::cout << "Number of 8s in the training data: " << count8 << std::endl;

    // Similarly, create test data from the next 5000 rows, excluding responses 7 and 8
    data.testData = filterData(data.fullData, 1000, 6000);

    // Print the number of 7 and 8 in the test data for a sanity check
    cv::Mat testResponses = data.testData->getResponses();
    // count7 = cv::countNonZero(testResponses == 7);
    // count8 = cv::countNonZero(testResponses == 8);
    // std::cout << "Number of 7s in the test data: " << count7 << std::endl;
    // std::cout << "Number of 8s in the test data: " << count8 << std::endl;

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

    // std::cout << "Train Mean: " << trainMean[0] << " Train Stddev: " << trainStddev[0] << std::endl;
    // std::cout << "Test Mean: " << testMean[0] << " Test Stddev: " << testStddev[0] << std::endl;

    // Perform PCA on the training and test samples for all dimensions
    cv::PCA pcaTrainAllDimensions(trainSamples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::PCA pcaTestAllDimensions(testSamples, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Save the eigenvalues to a CSV file
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

    /*
    For the amount of principal components to keep, I choose 1 since the model performed best with 1 principal component.
    In theory I would have chosen 85 since it is the number of dimensions that explain 95% of the variance which is a common threshold
    when performing PCA. A picture which shows the PCA explained variance ratio is attached in the moodle submission..
    */
    cv::PCA pcaTrain1D(trainSamples, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    cv::PCA pcaTest1D(testSamples, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);

    cv::Mat projected1DTrainSamples = pcaTrain1D.project(trainSamples);
    cv::Mat projected1DTestSamples = pcaTest1D.project(testSamples);

    // Create and train a logistic regression model with 1 principal component
    LogisticRegression model1D(1);

    // Map 7 to 0 and 8 to 1
    cv::Mat trainLabelsBinary = convertLabels(trainTarget); // Convert the training labels to binary
    cv::Mat testLabelsBinary = convertLabels(testTarget);   // Convert the test labels to binary

    /*
     So.. for the training it appears that 1 training cycle is preforming best. This could be due the data sample size
     relativly small or because the task is not complex enough to require more training cycles.
    */
    model1D.train(10, projected1DTrainSamples, trainLabelsBinary, projected1DTestSamples, testLabelsBinary);

    cv::waitKey(0);

    return 0;
}