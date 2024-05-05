#include "imagePreprocessor.h"

/**
 * @brief Constructs an instance of the imagePreprocessor class.
 *
 * This constructor is called when an object of the imagePreprocessor class is created.
 * It prints a message indicating that the imagePreprocessor object has been created.
 */
imagePreprocessor::imagePreprocessor()
{
    std::cout << "imagePreprocessor created" << std::endl;
}

/**
 * @brief Destructor for the imagePreprocessor class.
 *
 * This destructor is called when an instance of the imagePreprocessor class is destroyed.
 * It prints a message to indicate that the imagePreprocessor object has been destroyed.
 */
imagePreprocessor::~imagePreprocessor()
{
    std::cout << "imagePreprocessor destroyed" << std::endl;
}

/**
 * Captures a frame from the webcam using the provided VideoCapture object.
 *
 * @param cap A pointer to the VideoCapture object representing the webcam.
 * @return The captured frame as a cv::Mat object.
 */
cv::Mat imagePreprocessor::captureWebcam(cv::VideoCapture *cap)
{
    if (!cap->isOpened())
    {
        std::cerr << "Error opening webcam" << std::endl;
        return cv::Mat();
    }

    cv::Mat frame;
    *cap >> frame;
    // std::cout << "Webcam frame captured" << std::endl;
    return frame;
}

/**
 * Initializes the video capture with the specified width and height.
 *
 * @param width The desired width of the captured video frames.
 * @param height The desired height of the captured video frames.
 * @return The initialized cv::VideoCapture object.
 */
cv::VideoCapture imagePreprocessor::initializeVideoCapture(int width, int height)
{
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    return cap;
}

/**
 * Preprocesses the reference image by reading it from the specified file path, resizing it to a standard size,
 * and applying distortion correction using the provided calibration data.
 *
 * @param imagePath The file path of the reference image.
 * @param calibrationPath The file path of the calibration data.
 * @return The preprocessed reference image.
 * @throws std::runtime_error if there is an error reading the reference image.
 */
cv::Mat imagePreprocessor::prepareReferenceImage(const std::string &imagePath, const std::string &calibrationPath)
{
    cv::Mat refImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty())
    {
        throw std::runtime_error("Error reading reference image.");
    }
    cv::resize(refImage, refImage, cv::Size(640, 480));
    return undistortImage(refImage, calibrationPath);
}

/**
 * Displays the matches between two images along with their corresponding keypoints.
 *
 * @param img1 The first input image.
 * @param keypoints1 The keypoints detected in the first image.
 * @param img2 The second input image.
 * @param keypoints2 The keypoints detected in the second image.
 * @param matches The matches between keypoints in the two images.
 */
void imagePreprocessor::displayMatches(const cv::Mat &img1, const std::vector<cv::KeyPoint> &keypoints1,
                                       const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints2,
                                       const std::vector<cv::DMatch> &matches)
{
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    cv::imshow("Matches", imgMatches);
    if (cv::waitKey(30) >= 0)
    {
        throw std::runtime_error("Exit requested.");
    }
}

/**
 * @brief Undistorts an input image using camera calibration parameters.
 *
 * This function takes an input image and undistorts it using camera calibration parameters
 * loaded from a YAML file. The YAML file should contain the camera matrix and distortion
 * coefficients. If the camera matrix and distortion coefficients are found in the file,
 * the input image is undistorted and the undistorted image is returned. If the camera matrix
 * or distortion coefficients are not found in the file, an error message is printed and an
 * empty cv::Mat is returned.
 *
 * @param input The input image to be undistorted.
 * @param pathToCalibrationFile The path to the YAML file containing camera calibration parameters.
 * @return The undistorted image, or an empty cv::Mat if the camera matrix or distortion coefficients
 *         are not found in the file.
 */
cv::Mat imagePreprocessor::undistortImage(cv::Mat &input, std::string pathToCalibrationFile)
{
    try
    {
        YAML::Node config = YAML::LoadFile(pathToCalibrationFile);
        if (config["camera_matrix"] && config["distortion_coefficients"])
        {
            std::vector<double> cameraMatrixData = config["camera_matrix"]["data"].as<std::vector<double>>();
            std::vector<double> distCoeffsData = config["distortion_coefficients"]["data"].as<std::vector<double>>();

            cv::Mat cameraMatrix = cv::Mat(cameraMatrixData, true).reshape(0, 3);                 // reshaping to 3x3 matrix
            cv::Mat distCoeffs = cv::Mat(distCoeffsData, true).reshape(0, distCoeffsData.size()); // reshaping if needed

            cv::Mat undistortedInput;
            cv::undistort(input, undistortedInput, cameraMatrix, distCoeffs);
            // std::cout << "Undistorted image created" << std::endl;
            return undistortedInput;
        }
        else
        {
            std::cerr << "Camera matrix or distortion coefficients not found in file" << std::endl;
        }
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "Failed to parse " << pathToCalibrationFile << ": " << e.what() << std::endl;
    }
    return cv::Mat();
}

/**
 * Detects keypoints and computes descriptors using the SIFT algorithm.
 *
 * @param input The input image.
 * @param mask The optional mask specifying where to look for keypoints.
 * @param nFeatures The number of desired features.
 * @param nOctaveLayers The number of layers in each octave of the image pyramid.
 * @param contrastThreshold The contrast threshold used to filter out weak keypoints.
 * @param edgeThreshold The threshold used to filter out edge keypoints.
 * @param sigma The sigma value for Gaussian smoothing of the input image.
 * @return A structure containing the detected keypoints and computed descriptors.
 */
KeypointsAndDescriptors imagePreprocessor::siftDetect(cv::Mat &input, cv::Mat &mask, int nFeatures, int nOctaveLayers,
                                                      double contrastThreshold, double edgeThreshold, double sigma) const
{
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detect(input, keypoints, mask);
    detector->compute(input, keypoints, descriptors);

    // std::cout << "Detected " << keypoints.size() << " keypoints and computed descriptors." << std::endl;
    // std::cout << "SIFT detection completed" << std::endl;
    return {keypoints, descriptors};
}

/**
 * Creates a rectangle mask with the specified image size and rectangle size.
 *
 * @param imageSize The size of the image.
 * @param rectSize The size of the rectangle.
 * @return The created rectangle mask.
 */
cv::Mat imagePreprocessor::createRectangleMask(const cv::Size &imageSize, const cv::Size &rectSize)
{
    cv::Mat mask = cv::Mat::zeros(imageSize, CV_8U); // Create a black mask

    // Calculate the top-left corner to center the rectangle
    int x = (imageSize.width - rectSize.width) / 2;
    int y = (imageSize.height - rectSize.height) / 2;
    cv::Point topLeft(x, y);

    // Draw a white rectangle centered within the mask
    cv::rectangle(mask, cv::Rect(topLeft, rectSize), cv::Scalar(255), cv::FILLED);
    return mask;
}

/**
 * Filters the given keypoints and descriptors based on the provided indices.
 *
 * @param KpAndDesc The keypoints and descriptors to be filtered.
 * @param indices The indices of the keypoints and descriptors to be included in the filtered result.
 * @return A new KeypointsAndDescriptors object containing the filtered keypoints and descriptors.
 */
KeypointsAndDescriptors imagePreprocessor::filterKeypointsAndDescriptors(KeypointsAndDescriptors &KpAndDesc, const std::vector<int> &indices)
{
    std::vector<cv::KeyPoint> filteredKeypoints;
    cv::Mat filteredDescriptors;

    for (size_t index : indices)
    {
        if (index >= 0 && index < KpAndDesc.first.size())
        {
            filteredKeypoints.push_back(KpAndDesc.first[index]);
            filteredDescriptors.push_back(KpAndDesc.second.row(index));
        }
    }

    std::cout << "Filtered " << filteredKeypoints.size() << " keypoints and descriptors." << std::endl;
    return {filteredKeypoints, filteredDescriptors};
}

/**
 * Draws keypoints on the input image and returns the resulting image.
 *
 * @param input The input image on which keypoints will be drawn.
 * @param keypoints The keypoints to be drawn on the input image.
 * @return The image with keypoints drawn.
 */
cv::Mat imagePreprocessor::drawKeypoints(cv::Mat &input, std::vector<cv::KeyPoint> &keypoints) const
{
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    std::cout << "Keypoints drawn" << std::endl;
    return output;
}

/**
 * Writes the keypoints and descriptors to a CSV file.
 *
 * @param path The path to the output CSV file.
 * @param keypoints The vector of keypoints.
 * @param descriptors The matrix of descriptors.
 */
void imagePreprocessor::keypointsAndDescriptorsToCSV(std::string path, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    // Write header
    file << "X,Y,Size,Angle,Response,Octave,ClassID,";
    for (int j = 0; j < descriptors.cols; j++)
    {
        file << "D" << j << ",";
    }
    file << std::endl;

    // Write data
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        file << keypoints[i].pt.x << "," << keypoints[i].pt.y << "," << keypoints[i].size << "," << keypoints[i].angle << "," << keypoints[i].response << "," << keypoints[i].octave << "," << keypoints[i].class_id << ",";
        for (int j = 0; j < descriptors.cols; j++)
        {
            file << descriptors.at<float>(i, j) << ",";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Keypoints and descriptors written to " << path << std::endl;
}

/**
 * Calculates the variance of the SIFT feature movement based on a vector of points.
 *
 * @param points A vector of 2D points representing the SIFT feature movement.
 * @return The variance of the SIFT feature movement.
 */
double imagePreprocessor::calculateSiftFeatureMovementVariance(const std::vector<cv::Point2f> &points)
{
    if (points.size() < 2)
        return 0.0;
    cv::Point2f sum(0, 0);
    for (const auto &point : points)
    {
        sum += point;
    }
    cv::Point2f mean = sum * (1.0 / points.size());
    double variance = 0.0;
    for (const auto &point : points)
    {
        cv::Point2f diff = point - mean;
        variance += diff.x * diff.x + diff.y * diff.y;
    }
    return variance / points.size();
}

/**
 * Updates the feature tracks and counts based on the given matches and current keypoints.
 *
 * @param matches The vector of matches between reference and current keypoints.
 * @param currKeypoints The vector of current keypoints.
 * @param featureMatchCount The map of feature indices and their match counts.
 * @param featureTracks The map of feature indices and their corresponding tracks.
 */
void imagePreprocessor::updateFeatureTracksAndCounts(const std::vector<cv::DMatch> &matches,
                                                     const std::vector<cv::KeyPoint> &currKeypoints,
                                                     std::map<int, int> &featureMatchCount,
                                                     std::map<int, std::vector<cv::Point2f>> &featureTracks)
{
    for (const auto &match : matches)
    {
        int idx = match.queryIdx; // Index of the feature in the reference image
        featureMatchCount[idx]++;
        featureTracks[idx].push_back(currKeypoints[match.trainIdx].pt);
    }
}

/**
 * Saves the feature tracks to a CSV file.
 *
 * @param filePath The path to the CSV file.
 * @param featureMatchCount A map containing the match count for each feature index.
 * @param featureTracks A map containing the feature tracks for each feature index.
 * @param minMatches The minimum number of matches required for a feature track to be saved.
 */
void imagePreprocessor::saveFeatureTracksToCSV(const std::string &filePath,
                                               const std::map<int, int> &featureMatchCount,
                                               const std::map<int, std::vector<cv::Point2f>> &featureTracks,
                                               int minMatches) {
    std::ofstream outFile(filePath);
    if (!outFile) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    int countSaved = 0;
    outFile << "FeatureIndex,MatchCount,Variance\n";
    for (const auto &track : featureTracks) {
        int count = featureMatchCount.at(track.first);
        if (count >= minMatches) {
            double variance = calculateSiftFeatureMovementVariance(track.second);
            outFile << track.first << "," << count << "," << variance << "\n";
            countSaved++;
        }
    }
    outFile.close();
    std::cout << "Data saved to " << filePath << " with " << countSaved << " entries." << std::endl;
}

