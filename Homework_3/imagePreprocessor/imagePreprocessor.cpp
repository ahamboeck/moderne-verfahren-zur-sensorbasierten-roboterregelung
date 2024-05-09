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
 * Initializes the video capture from either a webcam or a video file.
 *
 * @param width The desired width of the captured video frames.
 * @param height The desired height of the captured video frames.
 * @param videoFilePath The path to a video file to read from. If empty, the default webcam (device index 0) will be used.
 * @return The initialized cv::VideoCapture object, or a non-opened object if initialization fails.
 */
cv::VideoCapture imagePreprocessor::initializeVideoCapture(int width, int height, const std::string &videoFilePath)
{
    cv::VideoCapture cap;

    if (videoFilePath.empty())
    {
        // Open the default webcam (device index 0)
        cap.open(0);
    }
    else
    {
        // Open the provided video file
        cap.open(videoFilePath);
    }

    // Verify that the capture object has been successfully opened
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera or video file" << std::endl;
        return cv::VideoCapture(); // Return a non-opened VideoCapture object
    }

    // Set the desired width and height properties (works for webcams only)
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
 * @param width The desired width of the reference image.
 * @param height The desired height of the reference image.
 * @return The preprocessed reference image.
 * @throws std::runtime_error if there is an error reading the reference image.
 */
cv::Mat imagePreprocessor::prepareReferenceImage(const std::string &imagePath, const std::string &calibrationPath, int width, int height)
{
    cv::Mat refImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty())
    {
        throw std::runtime_error("Error reading reference image.");
    }
    cv::resize(refImage, refImage, cv::Size(width, height));
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
            this->cameraMatrix_ = cameraMatrix.clone();
            this->distCoeffs_ = distCoeffs.clone();
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
 * Reads a list of indices from a CSV file.
 *
 * @param filepath The path to the CSV file.
 * @return A vector containing the indices read from the file.
 */
std::vector<int> imagePreprocessor::readIndicesFromCSV(const std::string &filepath)
{
    std::vector<int> indices;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open indices file: " << filepath << std::endl;
        return indices;
    }

    std::string line;
    while (std::getline(file, line))
    {
        try
        {
            indices.push_back(std::stoi(line));
        }
        catch (const std::exception &e)
        {
            std::cerr << "Invalid index in CSV: " << line << std::endl;
        }
    }

    return indices;
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
    file << "Index,X,Y,Size,Angle,Response,Octave,ClassID,";
    for (int j = 0; j < descriptors.cols; j++)
    {
        file << "D" << j << ",";
    }
    file << std::endl;

    // Write data
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        file << i << "," << keypoints[i].pt.x << "," << keypoints[i].pt.y << "," << keypoints[i].size << "," << keypoints[i].angle << "," << keypoints[i].response << "," << keypoints[i].octave << "," << keypoints[i].class_id << ",";
        for (int j = 0; j < descriptors.cols; j++)
        {
            file << descriptors.at<float>(i, j) << ",";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Keypoints and descriptors with indices written to " << path << std::endl;
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

    cv::Point2f mean(0, 0);
    double m2 = 0.0;
    int n = 0;

    for (const auto &point : points)
    {
        n++;
        cv::Point2f delta = point - mean;
        mean += delta / n;
        cv::Point2f delta2 = point - mean;
        m2 += delta.x * delta2.x + delta.y * delta2.y;
    }

    return m2 / n;
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
    std::cout << "Processing " << matches.size() << " matches." << std::endl;
    for (const auto &match : matches)
    {
        int idx = match.queryIdx; // Index of the feature in the reference image
        featureMatchCount[idx]++;
        featureTracks[idx].push_back(currKeypoints[match.trainIdx].pt);
        // Debugging output for each match
        // std::cout << "Feature Index: " << idx
        //           << " | Current Match Count: " << featureMatchCount[idx]
        //           << " | Current Point: " << currKeypoints[match.trainIdx].pt << std::endl;
    }
}

/**
 * Saves the feature tracks with extended metrics to a CSV file.
 * This includes feature index, match count, variance, max displacement, and average displacement.
 *
 * @param filePath The path to the CSV file.
 * @param featureMatchCount A map containing the match count for each feature index.
 * @param featureTracks A map containing the feature tracks for each feature index.
 * @param minMatches The minimum number of matches required for a feature track to be saved.
 */
void imagePreprocessor::saveFeatureTracksToCSV(const std::string &filePath,
                                               const std::map<int, int> &featureMatchCount,
                                               const std::map<int, std::vector<cv::Point2f>> &featureTracks,
                                               int minMatches)
{
    std::ofstream outFile(filePath);
    if (!outFile)
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    // Write the header for the CSV file
    outFile << "FeatureIndex,MatchCount,Variance,MaxDisplacement,AverageDisplacement\n";
    int countSaved = 0;

    for (const auto &track : featureTracks)
    {
        int count = featureMatchCount.at(track.first);
        if (count >= minMatches)
        {
            double variance = calculateSiftFeatureMovementVariance(track.second);
            double maxDisplacement = 0.0;
            double totalDisplacement = 0.0;
            cv::Point2f initialPoint = track.second.front();

            for (const auto &point : track.second)
            {
                double displacement = cv::norm(point - initialPoint);
                maxDisplacement = std::max(maxDisplacement, displacement);
                totalDisplacement += displacement;
            }
            double averageDisplacement = totalDisplacement / track.second.size();

            // Write the computed data to the CSV file
            outFile << track.first << "," << count << "," << variance << "," << maxDisplacement << "," << averageDisplacement << "\n";
            countSaved++;
        }
    }

    outFile.close();
    std::cout << "Extended feature track data saved to " << filePath << " with " << countSaved << " entries." << std::endl;
}

/**
 * Reads the top N features from a file and returns their indices based on a specific sorting criterion.
 *
 * @param filepath The path to the file containing the features.
 * @param topN The number of top features to retrieve. Default value is 15.
 * @param criteria The criteria by which to sort the features.
 * @return A vector of integers representing the indices of the top features.
 */
std::vector<int> imagePreprocessor::readTopFeatures(const std::string &filepath, int topN, SortCriteria criteria)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return {};
    }

    // Store index, match count, variance, max displacement, and average displacement
    std::vector<std::tuple<int, int, double, double, double>> features;
    std::string line;
    std::getline(file, line); // Skip the header

    // Read each line and parse feature details
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string idx, count, var, maxDisplacement, avgDisplacement;
        std::getline(iss, idx, ',');
        std::getline(iss, count, ',');
        std::getline(iss, var, ',');
        std::getline(iss, maxDisplacement, ',');
        std::getline(iss, avgDisplacement, ',');

        // Add the parsed data to the feature list
        features.emplace_back(std::stoi(idx), std::stoi(count), std::stod(var), std::stod(maxDisplacement), std::stod(avgDisplacement));
    }

    // Lambda comparator to sort by the selected criteria
    auto compare = [criteria](const auto &a, const auto &b)
    {
        switch (criteria)
        {
        case SortCriteria::MatchCount:
            return std::get<1>(a) < std::get<1>(b);
        case SortCriteria::Variance:
            return std::get<2>(a) < std::get<2>(b);
        case SortCriteria::MaxDisplacement:
            return std::get<3>(a) < std::get<3>(b);
        case SortCriteria::AverageDisplacement:
            return std::get<4>(a) < std::get<4>(b);
        default:
            return true; // Default comparison (won't actually be used)
        }
    };

    // Sort based on the chosen criteria
    std::sort(features.begin(), features.end(), compare);

    // Extract the top N indices
    std::vector<int> indices;
    for (int i = 0; i < topN && i < static_cast<int>(features.size()); ++i)
    {
        indices.push_back(std::get<0>(features[i])); // Get the index of the feature
    }

    // Print the indices
    std::cout << "Top " << indices.size() << " feature indices based on the selected criterion: ";
    for (int index : indices)
    {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    return indices;
}

std::map<int, cv::Point3f> imagePreprocessor::load3DPoints(const std::string &filepath) {
    std::map<int, cv::Point3f> indexedPoints;
    std::ifstream file(filepath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return indexedPoints; // Return empty map if file cannot be opened
    }

    // Read the header line first and ignore it
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        std::string cell;
        std::vector<float> parsedRow;

        // Parse each line by commas
        while (getline(linestream, cell, ',')) {
            try {
                parsedRow.push_back(std::stof(cell)); // Convert string to float and add to the row
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Invalid number found in file: " << ia.what() << '\n';
                continue;
            }
        }

        // Create a Point3f from the parsed row (assuming columns are: Index, X, Y, Z)
        if (parsedRow.size() >= 4) {  // Check if there are enough elements (index + coordinates)
            int index = static_cast<int>(parsedRow[0]);  // Convert float to int for the index
            cv::Point3f point(parsedRow[1], parsedRow[2], parsedRow[3]);
            indexedPoints[index] = point;  // Use index as the key
        }
    }

    file.close();
    return indexedPoints;
}


cv::Mat imagePreprocessor::getCameraMatrix() const
{
    return this->cameraMatrix_;
}

cv::Mat imagePreprocessor::getDistCoeffs() const
{
    return this->distCoeffs_;
}
