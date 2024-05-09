#include "../../imagePreprocessor/imagePreprocessor.h"

int main(int argc, char **argv)
{
    // SIFT feature detection parameters
    int nFeatures = 150;             // Number of features to detect
    int nOctaveLayers = 3;           // Number of octave layers within each octave
    double contrastThreshold = 0.06; // Threshold to filter out weak features in low-contrast regions
    int edgeThreshold = 10;          // Threshold to filter out edge-like features
    double sigma = 1.5;              // The sigma of the Gaussian applied to the input image at the octave #0

    // Parameters for feature matching from CSV
    int featuresFromCSV = 50;      // Number of features to use from CSV
    int matchCountThreshold = 500; // Matching count threshold for feature tracking

    // Determine the operation mode based on command line arguments
    std::string mode = (argc > 1) ? argv[1] : "filter";
    std::cout << "Running in mode: " << mode << std::endl;

    // Paths for data and configuration files
    std::string imagePath = "../../data/original/dragonball_new_setup.jpg";
    std::string calibrationFilePath = "../../camera_calib_data/calib_v0.5/ost.yaml";
    std::string bestFeaturesPath = "../../data/feature_subsets/matched_features.csv";
    std::string allFeaturesCSVPath = "../../data/feature_subsets/keypoints_and_descriptors.csv";
    std::string filteredIndicesPath = "../../data/feature_subsets/activeSet.csv";
    std::string filteredIndicesXYZCoordinatesPath = "../../data/feature_subsets/activeSet_XYZ.csv";
    std::string videoPath = "../../data/video/dragonball_video_noaudio.webm";

    std::vector<int> filteredIndices; // Container for filtered indices

    imagePreprocessor processor; // Image processing object

    // Load 3D points from CSV
    std::map<int, cv::Point3f> indexedPoints = processor.load3DPoints(filteredIndicesXYZCoordinatesPath);

    // Initialize video capture at specified resolution
    auto cap = processor.initializeVideoCapture(1280, 720, videoPath);

    // Prepare the reference image with specified parameters
    cv::Mat refImage = processor.prepareReferenceImage(imagePath, calibrationFilePath, 1280, 720);
    if (refImage.empty())
    {
        std::cerr << "Error reading reference image.\n";
        return -1;
    }

    // Create a rectangular mask to limit feature detection to specific areas
    cv::Mat mask = processor.createRectangleMask(refImage.size(), cv::Size(1280, 720));

    // Detect keypoints and descriptors using SIFT
    auto kpAndDesc = processor.siftDetect(refImage, mask, nFeatures, nOctaveLayers,
                                          contrastThreshold, edgeThreshold, sigma);

    // Save detected keypoints and descriptors to CSV
    processor.keypointsAndDescriptorsToCSV(allFeaturesCSVPath, kpAndDesc.first, kpAndDesc.second);

    // Setup the display window
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 3000, 2000);

    std::vector<cv::KeyPoint> keypointsToUse = kpAndDesc.first; // Keypoints to use
    cv::Mat descriptorsToUse = kpAndDesc.second;                // Descriptors of keypoints

    if (mode == "use")
    {
        std::cout << "Using the best matches from CSV file." << std::endl;
        std::vector<int> bestIndices = processor.readTopFeatures(bestFeaturesPath, featuresFromCSV, SortCriteria::AverageDisplacement);

        // Filter keypoints and descriptors based on the best indices
        auto filteredKpAndDesc = processor.filterKeypointsAndDescriptors(kpAndDesc, bestIndices);
        keypointsToUse = filteredKpAndDesc.first;
        descriptorsToUse = filteredKpAndDesc.second;

        // Label keypoints on the reference image
        for (size_t i = 0; i < keypointsToUse.size(); ++i)
        {
            cv::Point2f position = keypointsToUse[i].pt;
            std::string label = std::to_string(bestIndices[i]);
            cv::putText(refImage, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    else if (mode == "filter")
    {
        std::cout << "Filtering keypoints and descriptors using CSV file." << std::endl;
        filteredIndices = processor.readIndicesFromCSV(filteredIndicesPath);
        auto filteredKpAndDesc = processor.filterKeypointsAndDescriptors(kpAndDesc, filteredIndices);
        keypointsToUse = filteredKpAndDesc.first;
        descriptorsToUse = filteredKpAndDesc.second;

        // Label keypoints on the reference image
        for (size_t i = 0; i < keypointsToUse.size(); ++i)
        {
            cv::Point2f position = keypointsToUse[i].pt;
            std::string label = std::to_string(filteredIndices[i]);
            cv::putText(refImage, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    else
    {
        // Default behavior, use all detected keypoints and descriptors
        keypointsToUse = kpAndDesc.first;
        descriptorsToUse = kpAndDesc.second;
    }

    std::map<int, int> featureMatchCount;                  // Counts of matches per feature
    std::map<int, std::vector<cv::Point2f>> featureTracks; // Tracks of feature points

    // Output loaded indices for matching
    std::cout << "Loaded keypoints indices for matching: ";
    for (const auto &index : filteredIndices)
    {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    cv::BFMatcher matcher(cv::NORM_L2, true); // Brute Force Matcher with cross-check

    bool userExit = false; // Flag to control exit condition
    while (!userExit)      // Main processing loop
    {
        cv::Mat frame = processor.captureWebcam(&cap); // Capture frame from webcam
        if (frame.empty())                             // Check for failed capture
        {
            std::cout << "Failed to read frame from camera.\n";
            break;
        }
        std::vector<cv::Point2f> points2D;                                                  // Container for 2D points
        std::vector<cv::Point3f> points3D;                                                  // Container for 3D points
        std::vector<cv::DMatch> matches;                                                    // Container for matches
        cv::Mat currGray = processor.undistortImage(frame, calibrationFilePath);            // Undistort the current frame
        auto currKpAndDesc = processor.siftDetect(currGray, mask, nFeatures, nOctaveLayers, // Detect keypoints and descriptors in the current frame
                                                  contrastThreshold, edgeThreshold, sigma);

        // Draw coordinate frame lines
        cv::Point center(frame.cols / 2, frame.rows / 2);
        int axisLength = 50;                                                                                  // Length of the axis lines
        cv::arrowedLine(frame, center, cv::Point(center.x + axisLength, center.y), cv::Scalar(0, 0, 255), 2); // X-axis in red
        cv::arrowedLine(frame, center, cv::Point(center.x, center.y + axisLength), cv::Scalar(0, 255, 0), 2); // Y-axis in green

        cv::imshow("Live Video with Coordinate Frame", frame); // Display the frame

        matcher.match(descriptorsToUse, currKpAndDesc.second, matches); // Match descriptors between reference and current frame

        std::vector<cv::DMatch> goodMatches; // Container for good matches
        for (auto &match : matches)          // Filter matches based on distance
        {
            if (match.distance < 150)
            {
                goodMatches.push_back(match);
            }
        }
        processor.updateFeatureTracksAndCounts(matches, currKpAndDesc.first, featureMatchCount, featureTracks); // Update feature tracks and counts

        if (mode == "filter")
        {
            processor.displayMatches(refImage, keypointsToUse, currGray, currKpAndDesc.first, matches); // Display matches if in filter mode

            for (const auto &match : goodMatches) // Check for matched keypoints in the list of filtered indices
            {
                if (indexedPoints.count(filteredIndices[match.queryIdx]))
                {
                    points2D.push_back(currKpAndDesc.first[match.trainIdx].pt);         // Add 2D point to list
                    points3D.push_back(indexedPoints[filteredIndices[match.queryIdx]]); // Add corresponding 3D point to list
                }
            }

            if (points3D.size() >= 4) // Check if there are enough points to perform solvePnP
            {
                cv::Mat rvec, tvec; // Rotation and translation vectors
                if (cv::solvePnP(points3D, points2D, processor.getCameraMatrix(), processor.getDistCoeffs(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE))
                {
                    cv::Mat rotationMatrix;                                                        // Rotation matrix
                    cv::Rodrigues(rvec, rotationMatrix);                                           // Convert rotation vector to matrix
                    std::cout << "Translation in X in mm: " << tvec.at<double>(0, 0) << std::endl; // Output X translation
                    std::cout << "Translation in Y in mm: " << tvec.at<double>(1, 0) << std::endl; // Output Y translation
                    std::cout << "Translation in Z in mm: " << tvec.at<double>(2, 0) << std::endl; // Output Z translation
                }
            }
        }
        processor.displayMatches(refImage, keypointsToUse, currGray, currKpAndDesc.first, matches); // Display matches in the current frame

        int key = cv::waitKey(10); // Wait for a key press for 10 ms
        if ((key & 0xFF) == 'q')   // Check if 'q' was pressed
        {
            std::cout << "Exit requested by user. Key: " << key << std::endl;
            if (mode == "save")
            {
                std::cout << "Saving data to CSV." << std::endl;
                processor.saveFeatureTracksToCSV(bestFeaturesPath, featureMatchCount, featureTracks, matchCountThreshold); // Save feature tracks to CSV
            }
            userExit = true; // Set exit flag to true
        }
    }

    cv::destroyAllWindows(); // Destroy all OpenCV windows
    return 0;                // Return success
}
