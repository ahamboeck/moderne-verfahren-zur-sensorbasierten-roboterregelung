#include "../../imagePreprocessor/imagePreprocessor.h"

int main(int argc, char **argv)
{

    int nFeatures = 150;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.06;
    int edgeThreshold = 10;
    double sigma = 1.5;

    int featuresFromCSV = 50;
    int matchCountThreshold = 500;

    std::string mode = (argc > 1) ? argv[1] : "filter";
    std::cout << "Running in mode: " << mode << std::endl;

    std::string imagePath = "../../data/original/dragonball_new_setup.jpg";
    std::string calibrationFilePath = "../../camera_calib_data/calib_v0.5/ost.yaml";
    std::string bestFeaturesPath = "../../data/feature_subsets/matched_features.csv";
    std::string allFeaturesCSVPath = "../../data/feature_subsets/keypoints_and_descriptors.csv";
    std::string filteredIndicesPath = "../../data/feature_subsets/activeSet.csv";
    std::string filteredIndicesXYZCoordinatesPath = "../../data/feature_subsets/activeSet_XYZ.csv";
    std::string videoPath = "../../data/video/bender_video_20mb.mp4";

    std::vector<int> filteredIndices;

    imagePreprocessor processor;

    std::map<int, cv::Point3f> indexedPoints = processor.load3DPoints(filteredIndicesXYZCoordinatesPath);

    auto cap = processor.initializeVideoCapture(1280, 720, "");

    cv::Mat refImage = processor.prepareReferenceImage(imagePath, calibrationFilePath, 1280, 720);
    if (refImage.empty())
    {
        std::cerr << "Error reading reference image.\n";
        return -1;
    }

    cv::Mat mask = processor.createRectangleMask(refImage.size(), cv::Size(1700, 850));
    auto kpAndDesc = processor.siftDetect(refImage, mask, nFeatures, nOctaveLayers,
                                          contrastThreshold, edgeThreshold, sigma);

    // Save keypoints and descriptors to CSV
    processor.keypointsAndDescriptorsToCSV(allFeaturesCSVPath, kpAndDesc.first, kpAndDesc.second);

    // Setup window for display
    cv::namedWindow("Matches", cv::WINDOW_NORMAL); // Make window resizable
    cv::resizeWindow("Matches", 3000, 2000);       // Set initial size

    std::vector<cv::KeyPoint> keypointsToUse = kpAndDesc.first;
    cv::Mat descriptorsToUse = kpAndDesc.second;

    if (mode == "use")
    {
        std::cout << "Using the best matches from CSV file." << std::endl;
        std::vector<int> bestIndices = processor.readTopFeatures(bestFeaturesPath, featuresFromCSV, SortCriteria::AverageDisplacement);

        auto filteredKpAndDesc = processor.filterKeypointsAndDescriptors(kpAndDesc, bestIndices);
        keypointsToUse = filteredKpAndDesc.first;
        descriptorsToUse = filteredKpAndDesc.second;

        // Annotate keypoints with indices on the reference image
        for (size_t i = 0; i < keypointsToUse.size(); ++i)
        {
            cv::Point2f position = keypointsToUse[i].pt;
            std::string label = std::to_string(bestIndices[i]);
            cv::putText(refImage, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    else if (mode == "filter")
    {
        // If in "filter" mode, process the filtered indices file
        std::cout << "Filtering keypoints and descriptors using CSV file." << std::endl;
        filteredIndices = processor.readIndicesFromCSV(filteredIndicesPath);
        auto filteredKpAndDesc = processor.filterKeypointsAndDescriptors(kpAndDesc, filteredIndices);
        keypointsToUse = filteredKpAndDesc.first;
        descriptorsToUse = filteredKpAndDesc.second;

        // Annotate keypoints with indices on the reference image
        for (size_t i = 0; i < keypointsToUse.size(); ++i)
        {
            cv::Point2f position = keypointsToUse[i].pt;
            std::string label = std::to_string(filteredIndices[i]);
            cv::putText(refImage, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    else
    {
        // Default behavior for other modes
        keypointsToUse = kpAndDesc.first;
        descriptorsToUse = kpAndDesc.second;
    }

    std::map<int, int> featureMatchCount;
    std::map<int, std::vector<cv::Point2f>> featureTracks;

    // Check the indices being used right after loading and before matching
    std::cout << "Loaded keypoints indices for matching: ";
    for (const auto &index : filteredIndices)
    {
        std::cout << index << " ";
    }
    std::cout << std::endl;
    cv::BFMatcher matcher(cv::NORM_L2, true);

    bool userExit = false;
    while (!userExit)
    {
        cv::Mat frame = processor.captureWebcam(&cap);
        if (frame.empty())
        {
            std::cout << "Failed to read frame from camera.\n";
            break;
        }
        std::vector<cv::Point2f> points2D;
        std::vector<cv::Point3f> points3D;
        std::vector<cv::DMatch> matches;
        cv::Mat currGray = processor.undistortImage(frame, calibrationFilePath);
        auto currKpAndDesc = processor.siftDetect(currGray, mask, nFeatures, nOctaveLayers,
                                                  contrastThreshold, edgeThreshold, sigma);

        // Drawing the coordinate frame here
        cv::Point center(frame.cols / 2, frame.rows / 2);
        int axisLength = 50; // You can adjust this value as needed
        // Draw the X-axis in red
        cv::arrowedLine(frame, center, cv::Point(center.x + axisLength, center.y), cv::Scalar(0, 0, 255), 2);
        // Draw the Y-axis in green, pointing downwards
        cv::arrowedLine(frame, center, cv::Point(center.x, center.y + axisLength), cv::Scalar(0, 255, 0), 2);

        // Display the frame with the coordinate frame drawn
        cv::imshow("Live Video with Coordinate Frame", frame);

        matcher.match(descriptorsToUse, currKpAndDesc.second, matches);

        // Filter matches based on distance threshold
        std::vector<cv::DMatch> goodMatches;
        for (auto &match : matches)
        {
            if (match.distance < 150)
            { // Define a suitable THRESHOLD based on your context
                goodMatches.push_back(match);
            }
        }
        processor.updateFeatureTracksAndCounts(matches, currKpAndDesc.first, featureMatchCount, featureTracks);

        if (mode == "filter")
        {
            processor.displayMatches(refImage, keypointsToUse, currGray, currKpAndDesc.first, matches);

            for (const auto &match : goodMatches)
            {
                if (indexedPoints.count(filteredIndices[match.queryIdx]))
                {
                    points2D.push_back(currKpAndDesc.first[match.trainIdx].pt);
                    points3D.push_back(indexedPoints[filteredIndices[match.queryIdx]]);
                }
            }

            // std::cout << "Number of 3D points: " << points3D.size() << std::endl;
            if (points3D.size() >= 4)
            {
                cv::Mat rvec, tvec;
                if (cv::solvePnP(points3D, points2D, processor.getCameraMatrix(), processor.getDistCoeffs(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE))
                {
                    cv::Mat rotationMatrix;
                    cv::Rodrigues(rvec, rotationMatrix);
                    std::cout << "Translation in X in mm: " << tvec.at<double>(0, 0) << std::endl;
                    std::cout << "Translation in Y in mm: " << tvec.at<double>(1, 0) << std::endl;
                    std::cout << "Translation in Z in mm: " << tvec.at<double>(2, 0) << std::endl;
                }
            }
        }
        processor.displayMatches(refImage, keypointsToUse, currGray, currKpAndDesc.first, matches);

        int key = cv::waitKey(10);
        if ((key & 0xFF) == 'q')
        {
            std::cout << "Exit requested by user. Key: " << key << std::endl;
            if (mode == "save")
            {
                std::cout << "Saving data to CSV." << std::endl;
                processor.saveFeatureTracksToCSV(bestFeaturesPath, featureMatchCount, featureTracks, matchCountThreshold);
            }
            userExit = true;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
