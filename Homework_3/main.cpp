#include "imagePreprocessor/imagePreprocessor.h"

int main(int argc, char **argv)
{

    int nFeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    int edgeThreshold = 8;
    double sigma = 1.6;

    int featuresFromCSV = 75;
    int matchCountThreshold = 150;

    std::string mode = (argc > 1) ? argv[1] : "use";
    std::cout << "Running in mode: " << mode << std::endl;

    // Define the base path for all files
    std::string basePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/";

    // Derived paths from the base path
    std::string imagePath = basePath + "data/original/bender_good_setup.jpg";
    std::string calibrationFilePath = basePath + "camera_calib_data/calib_v0.4/ost.yaml";
    std::string bestFeaturesPath = basePath + "data/matched_features.csv";
    std::string allFeaturesCSVPath = basePath + "data/keypoints_and_descriptors.csv";
    std::string filterIndicesPath = basePath + "data/activeSet.csv";
    std::string videoPath = basePath + "data/video/bender_video_20mb.mp4";

    imagePreprocessor processor;
    auto cap = processor.initializeVideoCapture(1920, 1080, videoPath);

    cv::Mat refImage = processor.prepareReferenceImage(imagePath, calibrationFilePath, 1920, 1080);
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
    cv::resizeWindow("Matches", 1920, 1080);       // Set initial size

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
        std::vector<int> filterIndices = processor.readIndicesFromCSV(filterIndicesPath);

        auto filteredKpAndDesc = processor.filterKeypointsAndDescriptors(kpAndDesc, filterIndices);
        keypointsToUse = filteredKpAndDesc.first;
        descriptorsToUse = filteredKpAndDesc.second;

        // Annotate keypoints with indices on the reference image
        for (size_t i = 0; i < keypointsToUse.size(); ++i)
        {
            cv::Point2f position = keypointsToUse[i].pt;
            std::string label = std::to_string(filterIndices[i]);
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

        cv::Mat currGray = processor.undistortImage(frame, calibrationFilePath);
        auto currKpAndDesc = processor.siftDetect(currGray, mask, nFeatures, nOctaveLayers,
                                                  contrastThreshold, edgeThreshold, sigma);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptorsToUse, currKpAndDesc.second, matches);
        processor.updateFeatureTracksAndCounts(matches, currKpAndDesc.first, featureMatchCount, featureTracks);
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
