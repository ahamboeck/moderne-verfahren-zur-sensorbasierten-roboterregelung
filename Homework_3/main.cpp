#include "imagePreprocessor/imagePreprocessor.h"

int main(int argc, char **argv)
{

    int nFeatures = 150;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.06;
    int edgeThreshold = 10;
    double sigma = 1.4;

    int featuresFromCSV = 20;
    int matchCountThreshold = 1;

    std::string mode = (argc > 1) ? argv[1] : "use";
    std::cout << "Running in mode: " << mode << std::endl;

    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/bender_new.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string csvFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/matched_features.csv";

    imagePreprocessor processor;
    auto cap = processor.initializeVideoCapture(640, 480);

    cv::Mat refImage = processor.prepareReferenceImage(imagePath, calibrationFilePath);
    if (refImage.empty())
    {
        std::cerr << "Error reading reference image.\n";
        return -1;
    }

    cv::Mat mask = processor.createRectangleMask(refImage.size(), cv::Size(500, 400));
    auto kpAndDesc = processor.siftDetect(refImage, mask, nFeatures, nOctaveLayers,
                                          contrastThreshold, edgeThreshold, sigma);

    // Setup window for display
    cv::namedWindow("Matches", cv::WINDOW_NORMAL); // Make window resizable
    cv::resizeWindow("Matches", 800, 600);         // Set initial size

    std::vector<cv::KeyPoint> keypointsToUse = kpAndDesc.first;
    cv::Mat descriptorsToUse = kpAndDesc.second;

    // If in use mode, filter keypoints and descriptors using the saved CSV file
    if (mode == "use")
    {
        std::cout << "Using saved features from CSV file." << std::endl;
        std::vector<int> indices = processor.readTopFeatures(csvFilePath, featuresFromCSV);
        keypointsToUse.clear();
        descriptorsToUse.release(); // Clear the existing Mat to refill it

        for (size_t index : indices)
        {
            if (index < kpAndDesc.first.size())
            {
                keypointsToUse.push_back(kpAndDesc.first[index]);
                descriptorsToUse.push_back(kpAndDesc.second.row(index));
            }
        }
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

        int key = cv::waitKey(40);
        if ((key & 0xFF) == 'q')
        {
            std::cout << "Exit requested by user. Key: " << key << std::endl;
            if (mode != "use")
            {
                std::cout << "Saving data to CSV." << std::endl;
                processor.saveFeatureTracksToCSV(csvFilePath, featureMatchCount, featureTracks, matchCountThreshold);
            }
            userExit = true;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
