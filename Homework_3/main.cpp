#include "imagePreprocessor/imagePreprocessor.h"

int main()
{
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/dragonball.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string csvFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/matched_features.csv";

    imagePreprocessor processor;
    auto cap = processor.initializeVideoCapture(640, 480);

    try
    {
        cv::Mat refImage = processor.prepareReferenceImage(imagePath, calibrationFilePath);
        cv::Mat mask = processor.createRectangleMask(refImage.size(), cv::Size(500, 400));
        auto kpAndDesc = processor.siftDetect(refImage, mask, 150, 3, 0.04, 10, 1.6);

        std::map<int, int> featureMatchCount;
        std::map<int, std::vector<cv::Point2f>> featureTracks;
        cv::BFMatcher matcher(cv::NORM_L2, true);

        while (true)
        {
            cv::Mat frame = processor.captureWebcam(&cap);
            if (frame.empty())
                break;

            cv::Mat currGray = processor.undistortImage(frame, calibrationFilePath);
            auto currKpAndDesc = processor.siftDetect(currGray, mask, 150, 3, 0.04, 10, 1.6);
            std::vector<cv::DMatch> matches;
            matcher.match(kpAndDesc.second, currKpAndDesc.second, matches);
            std::cout << "Number of matches found: " << matches.size() << std::endl;
            processor.updateFeatureTracksAndCounts(matches, currKpAndDesc.first, featureMatchCount, featureTracks);
            processor.displayMatches(refImage, kpAndDesc.first, currGray, currKpAndDesc.first, matches);
            if (cv::waitKey(25) == 113)
            { // ASCII value for 'q'
                std::cout << "Exit requested by user." << std::endl;
                processor.saveFeatureTracksToCSV(csvFilePath, featureMatchCount, featureTracks, 250);
                break;
            }
        }
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    cv::destroyAllWindows();
    return 0;
}
