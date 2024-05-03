#include "imagePreprocessor/imagePreprocessor.h"

int main()
{
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/dasisgut.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string keypointsAndDescriptorsFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/keypoints_and_descriptors.csv";

    imagePreprocessor imagePreprocessor;
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Initialize the window for showing matches
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 800, 600);

    int nFeatures = 0;
    int nOctaveLayers = 3;
    int contrastThresholdInt = 40; // Initial value for contrast threshold * 1000
    double contrastThreshold = 0.04;
    int edgeThresholdInt = 10; // Initial value for edge threshold
    double edgeThreshold = 10.0;
    int sigmaInt = 16; // Initial value for sigma * 10
    double sigma = 1.6;
    
    // Flags to track parameter changes
    int lastFeatures = nFeatures;
    int lastOctaveLayers = nOctaveLayers;
    double lastContrastThreshold = contrastThreshold;
    double lastEdgeThreshold = edgeThreshold;
    double lastSigma = sigma;

    cv::namedWindow("Control Panel", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Features", "Control Panel", &nFeatures, 500);
    cv::createTrackbar("Octave Layers", "Control Panel", &nOctaveLayers, 10);
    cv::createTrackbar(
        "Contrast Threshold * 1000", "Control Panel", &contrastThresholdInt, 100, [](int pos, void *userData)
        { *(double *)userData = pos / 1000.0; },
        &contrastThreshold);
    cv::createTrackbar("Edge Threshold", "Control Panel", &edgeThresholdInt, 100);
    cv::createTrackbar(
        "Sigma * 10", "Control Panel", &sigmaInt, 40, [](int pos, void *userData)
        { *(double *)userData = pos / 10.0; },
        &sigma);

    // Step 1: Load the reference image and undistort it
    cv::Mat refImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty())
    {
        std::cerr << "Error reading reference image." << std::endl;
        return -1;
    }

    // Resize the reference image to match video frame dimensions (640x480)
    cv::resize(refImage, refImage, cv::Size(640, 480));

    // Undistort the reference image using calibration data
    cv::Mat undistortedRefImage = imagePreprocessor.undistortImage(refImage, calibrationFilePath);

    cv::Size rectSize(500, 400); // The desired size of the rectangle
    cv::Mat mask = imagePreprocessor.createRectangleMask(undistortedRefImage.size(), rectSize);

    // Step 2: Detect keypoints and compute descriptors on the undistorted reference image
    KeypointsAndDescriptors refKpAndDesc = imagePreprocessor.siftDetect(undistortedRefImage, mask, 150, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

    // Select indices manually
    std::vector<int> selectedIndices = {29, 30, 2, 1, 6, 7, 8, 22, 11, 34, 25, 30, 26, 28, 10}; // Manually selected indices of keypoints
    KeypointsAndDescriptors filteredRefKpAndDesc = imagePreprocessor.filterKeypointsAndDescriptors(refKpAndDesc, selectedIndices);
    filteredRefKpAndDesc = refKpAndDesc;

    // Annotate keypoints with numbers on the reference image
    cv::Mat annotatedRefImage = undistortedRefImage.clone();
    for (size_t i = 0; i < filteredRefKpAndDesc.first.size(); i++)
    {
        cv::circle(annotatedRefImage, filteredRefKpAndDesc.first[i].pt, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::putText(annotatedRefImage, std::to_string(i + 1), filteredRefKpAndDesc.first[i].pt + cv::Point2f(4, 4),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    // Step 3: Initialize the brute force matcher
    cv::BFMatcher matcher(cv::NORM_L2, true);

    for (;;)
    {
        // Detect parameter changes
        bool parametersChanged = lastFeatures != nFeatures ||
                                 lastOctaveLayers != nOctaveLayers ||
                                 lastContrastThreshold != contrastThreshold ||
                                 lastEdgeThreshold != edgeThreshold ||
                                 lastSigma != sigma;

        // Update parameters if changed
        if (parametersChanged)
        {
            // Update last known parameters
            lastFeatures = nFeatures;
            lastOctaveLayers = nOctaveLayers;
            lastContrastThreshold = contrastThreshold;
            lastEdgeThreshold = edgeThreshold;
            lastSigma = sigma;

            // Recalculate keypoints and descriptors for the reference image
            refKpAndDesc = imagePreprocessor.siftDetect(undistortedRefImage, mask, nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

            // Optionally re-annotate the reference image
            annotatedRefImage = undistortedRefImage.clone();
            for (size_t i = 0; i < refKpAndDesc.first.size(); i++)
            {
                cv::circle(annotatedRefImage, refKpAndDesc.first[i].pt, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                cv::putText(annotatedRefImage, std::to_string(i + 1), refKpAndDesc.first[i].pt + cv::Point2f(4, 4),
                            cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }

        // Step 1: Capture webcam
        cv::Mat im = imagePreprocessor.captureWebcam(&cap);

        // Step 2: Undistort image
        cv::Mat undistortedIm = imagePreprocessor.undistortImage(im, calibrationFilePath);

        // Step 3: Detect keypoints and compute descriptors for the current frame
        KeypointsAndDescriptors currentKpAndDesc = imagePreprocessor.siftDetect(undistortedIm, mask, nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

        // Step 4: Match the current frame's descriptors with the reference image's descriptors
        std::vector<cv::DMatch> matches;
        matcher.match(filteredRefKpAndDesc.second, currentKpAndDesc.second, matches);

        // Step 5: Sort matches by their distance to find good matches
        std::sort(matches.begin(), matches.end());
        std::vector<cv::DMatch> goodMatches(matches.begin(), matches.begin() + std::min(50, (int)matches.size()));

        // Step 6: Draw the good matches on the output image using annotated reference image
        cv::Mat imgMatches;
        cv::drawMatches(annotatedRefImage, filteredRefKpAndDesc.first, undistortedIm, currentKpAndDesc.first, goodMatches, imgMatches);

        // Step 7: Show the matched image in the resizable window
        imshow("Matches", imgMatches);

        // Exit condition - press any key to close
        if (cv::waitKey(25) >= 0)
            break;
    }

    cv::destroyAllWindows();

    return 0;
}
