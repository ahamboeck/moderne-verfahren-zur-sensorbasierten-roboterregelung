#include "imagePreprocessor/imagePreprocessor.h"

void onTrackbar(int position, void* userData) {
    std::pair<cv::Mat*, std::vector<cv::KeyPoint>*> data = *(std::pair<cv::Mat*, std::vector<cv::KeyPoint>*>*)userData;
    cv::Mat* refImage = data.first;
    std::vector<cv::KeyPoint>* keypoints = data.second;

    // Redraw the reference image with only the selected keypoint
    cv::Mat displayImage = refImage->clone();
    if (!keypoints->empty()) {
        cv::KeyPoint selectedKeypoint = (*keypoints)[position];
        cv::circle(displayImage, selectedKeypoint.pt, 3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(displayImage, std::to_string(position + 1), selectedKeypoint.pt + cv::Point2f(4, 4),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow("Matches", displayImage);
}


int main()
{
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/sift_fu.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string keypointsAndDescriptorsFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/keypoints_and_descriptors.csv";

    imagePreprocessor imagePreprocessor;
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Initialize the window for showing matches
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 800, 600);

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

    // Step 2: Detect keypoints and compute descriptors on the undistorted reference image
    KeypointsAndDescriptors refKpAndDesc = imagePreprocessor.siftDetect(undistortedRefImage);

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
    cv::BFMatcher matcher(cv::NORM_L2);

    for (;;)
    {
        // Step 1: Capture webcam
        cv::Mat im = imagePreprocessor.captureWebcam(&cap);

        // Step 2: Undistort image
        cv::Mat undistortedIm = imagePreprocessor.undistortImage(im, calibrationFilePath);

        // Step 3: Detect keypoints and compute descriptors for the current frame
        KeypointsAndDescriptors currentKpAndDesc = imagePreprocessor.siftDetect(undistortedIm);

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
