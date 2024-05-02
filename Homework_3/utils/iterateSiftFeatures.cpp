#include "../imagePreprocessor/imagePreprocessor.h"

// Define a callback function for the trackbar
void onTrackbar(int position, void* userData) {
    std::pair<cv::Mat*, std::vector<cv::KeyPoint>*>* data = reinterpret_cast<std::pair<cv::Mat*, std::vector<cv::KeyPoint>*>*>(userData);
    cv::Mat* refImage = data->first;
    std::vector<cv::KeyPoint>& keypoints = *(data->second);

    // Redraw the reference image with only the selected keypoint
    cv::Mat displayImage = refImage->clone();
    if (!keypoints.empty()) {
        cv::KeyPoint selectedKeypoint = keypoints[position];
        cv::circle(displayImage, selectedKeypoint.pt, 5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(displayImage, std::to_string(position + 1), selectedKeypoint.pt + cv::Point2f(4, 4),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow("Matches", displayImage);
}

int main() {
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/sift_fu.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";

    imagePreprocessor imagePreprocessor;
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Load the reference image and preprocess it
    cv::Mat refImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty()) {
        std::cerr << "Error reading reference image." << std::endl;
        return -1;
    }
    cv::resize(refImage, refImage, cv::Size(640, 480));
    cv::Mat undistortedRefImage = imagePreprocessor.undistortImage(refImage, calibrationFilePath);
    KeypointsAndDescriptors refKpAndDesc = imagePreprocessor.siftDetect(undistortedRefImage);

    // Initialize the window for showing matches
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 800, 600);

    // Prepare a pair to pass to the trackbar callback
    std::pair<cv::Mat*, std::vector<cv::KeyPoint>*> data(&undistortedRefImage, &refKpAndDesc.first);

    // Ensure the data is ready before setting up the trackbar
    int maxKeypointIndex = refKpAndDesc.first.size() - 1;
    int trackbarValue = 0;
    cv::createTrackbar("Keypoint", "Matches", &trackbarValue, maxKeypointIndex, onTrackbar, &data);

    // Initialize matcher
    cv::BFMatcher matcher(cv::NORM_L2);

    // Main loop
    while (true) {
        // Capture from webcam
        cv::Mat frame;
        cap.read(frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::Mat undistortedFrame = imagePreprocessor.undistortImage(frame, calibrationFilePath);
        KeypointsAndDescriptors currentKpAndDesc = imagePreprocessor.siftDetect(undistortedFrame);

        // Match features
        cv::Mat singleRefDescriptor = refKpAndDesc.second.row(trackbarValue);
        std::vector<cv::DMatch> matches;
        matcher.match(singleRefDescriptor, currentKpAndDesc.second, matches);

        // Filter matches
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
        std::vector<cv::DMatch> goodMatches(matches.begin(), matches.begin() + std::min(10, (int)matches.size()));

        // Draw matches
        cv::Mat imgMatches;
        std::vector<cv::KeyPoint> singleRefKeypoint = {refKpAndDesc.first[trackbarValue]};
        cv::drawMatches(undistortedRefImage, singleRefKeypoint, undistortedFrame, currentKpAndDesc.first, goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::imshow("Live Matches", imgMatches);

        // Check exit condition
        if (cv::waitKey(25) >= 0)
            break;
    }

    cv::destroyAllWindows();
    return 0;
}
