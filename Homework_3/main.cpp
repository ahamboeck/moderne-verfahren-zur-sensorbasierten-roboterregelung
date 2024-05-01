#include "imagePreprocessor/imagePreprocessor.h"

int main()
{
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/rsz_pattern_1.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string keypointsAndDescriptorsFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/keypoints_and_descriptors.csv";

    imagePreprocessor imagePreprocessor;
    while (true)
    {
        // Capture webcam as grayscale image
        cv::Mat im = imagePreprocessor.captureWebcam(640, 480, 0);

        // Undistort image
        cv::Mat undistortedIm = imagePreprocessor.undistortImage(im, calibrationFilePath);

        // Detect keypoints and compute descriptors
        KeypointsAndDescriptors keypointsAndDescriptors = imagePreprocessor.siftDetect(undistortedIm);

        // Draw keypoints
        cv::Mat output = imagePreprocessor.drawKeypoints(undistortedIm, keypointsAndDescriptors.first);

        // Show image
        cv::imshow("Output", output);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();

    return 0;
}