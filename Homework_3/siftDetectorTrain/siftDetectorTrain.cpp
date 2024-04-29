#include "siftDetectorTrain.h"

siftDetectorTrain::siftDetectorTrain()
{
    std::cout << "siftDetector created" << std::endl;
}

siftDetectorTrain::siftDetectorTrain(const cv::Mat &input, std::string pathToCalibrationFile)
    : input_(input), pathToCalibrationFile_(pathToCalibrationFile)
{
    try
    {
        YAML::Node config = YAML::LoadFile(pathToCalibrationFile_);
        if (config["camera_matrix"] && config["distortion_coefficients"])
        {
            std::vector<double> cameraMatrixData = config["camera_matrix"]["data"].as<std::vector<double>>();
            std::vector<double> distCoeffsData = config["distortion_coefficients"]["data"].as<std::vector<double>>();

            cameraMatrix_ = cv::Mat(cameraMatrixData, true).reshape(0, 3);                 // reshaping to 3x3 matrix
            distCoeffs_ = cv::Mat(distCoeffsData, true).reshape(0, distCoeffsData.size()); // reshaping if needed

            std::cout << "Camera Matrix:\n"
                      << cameraMatrix_ << std::endl;
            std::cout << "Distortion Coefficients:\n"
                      << distCoeffs_ << std::endl;
        }
        else
        {
            std::cerr << "Camera matrix or distortion coefficients not found in file" << std::endl;
        }
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "Failed to parse " << pathToCalibrationFile_ << ": " << e.what() << std::endl;
    }

    cv::undistort(this->input_, this->undistortedInput_, cameraMatrix_, distCoeffs_);
    std::cout << "Undistorted image created" << std::endl;
}

siftDetectorTrain::~siftDetectorTrain()
{
    std::cout << "siftDetector destroyed" << std::endl;
}

cv::Mat siftDetectorTrain::getInput() const
{
    return input_;
}

cv::Mat siftDetectorTrain::getUndistortedInput() const
{
    return undistortedInput_;
}

cv::Mat siftDetectorTrain::getOutput() const
{
    return output_;
}

void siftDetectorTrain::siftDetect()
{
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(this->undistortedInput_, keypoints);

    cv::drawKeypoints(this->undistortedInput_, keypoints, this->output_);
    cv::imwrite("original.jpg", this->input_);
    cv::imwrite("undistorted.jpg", this->undistortedInput_);
    cv::imwrite("sift_result_undistored.jpg", this->output_);
    std::cout << "SIFT detection completed" << std::endl;
}