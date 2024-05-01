#include "imagePreprocessor.h"

imagePreprocessor::imagePreprocessor()
{
    std::cout << "imagePreprocessor created" << std::endl;
}

imagePreprocessor::~imagePreprocessor()
{
    std::cout << "imagePreprocessor destroyed" << std::endl;
}

cv::Mat imagePreprocessor::captureWebcam(int width, int height, int device)
{
    cv::VideoCapture cap(device);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening webcam" << std::endl;
        return cv::Mat();
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat frame;
    cap >> frame;
    std::cout << "Webcam frame captured" << std::endl;
    return frame;
}

cv::Mat imagePreprocessor::undistortImage(cv::Mat &input, std::string pathToCalibrationFile)
{
    try
    {
        YAML::Node config = YAML::LoadFile(pathToCalibrationFile);
        if (config["camera_matrix"] && config["distortion_coefficients"])
        {
            std::vector<double> cameraMatrixData = config["camera_matrix"]["data"].as<std::vector<double>>();
            std::vector<double> distCoeffsData = config["distortion_coefficients"]["data"].as<std::vector<double>>();

            cv::Mat cameraMatrix = cv::Mat(cameraMatrixData, true).reshape(0, 3);                 // reshaping to 3x3 matrix
            cv::Mat distCoeffs = cv::Mat(distCoeffsData, true).reshape(0, distCoeffsData.size()); // reshaping if needed

            cv::Mat undistortedInput;
            cv::undistort(input, undistortedInput, cameraMatrix, distCoeffs);
            std::cout << "Undistorted image created" << std::endl;
            return undistortedInput;
        }
        else
        {
            std::cerr << "Camera matrix or distortion coefficients not found in file" << std::endl;
        }
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "Failed to parse " << pathToCalibrationFile << ": " << e.what() << std::endl;
    }
    return cv::Mat();
}

KeypointsAndDescriptors imagePreprocessor::siftDetect(cv::Mat &input) const
{
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    detector->detect(input, keypoints);
    detector->compute(input, keypoints, descriptors);

    std::cout << "Detected " << keypoints.size() << " keypoints and computed descriptors." << std::endl;
    std::cout << "SIFT detection completed" << std::endl;
    return {keypoints, descriptors};
    }

cv::Mat imagePreprocessor::drawKeypoints(cv::Mat &input, std::vector<cv::KeyPoint> &keypoints) const
{
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    std::cout << "Keypoints drawn" << std::endl;
    return output;
}

void imagePreprocessor::keypointsAndDescriptorsToCSV(std::string path, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    // Write header
    file << "X,Y,Size,Angle,Response,Octave,ClassID,";
    for (int j = 0; j < descriptors.cols; j++)
    {
        file << "D" << j << ",";
    }
    file << std::endl;

    // Write data
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        file << keypoints[i].pt.x << "," << keypoints[i].pt.y << "," << keypoints[i].size << "," << keypoints[i].angle << "," << keypoints[i].response << "," << keypoints[i].octave << "," << keypoints[i].class_id << ",";
        for (int j = 0; j < descriptors.cols; j++)
        {
            file << descriptors.at<float>(i, j) << ",";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Keypoints and descriptors written to " << path << std::endl;
}

