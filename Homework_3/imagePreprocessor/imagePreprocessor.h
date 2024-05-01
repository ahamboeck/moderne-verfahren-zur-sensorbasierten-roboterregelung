#pragma once

#include <iostream>
#include <vector>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>

using KeypointsAndDescriptors = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

class imagePreprocessor
{
    public:
        imagePreprocessor();
        ~imagePreprocessor();

        // Methods
        cv::Mat captureWebcam(int width = 640, int height = 480, int device = 0);
        cv::Mat undistortImage(cv::Mat& input, std::string pathToCalibrationFile);
        KeypointsAndDescriptors siftDetect(cv::Mat& input) const;
        cv::Mat drawKeypoints(cv::Mat& input, std::vector<cv::KeyPoint>& keypoints) const;
        void keypointsAndDescriptorsToCSV(std::string path, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

    private:
        const cv::Mat input_;
        cv::Mat undistortedInput_;
        cv::Mat output_;
        cv::Mat cameraMatrix_;
        cv::Mat distCoeffs_;
        std::string pathToCalibrationFile_;
};