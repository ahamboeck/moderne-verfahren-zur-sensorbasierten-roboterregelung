#pragma once

#include <iostream>
#include <vector>
#include <string.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <yaml-cpp/yaml.h>

using KeypointsAndDescriptors = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

class imagePreprocessor
{
    public:
        imagePreprocessor();
        ~imagePreprocessor();

        // Methods
        cv::Mat captureWebcam(cv::VideoCapture* cap = nullptr);
        cv::Mat undistortImage(cv::Mat& input, std::string pathToCalibrationFile);
        KeypointsAndDescriptors siftDetect(cv::Mat& input) const;
        KeypointsAndDescriptors filterKeypointsAndDescriptors(KeypointsAndDescriptors &KpAndDesc, const std::vector<int>& indices);
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