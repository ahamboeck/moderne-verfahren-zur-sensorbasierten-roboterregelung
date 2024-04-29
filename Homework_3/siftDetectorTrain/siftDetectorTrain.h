#pragma once

#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>

class siftDetectorTrain
{
private:
    const cv::Mat input_;
    cv::Mat undistortedInput_;
    cv::Mat output_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    std::string pathToCalibrationFile_;
public:
    siftDetectorTrain();
    siftDetectorTrain(const cv::Mat &input, std::string pathToCalibrationFile);
    ~siftDetectorTrain();

    // Getters
    cv::Mat getInput() const;
    cv::Mat getUndistortedInput() const;
    cv::Mat getOutput() const;

    // Methods
    void siftDetect();
};
