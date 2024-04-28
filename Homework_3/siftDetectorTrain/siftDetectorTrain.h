#pragma once

#include <iostream>
#include <string.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

class siftDetectorTrain
{
private:
    const cv::Mat input_;
    cv::Mat output_;
public:
    siftDetectorTrain();
    siftDetectorTrain(const cv::Mat &input);
    ~siftDetectorTrain();

    // Getters
    cv::Mat getInput() const;
    cv::Mat getOutput() const;

    // Methods
    void siftDetect();
};
