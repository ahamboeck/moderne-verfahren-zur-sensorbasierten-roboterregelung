#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

class SiftDetectorMatch
{
public:
    SiftDetectorMatch();
    SiftDetectorMatch(int width = 640, int height = 480, int device = 0);
    ~SiftDetectorMatch();

    cv::Mat getFrame();
private:
    cv::Mat frame;
    cv::VideoCapture cap;
};