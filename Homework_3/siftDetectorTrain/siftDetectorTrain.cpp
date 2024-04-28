#include "siftDetectorTrain.h"

siftDetectorTrain::siftDetectorTrain()
{
    std::cout << "siftDetector created" << std::endl;
}

siftDetectorTrain::siftDetectorTrain(const cv::Mat &input) : input_(input)
{
    std::cout << "siftDetector created" << std::endl;
}

siftDetectorTrain::~siftDetectorTrain()
{
    std::cout << "siftDetector destroyed" << std::endl;
}

cv::Mat siftDetectorTrain::getInput() const
{
    return input_;
}

cv::Mat siftDetectorTrain::getOutput() const
{
    return output_;
}

void siftDetectorTrain::siftDetect()
{
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(this->input_, keypoints);

    cv::drawKeypoints(this->input_, keypoints, this->output_);
    cv::imwrite("sift_result.jpg", this->output_);
}