#include "siftDetectorTrain/siftDetectorTrain.h"

int main()
{
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/happy_rick.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.1/ost.yaml";
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    siftDetectorTrain siftDetector(input, calibrationFilePath);
    siftDetector.siftDetect();

    return 0;
}