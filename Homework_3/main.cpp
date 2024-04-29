#include "siftDetectorTrain/siftDetectorTrain.h"

int main()
{
    cv::Mat input = cv::imread("/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/eifelturm.jpg", cv::IMREAD_GRAYSCALE);
    siftDetectorTrain siftDetector(input, "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/ost.yaml");
    siftDetector.siftDetect();

    return 0;
}