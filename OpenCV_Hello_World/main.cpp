#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{   
    if (argv[1] == NULL)
    {
        std::cout << "Please provide an image path" << std::endl;
        return 1;
    }
    std::string imagePath = argv[1];
    std::cout << "Image path: " << imagePath << std::endl;

    cv::Mat image = cv::imread(imagePath);
    cv::namedWindow("Hello World", 0);
    cv::imshow("Hello World", image);
    cv::waitKey(0);
    return 0;
}