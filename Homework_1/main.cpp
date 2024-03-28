#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>

class discreteConvolution
{
public:
    discreteConvolution(cv::Mat kernelMatrix)
        : kernelMatrix_(kernelMatrix) {}
    ~discreteConvolution() {}

    cv::Mat conv(const cv::Mat &image)
    {
        cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);

        int kernelOffsetX = kernelMatrix_.size().width / 2;
        int kernelOffsetY = kernelMatrix_.size().height / 2;

        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                float sum = 0;

                for (int k = -kernelOffsetY; k <= kernelOffsetY; k++)
                {
                    for (int l = -kernelOffsetX; l <= kernelOffsetX; l++)
                    {
                        int x = i + k;
                        int y = j + l;

                        if (x >= 0 && x < image.rows && y >= 0 && y < image.cols)
                        {
                            sum += static_cast<float>(image.at<uchar>(x, y)) * kernelMatrix_.at<float>(k + kernelOffsetY, l + kernelOffsetX);
                        }
                    }
                }
                result.at<float>(i, j) = sum;
            }
        }
        return result;
    }

private:
    cv::Mat kernelMatrix_;
};

class sobelDetector
{
public:
    sobelDetector(const cv::Mat &kernelX, const cv::Mat &kernelY)
        : sobelX(kernelX), sobelY(kernelY) {}

    cv::Mat getEdges(const cv::Mat &grayscaleImage)
    {
        cv::Mat sobelXImage = sobelX.conv(grayscaleImage);
        cv::Mat sobelYImage = sobelY.conv(grayscaleImage);

        cv::Mat absSobelXImage, absSobelYImage;
        cv::convertScaleAbs(sobelXImage, absSobelXImage);
        cv::convertScaleAbs(sobelYImage, absSobelYImage);
        cv::imshow("Sobel X Image", absSobelXImage);
        cv::imshow("Sobel Y Image", absSobelYImage);
        
        cv::Mat edgeImage;
        cv::addWeighted(absSobelXImage, 0.5, absSobelYImage, 0.5, 0, edgeImage);

        return edgeImage;
    }

private:
    discreteConvolution sobelX, sobelY;
};

// This struct is used to avoid global variables and to pass data to the trackbar callback
struct ImageData
{
    // Pretty self-explanatory
    cv::Mat originalImage;

    // We will convert the original image to HSV and store it here
    cv::Mat hsvImage;

    // We will use this to store a masked version of the original image
    cv::Mat normalizedGrayscaleResult;

    /*
    These are the values that we will use to filter the
    image and these will be adjusted with a slider. The default
    values are for a red filter, but you can change them to whatever
    you like.

    hueValue: 0-180
    saturationValue: 0-255
    valueValue: 0-255
    */
    int hueValue = 0;
    int saturationValue = 128;
    int valueValue = 159;
};

// This function will be called whenever the trackbar is moved
void on_trackbar(int, void *userdata)
{
    ImageData *imageData = (ImageData *)userdata;

    cv::Mat mask;

    cv::inRange(imageData->hsvImage, cv::Scalar(imageData->hueValue, imageData->saturationValue, imageData->valueValue), cv::Scalar(180, 255, 255), mask);

    cv::Mat result;
    imageData->originalImage.copyTo(result, mask);

    cv::Mat grayscaleResult;
    cv::cvtColor(result, grayscaleResult, cv::COLOR_BGR2GRAY);

    cv::normalize(grayscaleResult, imageData->normalizedGrayscaleResult, 0, 255, cv::NORM_MINMAX);

    cv::imshow("HSV Adjustment", imageData->normalizedGrayscaleResult);
}

int main(int argc, char *argv[])
{
    if (argv[1] == NULL)
    {
        std::cout << "Please provide an image path" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    std::cout << "Image path: " << imagePath << std::endl;

    ImageData data;
    data.originalImage = cv::imread(imagePath);

    if (data.originalImage.empty())
    {
        std::cout << "Image not found" << std::endl;
        return -1;
    }

    cv::cvtColor(data.originalImage, data.hsvImage, cv::COLOR_BGR2HSV);

    cv::namedWindow("HSV Adjustment", 0);

    cv::createTrackbar("Hue", "HSV Adjustment", &data.hueValue, 180, on_trackbar, &data);
    cv::createTrackbar("Saturation", "HSV Adjustment", &data.saturationValue, 255, on_trackbar, &data);
    cv::createTrackbar("Value", "HSV Adjustment", &data.valueValue, 255, on_trackbar, &data);

    on_trackbar(0, &data);

    discreteConvolution Denoize(cv::Mat::ones(3, 3, CV_32F) / 9);
    cv::Mat denoizedFloatImage = Denoize.conv(data.normalizedGrayscaleResult); // This is still in CV_32F

    cv::Mat denoizedImage;
    cv::convertScaleAbs(denoizedFloatImage, denoizedImage); // Correctly scales and converts to CV_8U

    cv::imshow("Denoized Image", denoizedImage);

    // Define Sobel kernels for X and Y directions
    cv::Mat kernelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    // Create an instance of sobelDetector
    sobelDetector detector(kernelX, kernelY);

    // Use the sobelDetector to get the edge image
    cv::Mat edgeImage = detector.getEdges(data.normalizedGrayscaleResult);

    // Visualize the edge image
    cv::imshow("Sobel Edge Image", edgeImage);

    cv::Mat normalizedSobelXYAbsoluteImage;
    cv::normalize(edgeImage, normalizedSobelXYAbsoluteImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("Normalized Absolute Sobel XY Image", normalizedSobelXYAbsoluteImage);

    cv::Mat OTSUThresholdedImage;
    cv::threshold(normalizedSobelXYAbsoluteImage, OTSUThresholdedImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imshow("OTSU Thresholded Image", OTSUThresholdedImage);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(OTSUThresholdedImage, circles, cv::HOUGH_GRADIENT, 1, OTSUThresholdedImage.rows / 16, 100, 30, 20, 50);

    cv::Mat circleImage;
    circleImage = data.originalImage.clone();

    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(circleImage, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        cv::circle(circleImage, center, radius, cv::Scalar(255, 0, 0), 3, 8, 0);
    }

    // Display the image with the circles
    cv::imshow("Hough Circles Image", circleImage);

    cv::waitKey(0);
    return 0;
}
