#include "imagePreprocessor/imagePreprocessor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

// Helper function to read top 15 features from CSV
std::vector<int> readTopFeatures(const std::string &filepath)
{
    std::ifstream file(filepath);
    std::vector<std::pair<int, int>> featureCounts; // Feature index and counts
    std::string line, idx, count, var;
    getline(file, line); // Skip header

    while (getline(file, line))
    {
        std::istringstream iss(line);
        getline(iss, idx, ',');
        getline(iss, count, ',');
        getline(iss, var, ',');
        featureCounts.push_back({stoi(idx), stoi(count)});
    }

    // Sort by match count descending
    sort(featureCounts.begin(), featureCounts.end(), [](const auto &a, const auto &b)
         { return a.second > b.second; });

    std::vector<int> indices;
    for (int i = 0; i < 5 && i < featureCounts.size(); ++i)
    {
        indices.push_back(featureCounts[i].first);
    }
    return indices;
}

int main()
{
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/dragonball.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string csvFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/matched_features.csv";

    std::vector<int> topFeatureIndices = readTopFeatures(csvFilePath);

    imagePreprocessor imagePreprocessor;
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 800, 600);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(150);
    cv::Mat refImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty())
    {
        std::cerr << "Error reading reference image.\n";
        return -1;
    }
    cv::resize(refImage, refImage, cv::Size(640, 480));
    cv::Mat undistortedRefImage = imagePreprocessor.undistortImage(refImage, calibrationFilePath);
    cv::Mat mask = imagePreprocessor.createRectangleMask(undistortedRefImage.size(), cv::Size(500, 400));

    std::vector<cv::KeyPoint> refKeypoints;
    cv::Mat refDescriptors;
    sift->detectAndCompute(undistortedRefImage, mask, refKeypoints, refDescriptors);

    // Filter keypoints and descriptors based on the top indices
    std::vector<cv::KeyPoint> filteredKeypoints;
    cv::Mat filteredDescriptors;
    for (int idx : topFeatureIndices)
    {
        if (idx < refKeypoints.size())
        {
            filteredKeypoints.push_back(refKeypoints[idx]);
            filteredDescriptors.push_back(refDescriptors.row(idx));
        }
    }

    cv::BFMatcher matcher(cv::NORM_L2, true);
    cv::Mat frame, currGray, prevGray;
    std::vector<cv::Point2f> prevPoints;
    std::vector<uchar> status;
    std::vector<float> err;

    while (true)
    {
        if (!cap.read(frame))
        {
            std::cerr << "Failed to read frame from camera.\n";
            break;
        }
        cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);
        cv::Mat undistortedCurr = imagePreprocessor.undistortImage(currGray, calibrationFilePath);

        std::vector<cv::KeyPoint> currKeypoints;
        cv::Mat currDescriptors;
        sift->detectAndCompute(undistortedCurr, mask, currKeypoints, currDescriptors);

        std::vector<cv::DMatch> matches;
        matcher.match(filteredDescriptors, currDescriptors, matches);

        // Display results
        cv::Mat imgMatches;
        cv::drawMatches(undistortedRefImage, filteredKeypoints, undistortedCurr, currKeypoints, matches, imgMatches);
        cv::imshow("Matches", imgMatches);
        if (cv::waitKey(30) >= 0)
            break;
    }

    cv::destroyAllWindows();
    return 0;
}
