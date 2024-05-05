#include "imagePreprocessor/imagePreprocessor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

// Function to calculate variance of feature movements
double calculateMovementVariance(const std::vector<cv::Point2f>& points) {
    if (points.size() < 2) return 0.0;

    cv::Point2f sum(0, 0);
    for (const auto& point : points) {
        sum += point;
    }
    cv::Point2f mean = sum * (1.0 / points.size());

    double variance = 0.0;
    for (const auto& point : points) {
        cv::Point2f diff = point - mean;
        variance += diff.x * diff.x + diff.y * diff.y;
    }
    return variance / points.size();
}

int main() {
    std::string imagePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/original/dragonball.jpg";
    std::string calibrationFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/calib_v0.3/ost.yaml";
    std::string csvFilePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/data/matched_features.csv";

    imagePreprocessor imagePreprocessor;
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matches", 800, 600);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(150);
    cv::Mat refImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (refImage.empty()) {
        std::cerr << "Error reading reference image.\n";
        return -1;
    }
    cv::resize(refImage, refImage, cv::Size(640, 480));
    cv::Mat undistortedRefImage = imagePreprocessor.undistortImage(refImage, calibrationFilePath);
    cv::Mat mask = imagePreprocessor.createRectangleMask(undistortedRefImage.size(), cv::Size(500, 400));

    std::vector<cv::KeyPoint> refKeypoints;
    cv::Mat refDescriptors;
    sift->detectAndCompute(undistortedRefImage, mask, refKeypoints, refDescriptors);

    std::map<int, int> featureMatchCount;
    std::map<int, std::vector<cv::Point2f>> featureTracks;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    cv::Mat frame, currGray, prevGray;
    std::vector<cv::Point2f> prevPoints;
    std::vector<uchar> status;
    std::vector<float> err;

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Failed to read frame from camera.\n";
            break;
        }
        cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);
        cv::Mat undistortedCurr = imagePreprocessor.undistortImage(currGray, calibrationFilePath);

        std::vector<cv::KeyPoint> currKeypoints;
        cv::Mat currDescriptors;
        sift->detectAndCompute(undistortedCurr, mask, currKeypoints, currDescriptors);

        std::vector<cv::DMatch> matches;
        matcher.match(refDescriptors, currDescriptors, matches);

        // Update feature tracks and counts based on matches
        for (auto& match : matches) {
            int idx = match.queryIdx;
            featureMatchCount[idx]++;
            featureTracks[idx].push_back(currKeypoints[match.trainIdx].pt);
        }

        // Display results
        cv::Mat imgMatches;
        cv::drawMatches(undistortedRefImage, refKeypoints, undistortedCurr, currKeypoints, matches, imgMatches);
        cv::imshow("Matches", imgMatches);
        if (cv::waitKey(30) >= 0) break;
    }

    // Evaluate variance and save results
    std::ofstream outFile(csvFilePath);
    outFile << "FeatureIndex,MatchCount,Variance\n";
    int minMatches = 500;  // Threshold for "most often" criteria
    for (auto& track : featureTracks) {
        double variance = calculateMovementVariance(track.second);
        int count = featureMatchCount[track.first];
        if (count >= minMatches) {
            outFile << track.first << "," << count << "," << variance << "\n";
        }
    }
    outFile.close();

    cv::destroyAllWindows();
    return 0;
}
