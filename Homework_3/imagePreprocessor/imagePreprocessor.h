#pragma once
#include <map>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>

using KeypointsAndDescriptors = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

/**
 * @class imagePreprocessor
 * @brief The imagePreprocessor class provides methods for preprocessing images, such as capturing webcam frames,
 * undistorting images, detecting keypoints and descriptors using SIFT, filtering keypoints and descriptors,
 * drawing keypoints on images, saving keypoints and descriptors to CSV files, creating rectangle masks,
 * calculating SIFT feature movement variance, and updating feature tracks and counts.
 */
class imagePreprocessor
{
public:
    /**
     * @brief Default constructor for the imagePreprocessor class.
     */
    imagePreprocessor();

    /**
     * @brief Destructor for the imagePreprocessor class.
     */
    ~imagePreprocessor();

    // Methods

    /**
     * @brief Captures a frame from the webcam.
     * @param cap A pointer to the cv::VideoCapture object representing the webcam.
     * @return The captured frame as a cv::Mat object.
     */
    cv::Mat captureWebcam(cv::VideoCapture *cap = nullptr);

    /**
     * Preprocesses the reference image for calibration.
     *
     * This function takes the path to the reference image and the calibration file as input.
     * It performs necessary preprocessing steps on the image, such as resizing, filtering, or
     * color correction, to prepare it for calibration.
     *
     * @param imagePath The path to the reference image.
     * @param calibrationPath The path to the calibration file.
     * @return The preprocessed reference image.
     */
    cv::Mat prepareReferenceImage(const std::string &imagePath, const std::string &calibrationPath);

    /**
     * Displays the matches between two images along with their corresponding keypoints.
     *
     * @param img1 The first image.
     * @param keypoints1 The keypoints detected in the first image.
     * @param img2 The second image.
     * @param keypoints2 The keypoints detected in the second image.
     * @param matches The matches between the keypoints in the two images.
     */
    void displayMatches(const cv::Mat &img1, const std::vector<cv::KeyPoint> &keypoints1,
                        const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints2,
                        const std::vector<cv::DMatch> &matches);

    /**
     * Initializes a video capture object with the specified width and height.
     *
     * @param width The desired width of the video capture.
     * @param height The desired height of the video capture.
     * @return A `cv::VideoCapture` object initialized with the specified width and height.
     */
    cv::VideoCapture initializeVideoCapture(int width, int height);

    /**
     * @brief Undistorts an input image using camera calibration data.
     * @param input The input image to be undistorted.
     * @param pathToCalibrationFile The path to the camera calibration file.
     * @return The undistorted image as a cv::Mat object.
     */
    cv::Mat undistortImage(cv::Mat &input, std::string pathToCalibrationFile);

    /**
     * @brief Detects keypoints and descriptors using SIFT.
     * @param input The input image to detect keypoints and descriptors on.
     * @param mask The optional mask specifying where to detect keypoints.
     * @param nFeatures The number of best features to retain.
     * @param nOctaveLayers The number of octave layers within each scale level.
     * @param contrastThreshold The contrast threshold used to filter out weak keypoints.
     * @param edgeThreshold The edge threshold used to filter out edge keypoints.
     * @param sigma The sigma value for Gaussian smoothing of the input image.
     * @return The detected keypoints and descriptors as a KeypointsAndDescriptors object.
     */
    KeypointsAndDescriptors siftDetect(cv::Mat &input, cv::Mat &mask, int nFeatures, int nOctaveLayers,
                                       double contrastThreshold, double edgeThreshold, double sigma) const;

    /**
     * @brief Filters keypoints and descriptors based on a set of indices.
     * @param KpAndDesc The input KeypointsAndDescriptors object to filter.
     * @param indices The indices of the keypoints and descriptors to retain.
     * @return The filtered KeypointsAndDescriptors object.
     */
    KeypointsAndDescriptors filterKeypointsAndDescriptors(KeypointsAndDescriptors &KpAndDesc, const std::vector<int> &indices);

    /**
     * @brief Draws keypoints on an input image.
     * @param input The input image to draw keypoints on.
     * @param keypoints The keypoints to be drawn.
     * @return The image with keypoints drawn as a cv::Mat object.
     */
    cv::Mat drawKeypoints(cv::Mat &input, std::vector<cv::KeyPoint> &keypoints) const;

    /**
     * @brief Saves keypoints and descriptors to a CSV file.
     * @param path The path to the CSV file.
     * @param keypoints The keypoints to be saved.
     * @param descriptors The descriptors to be saved.
     */
    void keypointsAndDescriptorsToCSV(std::string path, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const;

    /**
     * @brief Creates a rectangle mask with a specified size.
     * @param imageSize The size of the image the mask will be applied to.
     * @param rectSize The size of the rectangle in the mask.
     * @return The rectangle mask as a cv::Mat object.
     */
    cv::Mat createRectangleMask(const cv::Size &imageSize, const cv::Size &rectSize);

    /**
     * Calculates the variance of the movement of SIFT features.
     *
     * This function takes a vector of 2D points representing SIFT features and calculates the variance
     * of their movement. The variance is a measure of how spread out the points are from their mean
     * position, indicating the amount of movement in the feature points.
     *
     * @param points A vector of 2D points representing SIFT features.
     * @return The variance of the movement of the SIFT features.
     */
    double calculateSiftFeatureMovementVariance(const std::vector<cv::Point2f> &points);

    /**
     * @brief Updates feature tracks and counts based on matches and current keypoints.
     * @param matches The matches between keypoints in consecutive frames.
     * @param currKeypoints The keypoints in the current frame.
     * @param featureMatchCount The map storing the count of matches for each feature.
     * @param featureTracks The map storing the tracks of each feature.
     */
    void updateFeatureTracksAndCounts(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &currKeypoints,
                                      std::map<int, int> &featureMatchCount, std::map<int, std::vector<cv::Point2f>> &featureTracks);

    /**
     * Saves the feature tracks to a CSV file.
     *
     * @param filePath The file path where the CSV file will be saved.
     * @param featureMatchCount A map containing the count of matches for each feature.
     * @param featureTracks A map containing the feature tracks as a vector of 2D points for each feature.
     * @param minMatches The minimum number of matches required for a feature to be saved.
     */
    void saveFeatureTracksToCSV(const std::string &filePath,
                                const std::map<int, int> &featureMatchCount,
                                const std::map<int, std::vector<cv::Point2f>> &featureTracks,
                                int minMatches);

    /**
     * Reads the top N features from the specified file.
     *
     * @param filepath The path to the file containing the features.
     * @param topN The number of top features to read (default is 15).
     * @return A vector of integers representing the top features.
     */
    std::vector<int> readTopFeatures(const std::string &filepath, int topN = 15);

private:
    const cv::Mat input_;               // The input image
    cv::Mat undistortedInput_;          // The undistorted image
    cv::Mat output_;                    // The output image
    cv::Mat cameraMatrix_;              // The camera matrix for undistortion
    cv::Mat distCoeffs_;                // The distortion coefficients for undistortion
    std::string pathToCalibrationFile_; // The path to the camera calibration file
};