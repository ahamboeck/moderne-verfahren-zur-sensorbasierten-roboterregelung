#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const std::string filePath = "/home/fhtw_user/moderne-verfahren-zur-sensorbasierten-roboterregelung/Homework_3/camera_calib_data/ost.yaml";
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open file." << std::endl;
        return -1;
    }

    // Camera Matrix
    if (fs["camera_matrix"]["data"].isSeq() && fs["camera_matrix"]["rows"].isInt() && fs["camera_matrix"]["cols"].isInt()) {
        cv::Mat cameraMatrix;
        fs["camera_matrix"]["data"] >> cameraMatrix;
        if (!cameraMatrix.empty()) {
            cameraMatrix = cameraMatrix.reshape(0, fs["camera_matrix"]["rows"].real());
            std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;
        } else {
            std::cerr << "Camera matrix data is empty or incorrect." << std::endl;
        }
    } else {
        std::cerr << "Camera matrix is not accessible or has incorrect format." << std::endl;
    }

    // Distortion Coefficients
    if (fs["distortion_coefficients"]["data"].isSeq()) {
        cv::Mat distCoeffs;
        fs["distortion_coefficients"]["data"] >> distCoeffs;
        std::cout << "Distortion Coefficients:\n" << distCoeffs << std::endl;
    } else {
        std::cerr << "Distortion coefficients data is not accessible or incorrect." << std::endl;
    }

    // Rectification Matrix
    if (fs["rectification_matrix"]["data"].isSeq()) {
        cv::Mat rectificationMatrix;
        fs["rectification_matrix"]["data"] >> rectificationMatrix;
        if (!rectificationMatrix.empty()) {
            rectificationMatrix = rectificationMatrix.reshape(0, 3); // Assuming it's 3x3
            std::cout << "Rectification Matrix:\n" << rectificationMatrix << std::endl;
        } else {
            std::cerr << "Rectification matrix data is empty or incorrect." << std::endl;
        }
    } else {
        std::cerr << "Rectification matrix is not accessible or has incorrect format." << std::endl;
    }

    // Projection Matrix
    if (fs["projection_matrix"]["data"].isSeq()) {
        cv::Mat projectionMatrix;
        fs["projection_matrix"]["data"] >> projectionMatrix;
        if (!projectionMatrix.empty()) {
            projectionMatrix = projectionMatrix.reshape(0, 3); // Assuming it's 3x4
            std::cout << "Projection Matrix:\n" << projectionMatrix << std::endl;
        } else {
            std::cerr << "Projection matrix data is empty or incorrect." << std::endl;
        }
    } else {
        std::cerr << "Projection matrix is not accessible or has incorrect format." << std::endl;
    }

    fs.release();
    return 0;
}
