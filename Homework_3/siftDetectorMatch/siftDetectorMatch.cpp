#include "siftDetectorMatch.h"

SiftDetectorMatch::SiftDetectorMatch(){};

SiftDetectorMatch::SiftDetectorMatch(int width = 640, int height = 480, int device = 0)
{
    cap.open(device);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
    }
}

SiftDetectorMatch::~SiftDetectorMatch()
{
    cap.release();
    std::cout << "Camera released" << std::endl;
}

cv::Mat SiftDetectorMatch::getFrame()
{
    cap.read(frame);
    return frame;
}















int main(int, char **)
{
    cv::Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    // open the default camera using default API
    cap.open(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    // OR advance usage: select any API backend
    //  int deviceID = 0; // 0 = open default camera
    //  int apiID = cv::CAP_ANY; // 0 = autodetect default API
    // open selected camera using selected API
    //  cap.open(deviceID, apiID);
    //  // check if we succeeded
    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    //--- GRAB AND WRITE LOOP
    std::cout << "Start grabbing" << std::endl
              << "Press any key to terminate" << std::endl;
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty())
        {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);
        if (cv::waitKey(5) >= 0)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}