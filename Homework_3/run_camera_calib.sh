#!/bin/bash

# Start roscore in the background
roscore &
roscore_pid=$!

# Wait for roscore to initialize
sleep 2

# Start the USB camera node in the background
roslaunch usb_cam usb_cam-test.launch &
usb_cam_pid=$!

# Wait for the camera to initialize
sleep 2

# Start the camera calibration in the background
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.108 image:=/camera/image_raw camera:=/camera --no-service-check &
calibration_pid=$!

# Optional: Wait for a user input to kill all background processes before the script exits
read -p "Press any key to terminate all running processes..."
kill $roscore_pid $usb_cam_pid $calibration_pid
