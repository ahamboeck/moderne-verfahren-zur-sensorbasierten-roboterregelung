#!/bin/bash

# Start roscore in a new xterm window
xterm -hold -e "roscore" &

# Wait for roscore to initialize
sleep 2

# Start the USB camera node in a new xterm window
xterm -hold -e "roslaunch usb_cam usb_cam-test.launch" &

# Wait for the camera to initialize
sleep 2

# Start the camera calibration in a new xterm window
xterm -hold -e "rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.108 image:=/camera/image_raw camera:=/camera --no-service-check" &
