#!/bin/bash

# Start roscore in a new gnome-terminal window
gnome-terminal -- bash -c "roscore; exec bash"

# Wait for roscore to initialize
sleep 2

# Start the USB camera node in a new gnome-terminal window
gnome-terminal -- bash -c "roslaunch usb_cam usb_cam-test.launch; exec bash"

# Wait for the camera to initialize
sleep 2

# Start the camera calibration in a new gnome-terminal window
gnome-terminal -- bash -c "rosrun camera_calibration cameracalibrator.py --size 9x6 --square 0.025 image:=/camera/image_raw camera:=/camera --no-service-check; exec bash"
