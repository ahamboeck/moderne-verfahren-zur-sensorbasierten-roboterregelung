# moderne-verfahren-zur-sensorbasierten-roboterregelung

## Homework 3

### How to compile

```bash
cd Homework_3/poseEstimate/src
make
```

### Modes of Operation

#### Filter Mode

In `filter` mode, the application uses handpicked and previously filtered keypoints to identify and visualize specific sift features from an image. The selection process ensures that only the most significant keypoints remain. In filter mode also pose estimation is performed.

#### Save Mode

In `save` mode, the application filters keypoints based on specific metrics such as variance and displacement properties. This mode saves filtered keypoints into a CSV file, providing a carefully selected set of keypoints that can be reused later.

#### Use Mode

In `use` mode, the application leverages all previously saved and filtered keypoints to streamline the handpicking process. By reusing these curated keypoints, users can directly work with a focused subset that makes feature matching and tracking easier.

### Usage

To run the application in a specific mode (default is filter), use the following command-line syntax:

```bash
./poseEstimate <mode>
```
Important:
Resolution for calibration and distortion must be the same!

## Usage for OpenCV_Hello_World

To compile the program, use the following command in the terminal:
```bash
make
```
After compiling, you can start the program by running:
```bash
./helloWorld <path-to-image>
```
Replace <path-to-image> with the actual path to the image file you want to display. For example:
```bash
./helloWorld /path/to/your/image.jpg
```
This will open a window displaying the specified image.

