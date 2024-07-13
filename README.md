# Real-time-road-lane-and-Human-detection-using-OpenCV-and-YOLO


## Introduction

This project implements real-time road lane and human detection using OpenCV and the YOLO (You Only Look Once) object detection model. The system can be used for applications such as autonomous driving, traffic monitoring, and pedestrian safety systems.

## Features

- Real-time detection of road lanes and humans.
- Utilizes YOLO for human detection.
- Employs OpenCV for image processing and lane detection.
- Efficient and accurate detection suitable for real-time applications.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- YOLOv3 model files (`yolov3.weights`, `yolov3.cfg`, `coco.names`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ChakraborttySouvik/Real-time-road-lane-and-Human-detection-using-OpenCV-and-YOLO
   cd Real-time-road-lane-and-Human-detection-using-OpenCV-and-YOLO
Install the required packages:

bash
Copy code
pip install opencv-python numpy
Download the YOLOv3 model files:

yolov3.weights
yolov3.cfg
coco.names
Place the YOLOv3 model files in the project directory.

Usage
Ensure the YOLOv3 model files (yolov3.weights, yolov3.cfg, coco.names) are in the project directory.

Modify the yolo.py script to point to the correct paths for your input video and model files:

python
Copy code
video_path = "path/to/your/video.mp4"
config_path = "path/to/yolov3.cfg"
weights_path = "path/to/yolov3.weights"
names_path = "path/to/coco.names"
Run the yolo.py script:

bash
Copy code
python yolo.py
The script will process the input video and display the real-time road lane and human detection results.

Project Structure
yolo.py: Main script for running the YOLO-based human detection and OpenCV-based lane detection.
yolov3.cfg: YOLOv3 configuration file.
coco.names: File containing the names of the classes recognized by YOLO.
Example
python
Copy code
# Example usage
python yolo.py
The above command will run the detection on the specified input video and display the results in a window.

Acknowledgements
YOLO (You Only Look Once)
OpenCV
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions or feedback, please contact [Souvik.chakrabortty123@gmail.com]



















