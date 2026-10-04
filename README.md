# Real-Time Object Detection and Movement Analysis

## Overview

A real-time computer vision system developed using Python, OpenCV and YOLO. The system uses a webcam to detect objects, display bounding boxes and confidence scores, and analyze basic object movement based on changes in object position.

## Features

* Real-time webcam object detection
* YOLO-based object recognition
* Bounding box visualization
* Object class and confidence display
* Object center-position calculation
* Basic movement analysis
* Moving/Stationary status detection

## Technologies Used

* Python
* OpenCV
* YOLO
* Ultralytics
* NumPy

## How It Works

```text
Webcam
   ↓
OpenCV
   ↓
YOLO Object Detection
   ↓
Bounding Boxes
   ↓
Object Center Position
   ↓
Position Comparison
   ↓
MOVING / STATIONARY
```

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

Activate the virtual environment and run:

```bash
python object_detection.py
```

The webcam will open and the system will detect objects in real time.

Press `Q` to stop the program.

## Project Output

The system displays:

* Detected object name
* Confidence score
* Bounding box
* Movement status

## Future Improvements

* Multi-object tracking with unique IDs
* Improved movement tracking
* Object counting
* Video file processing
* Distance estimation
* Geospatial/drone-image analysis
