# Tennis Analysis Project

This Tennis Analysis Project utilizes advanced computer vision and machine learning techniques to track and analyze tennis players and ball movements throughout video sequences. The project integrates YOLO models for object detection, Depth Anything for depth estimation, and CNNs for court line detection to provide comprehensive 2D and 3D analytics.

## Features

- **Player Tracking**: Uses YOLOv8 to detect, track, and visualize tennis players in video sequences.
- **Ball Tracking and Interpolation**: Employs YOLOv5 to track the tennis ball and interpolates its position for frames where it's not detected.
- **Court Line Detection**: Leverages retrained ResNet50 and DenseNet121 models to detect key points on the tennis court from monocular 2D footage.
- **Depth Estimation**: Integrates the Depth Anything model to estimate and add depth information (z-coordinate) to the tennis ball's trajectory, transitioning from 2D to 3D tracking.
- **Visualization**: Uses OpenCV to draw bounding boxes, player information, tennis ball trajectory, and court keypoints on new video frames.

## Getting Started

### Prerequisites

Before running the Tennis Analysis Project, ensure you have the following installed:

- Python 3.8 or higher
- OpenCV
- PyTorch
- torchvision
- ultralytics
- pandas
- numpy

You can install the necessary Python libraries using:

```bash
pip install opencv-python torch torchvision ultralytics pandas numpy
