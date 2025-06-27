# Handwritten Digit Detection

## Problem Statement

This project focuses on the task of detecting multiple handwritten digits within a single image. Unlike traditional classification tasks, where the goal is to assign a single label to an entire image (e.g., identifying a single digit in an MNIST image), this problem involves localizing and identifying multiple digits in a complex scene. The challenge lies in accurately detecting the position and identity of each digit, which requires object detection techniques rather than simple classification.

The goal of this repository is to develop a solution for detecting multiple handwritten digits in images, leveraging object detection frameworks to achieve this.

## Approach

To tackle this problem, I have chosen to start with the YOLO (You Only Look Once) object detection model due to its speed and ease of use. YOLO provides a robust starting point for detecting multiple objects (in this case, digits) in an image efficiently. If successful, the generated dataset and initial results can be used to experiment with other models, such as Faster R-CNN or SSD, to compare performance.

### Dataset Generation

To train the model, I created a script that generates a custom dataset by combining handwritten digit images from the MNIST dataset. The script places multiple digits in varied positions on a single image, simulating real-world scenarios where digits appear in different locations and orientations. This dataset is designed to train YOLO to detect and classify digits in a multi-object context.

### Example Image

Below is an example of a generated image from the dataset, showing multiple handwritten digits:

![Example Image](examples/example.png)

## Next Steps

- Train a YOLO model on the generated dataset.
- Evaluate the model's performance in detecting and classifying multiple digits.
- Experiment with other object detection models using the same dataset to compare results.
