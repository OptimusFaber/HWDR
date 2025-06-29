# Handwritten Digit Detection

## Problem Statement

This project focuses on the task of detecting multiple handwritten digits within a single image. Unlike traditional classification tasks, where the goal is to assign a single label to an entire image (e.g., identifying a single digit in an MNIST image), this problem involves localizing and identifying multiple digits in a complex scene. The challenge lies in accurately detecting the position and identity of each digit, which requires object detection techniques rather than simple classification.

The goal of this repository is to develop a solution for detecting multiple handwritten digits in images, leveraging object detection frameworks to achieve this.

## Approach

To tackle this problem, I have chosen to start with the YOLO (You Only Look Once) object detection model due to its speed and ease of use. YOLO provides a robust starting point for detecting multiple objects (in this case, digits) in an image efficiently. If successful, the generated dataset and initial results can be used to experiment with other models, such as Faster R-CNN or SSD, to compare performance.

### Dataset Generation

To train the model, I created a script that generates a custom dataset by combining handwritten digit images from the MNIST dataset and a new dataset called [Touching Digits Dataset](https://web.inf.ufpr.br/vri/databases/touching-digits/). The MNIST dataset consists of 500,000 images, each containing multiple digits placed in varied positions to simulate real-world scenarios. The Touching Digits dataset, comprising 240,000 images, includes handwritten digits that may overlap or touch, adding complexity to the detection task.

Below is an example from the Touching Digits dataset:

![Touching Digits Example](examples/TouchingDigits.png)

#### Dataset Updates

The dataset has been enhanced to increase robustness and simulate more challenging conditions:
- **Background Noise**: Added random lines behind the digits and various shapes (circles, squares, rectangles) to mimic cluttered backgrounds.
- **Salt and Pepper Noise**: Introduced Salt and Pepper noise, a type of impulse noise where random pixels in the image are set to either black or white, resembling specks of salt and pepper. This noise simulates scenarios where image binarization (e.g., converting to black-and-white) may fail due to pixel-level distortions, making detection more challenging.

Below is an example of an image with Salt and Pepper noise applied:

![Salt and Pepper Noise Example](examples/saltnpepper.jpg)

#### Data Augmentation

To improve model generalization, I applied specific data augmentations during dataset generation:
- **Scale**: Adjusted the size of digits to simulate variations in digit size.
- **Translate**: Shifted digits to different positions within the image to account for positional variability.

Other augmentations (e.g., rotation, shear) were disabled, as they were found to negatively impact digit recognition performance.

### Example Image

Below is an updated example of a generated image from the combined dataset, showing multiple handwritten digits with added background noise:

![Example Image](examples/example.jpg)

From experiments conducted, it is evident that the model struggles to accurately detect handwritten digits. Specifically:
- The model consistently fails to detect the digit "1".
- There is frequent confusion between digits, with the digit "4" often misclassified as "9", and the digit "9" mistaken for "3".
- The model has increasingly confused the digit "7" with "4", which may indicate overfitting due to the large dataset size.

### Results

The model trained on the combined dataset (500,000 MNIST images + 240,000 Touching Digits images) performs poorly on the test set, as shown below:

![Results Image](examples/results.png)

The poor performance suggests potential overfitting, likely due to the large volume of training data (740,000 images in total).

## Next Steps

- Reduce the dataset size to 20,000–40,000 images to mitigate overfitting and improve model generalization.
- Retrain the YOLO model on the smaller dataset with added noise and selected augmentations.
- Evaluate the model's performance in detecting and classifying multiple digits under noisy conditions.
- Experiment with other object detection models (e.g., Faster R-CNN, SSD) using the reduced dataset to compare results.