# Handwritten Digit Detection Project

## Problem Statement

This project aims to detect multiple handwritten digits in a single image using object detection techniques. Unlike classification tasks that assign a single label to an image, this task involves localizing and identifying multiple digits within complex scenes. The challenge is to accurately detect the position and identity of each digit.

## Issues Encountered

During the development and training process, several issues were identified:
- The model consistently fails to recognize the digit "1".
- Frequent confusion between digits: "4" is mistaken for "9", "9" for "3", and "7" for "4".
- Models trained on synthetic data (e.g., MNIST-based) perform worse than those trained on the Connected Digits dataset.
- Data augmentations critically impact performance, with some augmentations degrading training outcomes.
- Training instability: the model "forgets" certain digits on later epochs, leading to inconsistent performance.

## Actions Taken

### Dataset Transition

Initially, the MNIST dataset was used to generate synthetic images for training. However, MNIST has significant limitations, as outlined in this article: [Why MNIST is the Worst Thing That Has Ever Happened to Humanity](https://matteo-a-barbieri.medium.com/why-mnist-is-the-worst-thing-that-has-ever-happened-to-humanity-49fd053f0f66). Below is an example of an MNIST-based image, highlighting its simplistic and unrealistic nature:

![MNIST Example](examples/mnist.jpg)

Due to these limitations, MNIST was replaced with a more robust dataset: [Handwritten Digits Dataset (Not in MNIST)](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist), available for download at [GitHub - Handwritten-Digit-Dataset v1.2.0](https://github.com/JC-ProgJava/Handwritten-Digit-Dataset/releases/tag/v1.2.0) in the `dataset.zip` file.

### Dataset Generation

A new dataset of 20,000 images was created by combining the [Handwritten Digits Dataset (Not in MNIST)](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist) with the [Touching Digits Dataset](https://web.inf.ufpr.br/vri/databases/touching-digits/). A script was developed to overlay digits from both datasets onto images, simulating complex scenes with multiple digits. 

Below is an example from the Touching Digits dataset:

![Touching Digits Example](examples/TouchingDigits.png)

To download and use the Handwritten Digits Dataset (Not in MNIST), you can follow these steps:

```bash
# Download the dataset from GitHub
wget https://github.com/JC-ProgJava/Handwritten-Digit-Dataset/releases/download/v1.2.0/dataset.zip

# Unzip the dataset
unzip dataset.zip -d handwritten_digits_dataset
```

### Updated Results

Training was conducted on the new dataset of 20,000 images for 15 epochs. The model shows improved digit recognition compared to previous experiments. However, a new issue has emerged: the model tends to classify non-digit objects as digits, leading to false positives. The results are shown below:

![Results Image-1](examples/result-1.png)

![Results Image-2](examples/result-2.png)

## Next Steps

- Address the issue of false positives by refining the dataset or adjusting the model’s classification thresholds.
- Experiment with different augmentation strategies to identify which ones improve performance without degrading results.
- Investigate techniques to stabilize training and prevent the model from "forgetting" digits on later epochs.
- Test alternative object detection models (e.g., Faster R-CNN, SSD) to compare performance with YOLO.