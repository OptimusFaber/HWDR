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

### Datasets Used

Initially, the MNIST dataset was used to generate synthetic images for training. However, MNIST has significant limitations, as outlined in this article: [Why MNIST is the Worst Thing That Has Ever Happened to Humanity](https://matteo-a-barbieri.medium.com/why-mnist-is-the-worst-thing-that-has-ever-happened-to-humanity-49fd053f0f66). Below is an example of an MNIST-based image, highlighting its simplistic and unrealistic nature:

![MNIST Example](examples/mnist.png)

To create a robust dataset for training, the following datasets were utilized:
- **[Handwritten Digits Dataset (Not in MNIST)](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist)**: A dataset of handwritten digits, available for download at [GitHub - Handwritten-Digit-Dataset v1.2.0](https://github.com/JC-ProgJava/Handwritten-Digit-Dataset/releases/tag/v1.2.0) in the `dataset.zip` file.
- **[Touching Digits Dataset](https://web.inf.ufpr.br/vri/databases/touching-digits/)**: A dataset containing handwritten digits that may overlap or touch, simulating complex detection scenarios.
- **[English Handwritten Characters Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset?select=english.csv)**: A dataset of handwritten digits and letters, used to introduce realistic background noise and character variations.
- **Custom Handwritten Digits Dataset**: A proprietary dataset of handwritten digits, created to further diversify the training data and enhance model robustness.

![Elements from above datasets](examples/datasets.png)

### Dataset Generation

A dataset of 16,000 images was created by combining the above datasets, including the custom handwritten digits dataset. A script was developed to overlay digits and letters from these datasets onto images, simulating complex scenes with multiple characters. To improve robustness, the English Handwritten Characters dataset was used to introduce background noise, as I noticed that without this, the model was overly sensitive to non-digit elements (e.g., random shapes or patterns). Letters and quotation marks were included in the images but not labeled as a separate class to train the model to ignore them.

The example of dataset frame:

![Dataset Image Example](examples/example.png)

#### Initial Binary Dataset

The first version of the dataset was binary, where images were converted to black-and-white to simplify processing. However, I decided to move away from this format because binarization results in significant information loss. For example, subtle variations in stroke thickness, shading, and texture of handwritten digits are discarded, which reduces the model’s ability to generalize to real-world scenarios where digits may appear on varied backgrounds or with different writing styles. This loss of detail also made it harder for the model to distinguish between similar digits (e.g., "4" and "9") and contributed to poor performance in noisy environments.

#### Transition to Color Dataset

To address the limitations of the binary dataset, I transitioned to a color dataset. The background is generated to resemble the texture and appearance of a paper sheet, incorporating realistic elements like faint lines, creases, or slight color variations. The digits and letters are rendered in various colors and styles to mimic real-world writing instruments (e.g., pens or markers with different stroke widths and forms). In the updated script, digits and letters from the new custom handwritten digits dataset and other sources are generated with varied colors and forms to imitate pen writing. Quotation marks were also added to the images to train the model to ignore them. This approach offers several advantages:
- **Improved Realism**: The paper-like background better simulates real-world conditions, making the model more robust to variations in lighting, texture, and noise.
- **Enhanced Character Representation**: Colored digits and letters preserve more visual information, such as stroke intensity and slight variations in hue, which help the model distinguish between similar characters.
- **Better Generalization**: Training on color images with realistic backgrounds reduces overfitting to simplistic or artificial patterns, improving performance on diverse test sets.

Below is an example of an image from the color dataset, showcasing the paper-like background and digits/letters rendered to imitate pen writing:

![Pen Imitation Example](examples/pen_imitation.png)

### Model Development

Two models were prepared to address the digit detection task: one for object detection and another for segmentation.

#### Detection Model

The detection model, trained on the combined dataset of 16,000 images, shows reduced sensitivity to background noise compared to previous iterations. However, new issues have emerged:
- The model struggles to distinguish between digits, often failing to recognize them correctly.
- In some cases, the model fails to detect digits entirely, particularly on later epochs where predictions disappear, indicating potential overfitting.

Below is an example illustrating these issues:

![Detection Result](examples/detection.png)

#### Segmentation Model

A segmentation model was also trained to explore an alternative approach. This model demonstrates improved robustness to background noise compared to earlier versions but faces significant challenges:
- The model frequently misclassifies non-digit characters, such as the Russian letter "й" and the upper part of the letter "р", as digits.
- Training is unstable, with performance fluctuating between epochs. For example, early epochs fail to detect digits like "5" and "0", while later epochs miss "2" and "5", indicating overfitting as training progresses.

Below is an example of the segmentation model's output:

![Segmentation Result](examples/segmentation.png)

### Updated Results with Color Dataset

The updated color dataset, now incorporating digits from the custom handwritten digits dataset and including quotation marks to desensitize the model to non-digit elements, has led to some improvements in the segmentation model compared to previous iterations. However, the overall performance remains unsatisfactory. The segmentation model still struggles with misclassifying non-digit characters, such as the Russian letter "й" and parts of the letter "р", as digits (often assigning them classes like "1" or "4"). This may be due to thin segmentation masks, suggesting that more robust dataset annotation is needed. Additionally, the model shows signs of overfitting, as evidenced by the shifting detection patterns across epochs (e.g., missing "5" and "0" early on, then "2" and "5" later). The detection model performs even worse than the segmentation model, with a noticeable trend of failing to detect digits on later epochs, despite reasonable predictions early in training. The misclassification of the letter "й" as a digit persists in both models. These results suggest that the dataset size may still be too large, leading to overfitting, and that increasing the number of digits per image could improve performance.

## Next Steps

- Reduce the dataset size (e.g., to 8,000–12,000 images) to mitigate overfitting and experiment with including more digits per image to improve detection robustness.
- Refine the color dataset annotation process to create more precise and robust segmentation masks, addressing issues with thin mask edges.
- Experiment with targeted augmentations (e.g., color jitter, subtle rotations) to enhance model generalization without degrading performance.
- Make bigger test to find the other problems of my model.
- Test alternative object detection and segmentation models (e.g., Faster R-CNN, SSD, or U-Net) to compare performance with the current YOLO-based approach.