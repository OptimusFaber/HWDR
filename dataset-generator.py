import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import mnist
import albumentations as A
import random
from tqdm import tqdm

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

def create_augmentation_pipeline():
    """Create augmentation pipeline using albumentations"""
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.ISONoise(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.Rotate(limit=15, p=0.5),
            A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=0.5),
        ], p=0.3),
    ])

def generate_multi_digit_image(digits, labels, target_size=416, min_height_ratio=0.04, max_height_ratio=0.12, max_overlap=0.05):
    """Generate image with multiple digits with specified constraints"""
    image = np.zeros((target_size, target_size), dtype=np.uint8)
    boxes = []  # [x1, y1, x2, y2, class_id]
    
    min_height = int(target_size * min_height_ratio)
    max_height = int(target_size * max_height_ratio)
    
    num_digits = random.randint(1, 5)  # Random number of digits per image
    attempts = 0
    max_attempts = 100
    
    while len(boxes) < num_digits and attempts < max_attempts:
        attempts += 1
        
        # Select random digit
        idx = random.randint(0, len(digits) - 1)
        digit = digits[idx]
        label = labels[idx]
        
        # Random size within constraints
        height = random.randint(min_height, max_height)
        aspect_ratio = digit.shape[1] / digit.shape[0]
        width = int(height * aspect_ratio)
        
        # Random position
        x = random.randint(0, target_size - width)
        y = random.randint(0, target_size - height)
        
        new_box = [x, y, x + width, y + height]
        
        # Check overlap with existing boxes
        overlap = False
        for box in boxes:
            if calculate_iou(new_box, box[:4]) > max_overlap:
                overlap = True
                break
        
        if not overlap:
            # Resize digit
            digit_resized = cv2.resize(digit, (width, height))
            
            # Place digit on image
            mask = digit_resized > 0
            image[y:y+height, x:x+width][mask] = digit_resized[mask]
            
            # Add box with class
            boxes.append([x, y, x + width, y + height, label])
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image, boxes

def convert_to_yolo_format(boxes, image_width, image_height):
    """Convert boxes to YOLO format (class_id, x_center, y_center, width, height)"""
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        
        # Normalize coordinates
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        yolo_boxes.append([class_id, x_center, y_center, width, height])
    
    return yolo_boxes

def main():
    # Create output directories
    output_dir = Path("dataset_yolo")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    for dir_path in [images_dir, labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Combine train and test data
    digits = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Generate multi-digit images
    num_multi_images = 5000
    print("Generating multi-digit images...")
    for i in tqdm(range(num_multi_images)):
        image, boxes = generate_multi_digit_image(digits, labels)
        
        # Convert boxes to YOLO format
        yolo_boxes = convert_to_yolo_format(boxes, 416, 416)
        
        # Save image
        cv2.imwrite(str(images_dir / f"multi_{i}.jpg"), image)
        
        # Save labels
        with open(labels_dir / f"multi_{i}.txt", 'w') as f:
            for box in yolo_boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
    
    # Generate single-digit images with augmentation
    num_single_images = 5000
    print("\nGenerating single-digit images with augmentation...")
    for i in tqdm(range(num_single_images)):
        # Select random digit
        idx = random.randint(0, len(digits) - 1)
        digit = digits[idx]
        label = labels[idx]
        
        # Create empty image
        image = np.zeros((416, 416), dtype=np.uint8)
        
        # Random size within constraints
        height = random.randint(int(416 * 0.04), int(416 * 0.12))
        aspect_ratio = digit.shape[1] / digit.shape[0]
        width = int(height * aspect_ratio)
        
        # Random position
        x = random.randint(0, 416 - width)
        y = random.randint(0, 416 - height)
        
        # Resize digit
        digit_resized = cv2.resize(digit, (width, height))
        
        # Place digit on image
        image[y:y+height, x:x+width] = digit_resized
        
        # Convert to RGB before augmentation
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply augmentation
        transformed = transform(image=image)
        image = transformed['image']
        
        # Convert box to YOLO format
        box = [x, y, x + width, y + height, label]
        yolo_box = convert_to_yolo_format([box], 416, 416)[0]
        
        # Save image
        cv2.imwrite(str(images_dir / f"single_{i}.jpg"), image)
        
        # Save label
        with open(labels_dir / f"single_{i}.txt", 'w') as f:
            f.write(f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}\n")

if __name__ == "__main__":
    main() 