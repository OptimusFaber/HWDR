import os
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import warnings
from collections import defaultdict

# Disable all warnings
warnings.filterwarnings('ignore')

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

def is_valid_overlap(box1, box2):
    """Check if the overlap between two boxes is valid"""
    iou = calculate_iou(box1, box2)
    return iou < 0.05  # Allow overlap of no more than 5%

def add_noise_and_lines(image, boxes, probability=1.0):
    """Add random noise and lines that intersect with digits from below"""
    if random.random() > probability:
        return image
    
    noise_layer = np.zeros_like(image)
    
    if boxes:
        min_x = min(box[0] for box in boxes)
        max_x = max(box[2] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_y = max(box[3] for box in boxes)
        
        # Add horizontal lines under digits
        if random.random() < 0.5:
            num_h_lines = random.randint(2, 4)
            for _ in range(num_h_lines):
                extension = random.randint(20, 50)
                line_start_x = max(0, min_x - extension)
                line_end_x = min(image.shape[1], max_x + extension)
                y_pos = int(max_y + random.randint(5, 20))
                
                num_points = random.randint(4, 7)
                points = []
                for i in range(num_points):
                    x = line_start_x + (line_end_x - line_start_x) * i / (num_points - 1)
                    y = y_pos + random.randint(-2, 2)
                    points.append((int(x), int(y)))
                
                thickness = random.randint(2, 4)
                for i in range(len(points) - 1):
                    cv2.line(noise_layer, points[i], points[i + 1], (255, 255, 255), thickness)
    
    # Add lines
    image = cv2.add(image, noise_layer)
    
    return image

def get_bounding_box(image):
    """Get bounding box coordinates for non-zero pixels with padding 2-4 pixels"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    coords = np.argwhere(gray == 0)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    padding = random.randint(2, 4)
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1] - 1, x_max + padding)
    y_max = min(image.shape[0] - 1, y_max + padding)
    return [x_min, y_min, x_max, y_max]

def convert_to_yolo_format(box, image_width, image_height):
    """Convert bounding box to YOLO format [x_center, y_center, width, height]"""
    x_min, y_min, x_max, y_max = box
    
    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return [x_center, y_center, width, height]

class DigitSelector:
    def __init__(self, dataset_path):
        self.digits = []
        self.labels = []
        self.usage_count = defaultdict(int)
        self.class_indices = defaultdict(list)
        
        # Load digits from dataset
        for digit_class in range(10):
            digit_dir = os.path.join(dataset_path, str(digit_class), str(digit_class))
            if not os.path.exists(digit_dir):
                print(f"Warning: Directory {digit_dir} does not exist")
                continue
                
            for img_name in os.listdir(digit_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(digit_dir, img_name)
                    # Load image with alpha channel
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # If the image has an alpha channel (4 channels)
                        if img.shape[-1] == 4:
                            # Take only the alpha channel
                            alpha = img[:, :, 3]
                            # Create a white image
                            digit = np.ones_like(alpha) * 255
                            # Where alpha > 0, set to 0 (black)
                            digit[alpha > 0] = 0
                        else:
                            # If there is no alpha channel, use as is
                            digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        self.digits.append(digit)
                        self.labels.append(digit_class)
                        self.class_indices[digit_class].append(len(self.digits) - 1)
        
        if not self.digits:
            raise ValueError(f"No images found in {dataset_path}. Please check the dataset path and image files.")
        
        print(f"Loaded {len(self.digits)} digits in total")
        for i in range(10):
            print(f"Class {i}: {len(self.class_indices[i])} digits")
    
    def get_digit(self, class_id=None):
        if class_id is not None:
            available_indices = [idx for idx in self.class_indices[class_id] 
                               if self.usage_count[idx] < 20]
            if not available_indices:
                for idx in self.class_indices[class_id]:
                    self.usage_count[idx] = 0
                available_indices = self.class_indices[class_id]
        else:
            # Choose a random digit class
            available_classes = list(range(10))
            random.shuffle(available_classes)
            for class_id in available_classes:
                available_indices = [idx for idx in self.class_indices[class_id] 
                                   if self.usage_count[idx] < 20]
                if available_indices:
                    break
            if not available_indices:
                self.usage_count.clear()
                available_indices = self.class_indices[class_id]
        
        if not available_indices:
            raise ValueError("No available digits found. This should not happen if the dataset was loaded correctly.")
        
        idx = min(available_indices, key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        return self.digits[idx], self.labels[idx]

def generate_multi_digit_image(digit_selector, target_size=416, min_digits=5, max_digits=8, max_attempts=20):
    """Generate an image with 5-8 digits placed in a row with correct overlap"""
    for attempt in range(max_attempts):
        image = np.ones((target_size, target_size), dtype=np.uint8) * 255
        boxes = []  # [x1, y1, x2, y2, class_id]
        base_height = int(target_size * 0.12)
        num_digits = random.randint(min_digits, max_digits)
        
        # First, gather information about all digits
        digit_info = []
        total_width = 0
        for _ in range(num_digits):
            digit, label = digit_selector.get_digit()
            aspect_ratio = digit.shape[1] / digit.shape[0]
            width = int(base_height * aspect_ratio)
            digit_info.append((width, base_height, digit, label))
            total_width += width
        
        # Base spacing between digits
        spacing = int(base_height * 0.1)
        total_width += spacing * (num_digits - 1)
        
        # Check if all digits fit
        if total_width > target_size * 0.9:  # Leave 10% margin
            continue
        
        # Center horizontally
        start_x = (target_size - total_width) // 2
        start_y = (target_size - base_height) // 2
        
        # First, place all digits with base spacing
        positions = []
        current_x = start_x
        for width, height, digit, label in digit_info:
            positions.append((current_x, width, height, digit, label))
            current_x += width + spacing
        
        # Now add overlap for some pairs of digits
        for i in range(len(positions) - 1):
            if random.random() < 0.9:  # Increase chance of overlap to 90%
                # Randomly choose the degree of overlap
                overlap_type = random.random()
                if overlap_type < 0.3:  # 30% chance of strong overlap
                    overlap = int(positions[i+1][1] * 0.5)  # 50% overlap
                elif overlap_type < 0.6:  # 30% chance of medium overlap
                    overlap = int(positions[i+1][1] * 0.3)  # 30% overlap
                else:  # 40% chance of weak overlap
                    overlap = int(positions[i+1][1] * 0.2)  # 20% overlap
                
                # Shift all subsequent digits
                for j in range(i+1, len(positions)):
                    x, w, h, d, l = positions[j]
                    positions[j] = (x - overlap, w, h, d, l)
        
        # Place digits on the image
        for x, width, height, digit, label in positions:
            # Check if the digit goes out of bounds
            if x + width > target_size:
                continue
                
            # Place the digit while preserving existing pixels
            resized_digit = cv2.resize(digit, (width, height))
            roi = image[start_y:start_y + height, x:x + width]
            # Create a mask for digit pixels (not white)
            mask = resized_digit < 255
            # Update only those pixels where the background is white or where there are digit pixels
            roi[mask] = resized_digit[mask]
            
            # Get bounding box
            box = get_bounding_box(resized_digit)
            if box is not None:
                # Shift bounding box coordinates
                box[0] += x
                box[2] += x
                box[1] += start_y
                box[3] += start_y
                boxes.append(box + [label])
        
        if len(boxes) == num_digits:
            print(f"Successfully placed {len(boxes)} digits")
            return image, boxes
            
    print("Failed to place all digits correctly after several attempts")
    return image, boxes

def main():
    # Create output directories
    output_dir = Path("Datasets/Synthetic-Digits")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize digit selector
    dataset_path = "Datasets/Single-Digits/dataset"
    digit_selector = DigitSelector(dataset_path)
    
    # Generate images
    num_images = 20000  # Reduce the number of images for testing
    for i in tqdm(range(num_images)):
        image, boxes = generate_multi_digit_image(digit_selector)
        
        # Invert the image before adding noise and lines
        image = 255 - image
        
        # Add noise and lines to the final image
        image = add_noise_and_lines(image, boxes)
        
        # Save image
        image_path = images_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save labels in YOLO format
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                x_min, y_min, x_max, y_max, class_id = box
                yolo_box = convert_to_yolo_format([x_min, y_min, x_max, y_max], 
                                                image.shape[1], image.shape[0])
                f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

if __name__ == "__main__":
    main() 