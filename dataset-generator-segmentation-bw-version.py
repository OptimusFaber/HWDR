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

class DigitSelector:
    def __init__(self, dataset_path):
        self.digits = []
        self.masks = []  # Add mask storage
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
                        # If image has alpha channel (4 channels)
                        if img.shape[-1] == 4:
                            # Take only alpha channel
                            alpha = img[:, :, 3]
                            # Create white image
                            digit = np.ones_like(alpha) * 255
                            # Where alpha > 0, set to 0 (black)
                            digit[alpha > 0] = 0
                            # Create mask
                            mask = (alpha > 0).astype(np.uint8) * 255
                        else:
                            # If no alpha channel, use as is
                            digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # Create mask from the image itself
                            _, mask = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY_INV)
                        
                        self.digits.append(digit)
                        self.masks.append(mask)
                        self.labels.append(digit_class)
                        self.class_indices[digit_class].append(len(self.digits) - 1)
        
        if not self.digits:
            raise ValueError(f"No images found in {dataset_path}. Check the dataset path and image files.")
        
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
            # Select a random digit class
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
            raise ValueError("No available digits. This should not happen if the dataset was loaded correctly.")
        
        idx = min(available_indices, key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        return self.digits[idx], self.masks[idx], self.labels[idx]

def check_alignment(positions, base_height, tolerance=5):
    """Checks that all digits are on the same line"""
    if not positions:
        return True
    # Take the middle line of each position
    mid_lines = [(pos[0] + base_height/2) for pos in positions]
    # Check that all middle lines are within tolerance from the first one
    reference = mid_lines[0]
    return all(abs(mid - reference) <= tolerance for mid in mid_lines)

def add_noise_and_lines(image, probability=1.0):
    """Adds random noise and lines"""
    if random.random() > probability:
        return image
    
    noise_layer = np.zeros_like(image)
    
    # Find areas with digits
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find common area with digits
        x_coords = []
        y_coords = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
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

def get_contour_points(mask, simplify=True):
    """Gets contour points from mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    if simplify:
        # Simplify contour to reduce number of points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert to list of points
    points = contour.reshape(-1, 2)
    return points

def generate_multi_digit_image(digit_selector, target_size=416, min_digits=5, max_digits=8, max_attempts=50):
    """Generates image with digits and their segmentation masks"""
    for attempt in range(max_attempts):
        image = np.ones((target_size, target_size), dtype=np.uint8) * 255
        digit_masks = []  # List of masks and their classes
        num_digits = random.randint(min_digits, max_digits)
        
        base_height = int(target_size * 0.12)
        
        # First collect information about all digits
        digit_info = []
        total_width = 0
        for _ in range(num_digits):
            digit, digit_mask, label = digit_selector.get_digit()
            aspect_ratio = digit.shape[1] / digit.shape[0]
            width = int(base_height * aspect_ratio)
            digit_info.append((width, base_height, digit, digit_mask, label))
            total_width += width
        
        # Base spacing between digits
        spacing = int(base_height * 0.1)
        total_width += spacing * (num_digits - 1)
        
        # Check if all digits will fit
        if total_width > target_size * 0.9:  # Leave 10% margin
            continue
        
        # Center horizontally and vertically
        start_x = (target_size - total_width) // 2
        start_y = (target_size - base_height) // 2
        
        # First place all digits with base spacing
        positions = []
        current_x = start_x
        for width, height, digit, digit_mask, label in digit_info:
            positions.append((current_x, start_y, width, height, digit, digit_mask, label))
            current_x += width + spacing
        
        # Now add overlap for some digit pairs
        for i in range(len(positions) - 1):
            if random.random() < 0.9:  # Increase overlap chance to 90%
                # Randomly choose overlap degree
                overlap_type = random.random()
                if overlap_type < 0.3:  # 30% chance of strong overlap
                    overlap = int(positions[i+1][2] * 0.5)  # 50% overlap
                elif overlap_type < 0.6:  # 30% chance of medium overlap
                    overlap = int(positions[i+1][2] * 0.3)  # 30% overlap
                else:  # 40% chance of weak overlap
                    overlap = int(positions[i+1][2] * 0.2)  # 20% overlap
                
                # Shift all subsequent digits
                for j in range(i+1, len(positions)):
                    x, y, w, h, d, m, l = positions[j]
                    positions[j] = (x - overlap, y, w, h, d, m, l)
        
        # Check alignment
        if not check_alignment([(y, h) for _, y, _, h, _, _, _ in positions], base_height):
            continue
            
        # Place digits on image and create masks
        for x, y, width, height, digit, digit_mask, label in positions:
            # Check if digit goes beyond boundaries
            if x + width > target_size:
                continue
                
            # Place digit
            resized_digit = cv2.resize(digit, (width, height))
            resized_mask = cv2.resize(digit_mask, (width, height))
            
            # Place digit on image
            roi = image[y:y + height, x:x + width]
            digit_pixels = resized_digit < 255
            roi[digit_pixels] = resized_digit[digit_pixels]
            
            # Create full mask for digit
            full_mask = np.zeros((target_size, target_size), dtype=np.uint8)
            full_mask[y:y + height, x:x + width] = resized_mask
            
            # Get contour and add to list
            points = get_contour_points(full_mask)
            if points is not None:
                # Normalize coordinates
                points = points.astype(float)
                points[:, 0] /= target_size  # x coordinates
                points[:, 1] /= target_size  # y coordinates
                digit_masks.append((label, points))
        
        # Invert image before adding noise and lines
        image = 255 - image
        
        # Add noise and lines to final image
        image = add_noise_and_lines(image)
        
        return image, digit_masks

def main():
    # Create output directories
    output_dir = Path("/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/yolo-dataset-v6-seg")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize digit selector
    dataset_path = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Single-Digits/dataset"
    digit_selector = DigitSelector(dataset_path)
    
    # Generate images
    num_images = 20000
    for i in tqdm(range(num_images)):
        image, digit_masks = generate_multi_digit_image(digit_selector)
        
        # Save image
        image_path = images_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save masks in YOLO segmentation format
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, 'w') as f:
            for label, points in digit_masks:
                # Write class and point coordinates separated by space
                points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in points])
                f.write(f"{label} {points_str}\n")

if __name__ == "__main__":
    main() 