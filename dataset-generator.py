import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import mnist
import random
from tqdm import tqdm
import warnings

# Отключаем все предупреждения
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
    # Calculate IoU
    iou = calculate_iou(box1, box2)
    
    # Check if one box is mostly inside another
    x1_center = (box1[0] + box1[2]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    
    # Calculate horizontal overlap percentage
    overlap_width = min(box1[2], box2[2]) - max(box1[0], box2[0])
    min_width = min(box1[2] - box1[0], box2[2] - box2[0])
    overlap_percentage = overlap_width / min_width if min_width > 0 else 0
    
    # Conditions for valid overlap:
    # 1. IoU should be small (digits don't overlap too much)
    # 2. Horizontal overlap should be small
    # 3. Centers should be reasonably far apart
    return (iou < 0.3 and 
            overlap_percentage < 0.4 and 
            abs(x1_center - x2_center) > min_width * 0.4)

def add_noise_and_lines(image, boxes, probability=0.5):
    """Add random noise and lines that intersect with digits from below"""
    if random.random() > probability:
        return image
    
    # Create a separate layer for noise and lines
    noise_layer = np.zeros_like(image)
    
    # Get the overall region where digits are located
    if boxes:
        min_x = min(box[0] for box in boxes)
        max_x = max(box[2] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_y = max(box[3] for box in boxes)
        
        # Add horizontal lines under digits (50% chance)
        if random.random() < 0.5:
            num_h_lines = random.randint(2, 4)
            for _ in range(num_h_lines):
                # Line starts and ends beyond the digit group
                extension = random.randint(20, 50)
                line_start_x = max(0, min_x - extension)
                line_end_x = min(image.shape[1], max_x + extension)
                
                # Position line below digits
                y_pos = int(max_y + random.randint(5, 20))
                
                # Create wavy line
                num_points = random.randint(4, 7)
                points = []
                for i in range(num_points):
                    x = line_start_x + (line_end_x - line_start_x) * i / (num_points - 1)
                    y = y_pos + random.randint(-2, 2)  # Small vertical variation
                    points.append((int(x), int(y)))
                
                # Draw the wavy line with varying thickness
                thickness = random.randint(2, 4)
                for i in range(len(points) - 1):
                    cv2.line(noise_layer, points[i], points[i + 1], (255, 255, 255), thickness)
        
        # Add random artifacts (shapes and letters) near digits but not intersecting
        num_artifacts = random.randint(3, 8)
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'X', 'Y', 'Z']
        
        for _ in range(num_artifacts):
            # Choose position near digits but not intersecting
            art_size = random.randint(10, 20)
            
            # Decide if artifact should be above, below, or to the sides
            position = random.choice(['above', 'below', 'sides'])
            
            if position == 'above':
                art_x = random.randint(int(min_x - 20), int(max_x + 20))
                art_y = int(min_y - art_size - random.randint(5, 15))
            elif position == 'below':
                art_x = random.randint(int(min_x - 20), int(max_x + 20))
                art_y = int(max_y + random.randint(5, 15))
            else:  # sides
                if random.random() < 0.5:  # left side
                    art_x = int(min_x - art_size - random.randint(5, 15))
                else:  # right side
                    art_x = int(max_x + random.randint(5, 15))
                art_y = random.randint(int(min_y - 10), int(max_y + 10))
            
            # Draw random artifact (shape or letter)
            shape_type = random.choice(['circle', 'rectangle', 'line', 'letter', 'letter'])  # Higher chance for letters
            if shape_type == 'circle':
                cv2.circle(noise_layer, (art_x, art_y), art_size // 2, (255, 255, 255), -1)
            elif shape_type == 'rectangle':
                cv2.rectangle(noise_layer, (art_x, art_y), 
                            (art_x + art_size, art_y + art_size), (255, 255, 255), -1)
            elif shape_type == 'line':
                end_x = art_x + random.randint(-20, 20)
                end_y = art_y + random.randint(-20, 20)
                cv2.line(noise_layer, (art_x, art_y), (end_x, end_y), (255, 255, 255), 
                        thickness=random.randint(1, 3))
            else:  # letter
                letter = random.choice(letters)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = random.uniform(0.5, 1.0)
                thickness = random.randint(1, 2)
                cv2.putText(noise_layer, letter, (art_x, art_y), font, font_scale, (255, 255, 255), thickness)
        
        # Add horizontal lines that might intersect with digits
        num_lines = random.randint(1, 3)
        for _ in range(num_lines):
            # Decide line type: touching, slightly intersecting, or below
            line_type = random.choice(['touching', 'intersecting', 'below'])
            
            # Calculate line position
            if line_type == 'touching':
                y_pos = int(max_y)
            elif line_type == 'intersecting':
                y_pos = int(max_y - random.randint(3, 8))
            else:  # below
                y_pos = int(max_y + random.randint(5, 15))
            
            # Calculate line length and position
            line_start_x = max(0, min_x - random.randint(10, 30))
            line_end_x = min(image.shape[1], max_x + random.randint(10, 30))
            
            # Add some waviness to the line
            num_points = random.randint(3, 6)
            points = []
            for i in range(num_points):
                x = line_start_x + (line_end_x - line_start_x) * i / (num_points - 1)
                y = y_pos + random.randint(-2, 2)
                points.append((int(x), int(y)))
            
            # Draw wavy line
            for i in range(len(points) - 1):
                cv2.line(noise_layer, points[i], points[i + 1], (255, 255, 255), 
                        thickness=random.randint(2, 4))
    
    # Combine the noise layer with the original image (full intensity)
    return cv2.add(image, noise_layer)

def apply_otsu_binarization(image):
    """Apply Otsu's binarization to the image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def is_position_valid(current_box, placed_boxes, max_iou=0.1):
    """Check if the current box position is valid relative to all placed boxes"""
    if not placed_boxes:
        return True
    
    # Check IoU with all placed boxes
    for box in placed_boxes:
        iou = calculate_iou(current_box, box)
        if iou > max_iou:
            return False
        
        # Also check for exact or near-exact coordinate matches
        curr_center_x = (current_box[0] + current_box[2]) / 2
        box_center_x = (box[0] + box[2]) / 2
        if abs(curr_center_x - box_center_x) < 1:  # 1 pixel threshold
            return False
    
    return True

def generate_multi_digit_image(digits, labels, target_size=416, min_digits=4, max_digits=8):
    """Generate image with multiple digits with specified constraints"""
    image = np.zeros((target_size, target_size), dtype=np.uint8)
    boxes = []  # [x1, y1, x2, y2, class_id]
    
    # Increase digit size by 10%
    base_height = int(target_size * 0.12 * 1.1)  # 12% of image height + 10%
    
    # Choose random number of digits (4-8)
    num_digits = random.randint(min_digits, max_digits)
    
    # Choose layout (center, left, right)
    layout = random.choice(['center', 'left', 'right'])
    
    # Calculate total width needed
    digit_info = []
    total_width = 0
    for _ in range(num_digits):
        idx = random.randint(0, len(digits) - 1)
        digit = digits[idx]
        aspect_ratio = digit.shape[1] / digit.shape[0]
        width = int(base_height * aspect_ratio)
        digit_info.append((width, base_height, idx))
        total_width += width
    
    # Add spacing between digits
    spacing = int(base_height * 0.2)  # 20% of height for spacing
    total_width += spacing * (num_digits - 1)
    
    # Calculate starting position based on layout
    margin = int(target_size * 0.1)  # 10% margin
    if layout == 'center':
        start_x = (target_size - total_width) // 2
    elif layout == 'left':
        start_x = margin
    else:  # right
        start_x = target_size - total_width - margin
    
    # Base Y position - keep all digits at roughly same height
    base_y = (target_size - base_height) // 2
    max_y_variation = 5  # Reduced vertical variation
    
    current_x = start_x
    placed_boxes = []  # Keep track of placed boxes for overlap checking
    
    for i, (width, height, idx) in enumerate(digit_info):
        # Add very small random vertical variation
        y_offset = random.randint(-max_y_variation, max_y_variation)
        y = base_y + y_offset
        
        # Ensure coordinates are within image bounds
        current_x = max(0, min(target_size - width, current_x))
        y = max(0, min(target_size - height, y))
        
        # Calculate current box
        current_box = [current_x, y, current_x + width, y + height]
        
        # Check if we should create overlap with previous digit (50% chance)
        should_overlap = random.random() < 0.5 and i > 0
        
        if should_overlap:
            # Try different overlap amounts until we find a valid position
            for overlap_percent in np.arange(0.03, 0.25, 0.03):  # 3% to 24% overlap
                prev_width = digit_info[i-1][0]
                overlap_amount = int(prev_width * overlap_percent)
                test_x = current_x - overlap_amount
                test_box = [test_x, y, test_x + width, y + height]
                
                if is_position_valid(test_box, placed_boxes):
                    current_x = test_x
                    current_box = test_box
                    break
            else:
                # If no valid overlap found, keep original position
                should_overlap = False
        
        # Verify final position is valid
        if not is_position_valid(current_box, placed_boxes):
            # If position is invalid, try to adjust slightly
            for offset in range(1, 6):  # Try up to 5 pixels right
                test_box = [current_x + offset, y, current_x + offset + width, y + height]
                if is_position_valid(test_box, placed_boxes):
                    current_x += offset
                    current_box = test_box
                    break
            else:
                # If still invalid, skip this digit
                continue
        
        digit = digits[idx]
        label = labels[idx]
        
        # Resize digit
        digit_resized = cv2.resize(digit, (width, height))
        
        try:
            # Get the region where we'll place the digit
            y_end = min(y + height, target_size)
            x_end = min(current_x + width, target_size)
            region = image[y:y_end, current_x:x_end]
            
            # Ensure digit_resized is cropped to match region size
            digit_crop = digit_resized[:y_end-y, :x_end-current_x]
            mask = digit_crop > 0
            
            if should_overlap:
                # Only modify non-zero pixels in overlap region
                overlap_width = overlap_amount
                overlap_region = region[:, :overlap_width]
                overlap_mask = mask[:, :overlap_width]
                overlap_digit = digit_crop[:, :overlap_width]
                
                # Keep maximum value for overlapping pixels
                overlap_region[overlap_mask] = np.maximum(
                    overlap_region[overlap_mask],
                    overlap_digit[overlap_mask]
                )
                
                # Place non-overlapping part normally
                non_overlap_region = region[:, overlap_width:]
                non_overlap_mask = mask[:, overlap_width:]
                non_overlap_digit = digit_crop[:, overlap_width:]
                non_overlap_region[non_overlap_mask] = non_overlap_digit[non_overlap_mask]
            else:
                region[mask] = digit_crop[mask]
            
            # Add box with class
            boxes.append([current_x, y, x_end, y_end, label])
            placed_boxes.append([current_x, y, x_end, y_end])
            
        except ValueError:
            continue
        
        # Move to next position
        current_x += width + spacing
    
    if not boxes:  # If no digits were placed successfully
        return generate_multi_digit_image(digits, labels, target_size, min_digits, max_digits)
    
    # Add noise and lines
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = add_noise_and_lines(image, boxes)
    
    # Apply Otsu's binarization
    image = apply_otsu_binarization(image)
    
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
    output_dir = Path("dataset_yolo_v3")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    for dir_path in [images_dir, labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print("\nMNIST Dataset sizes:")
    print(f"Training set: {len(x_train)} images")
    print(f"Test set: {len(x_test)} images")
    print(f"Total: {len(x_train) + len(x_test)} images\n")
    
    # Combine train and test data
    digits = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])
    
    # Generate dataset - увеличиваем количество изображений
    num_images = 200  # Генерируем 500,000 изображений
    print(f"Generating {num_images} images...")
    
    for i in tqdm(range(num_images)):
        image, boxes = generate_multi_digit_image(digits, labels)
        if not boxes:  # If no digits were placed successfully
            continue  # Skip this image and try again
        
        # Convert boxes to YOLO format
        yolo_boxes = convert_to_yolo_format(boxes, 416, 416)
        
        # Save image
        cv2.imwrite(str(images_dir / f"image_{i:06d}.jpg"), image)
        
        # Save labels
        with open(labels_dir / f"image_{i:06d}.txt", 'w') as f:
            for box in yolo_boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

if __name__ == "__main__":
    main() 