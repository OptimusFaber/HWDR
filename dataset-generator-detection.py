import os
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import warnings
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# Disable all warnings
warnings.filterwarnings('ignore')

def create_paper_texture(size, noise_intensity=0.1):
    """Creates paper texture with various shades of white"""
    # Create base light background (slightly creamy tint)
    base_color = np.random.uniform(0.92, 0.98)  # Base white shade
    texture = np.ones((*size, 3), dtype=np.float32) * base_color
    
    # Add slight yellow/cream tint
    yellow_tint = np.random.uniform(0, 0.02)  # Very light yellow tint
    texture[:, :, 0] *= (1.0 - yellow_tint)  # Slightly reduce blue channel
    texture[:, :, 1] *= (1.0 - yellow_tint * 0.5)  # Reduce green channel a bit less
    
    # Add paper texture (fine irregularities)
    fine_noise = np.random.randn(*size) * noise_intensity * 0.5
    fine_noise = gaussian_filter(fine_noise, sigma=0.5)
    fine_noise = fine_noise[:, :, np.newaxis]
    
    # Add larger irregularities
    coarse_noise = np.random.randn(*size) * noise_intensity
    coarse_noise = gaussian_filter(coarse_noise, sigma=2.0)
    coarse_noise = coarse_noise[:, :, np.newaxis]
    
    # Combine noises
    combined_noise = (fine_noise + coarse_noise) * 0.5
    texture += np.repeat(combined_noise, 3, axis=2)
    
    # Add light vertical or horizontal stripes (paper structure imitation)
    if random.random() < 0.5:  # 50% chance for vertical stripes
        stripe_noise = np.random.randn(1, size[1]) * noise_intensity * 0.3
        stripe_noise = np.repeat(stripe_noise, size[0], axis=0)
    else:  # horizontal stripes
        stripe_noise = np.random.randn(size[0], 1) * noise_intensity * 0.3
        stripe_noise = np.repeat(stripe_noise, size[1], axis=1)
    stripe_noise = gaussian_filter(stripe_noise, sigma=3.0)
    texture += np.dstack([stripe_noise, stripe_noise, stripe_noise])
    
    # Normalize values and convert to uint8
    texture = np.clip(texture * 255, 192, 255).astype(np.uint8)
    
    return texture

def apply_pen_effect(digit_image, pressure_variation=True):
    """Applies pen effect to the digit image"""
    # Normalize image (now digits will be 1, and background 0)
    digit = (digit_image < 128).astype(np.float32)
    
    # Create base blue color (BGR)
    blue_color = np.array([230, 30, 10]) / 255.0  # Saturated blue color (BGR)
    
    # Add very small color variations for realism
    color_variation = np.random.randn(3) * 0.02
    color_variation[0] *= 0.5  # Less variation for blue channel
    blue_color = np.clip(blue_color + color_variation, 0, 1)
    
    # Create pressure map
    if pressure_variation:
        # Create base pressure map with multiple layers
        pressure_base = np.random.randn(*digit.shape) * 0.1 + 0.9
        pressure_detail = np.random.randn(*digit.shape) * 0.05
        
        # Apply different blur for each layer
        pressure_base = gaussian_filter(pressure_base, sigma=2.0)
        pressure_detail = gaussian_filter(pressure_detail, sigma=0.5)
        
        # Combine layers
        pressure_map = pressure_base + pressure_detail
        pressure_map = np.clip(pressure_map, 0.75, 1.0)
        
        # Add random ink "drops"
        drops = np.random.rand(*digit.shape) < 0.01
        drops = gaussian_filter(drops.astype(float), sigma=0.5)
        pressure_map += drops * 0.1
        
        # Apply pressure only where digit exists
        digit = digit * pressure_map
    
    # Add stroke texture (multiple layers)
    # Coarse texture
    coarse_texture = np.random.randn(*digit.shape) * 0.08
    coarse_texture = gaussian_filter(coarse_texture, sigma=1.0)
    
    # Fine texture
    fine_texture = np.random.randn(*digit.shape) * 0.04
    fine_texture = gaussian_filter(fine_texture, sigma=0.3)
    
    # Combine textures
    stroke_texture = coarse_texture + fine_texture
    
    # Add stroke directionality (vertical or horizontal stripes)
    if random.random() < 0.5:
        # Vertical strokes
        vertical_texture = np.random.randn(1, digit.shape[1]) * 0.05
        vertical_texture = np.repeat(vertical_texture, digit.shape[0], axis=0)
    else:
        # Horizontal strokes
        horizontal_texture = np.random.randn(digit.shape[0], 1) * 0.05
        horizontal_texture = np.repeat(horizontal_texture, digit.shape[1], axis=1)
        stroke_texture += horizontal_texture
    
    # Apply texture only to digit
    digit += (digit > 0.1) * stroke_texture
    digit = np.clip(digit, 0, 1)
    
    # Create colored image
    result = np.ones((*digit.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - digit * (1.0 - blue_color[i])
    
    # Add slight blur to imitate ink spreading
    result = cv2.GaussianBlur(result, (3, 3), 0.3)
    
    # Add random small dots (ink splatter)
    if random.random() < 0.3:  # 30% chance
        splatter = np.random.rand(*digit.shape) < 0.001
        splatter = gaussian_filter(splatter.astype(float), sigma=0.5)
        splatter = np.dstack([splatter] * 3)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * (1 - splatter[:, :, i]) + blue_color[i] * splatter[:, :, i]
    
    # Convert to uint8
    result = (result * 255).astype(np.uint8)
    
    return result

class TouchingDigitsLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.numbers = []
        self.labels = []
        self.usage_count = defaultdict(int)
        
        # Load all numbers from dataset
        print("Loading numbers from SyntheticDigitStrings...")
        number_dirs = list(self.dataset_path.iterdir())
        for number_dir in tqdm(number_dirs, desc="Loading directories"):
            if not number_dir.is_dir():
                continue
                
            number = number_dir.name
            txt_files = list(number_dir.glob('*.txt'))
            if not txt_files:
                continue
                
            # Take only one file from each directory to speed up
            txt_file = random.choice(txt_files)
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:  # Skip files without markup
                    continue
                
                # Get image
                img_path = txt_file.with_suffix('.png')
                if not img_path.exists():
                    continue
                    
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                
                # Convert image to grayscale
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA
                        alpha = img[:, :, 3]
                        digit = np.ones_like(alpha) * 255
                        digit[alpha > 0] = 0
                    else:  # RGB
                        digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:  # Already grayscale
                    digit = img
                
                # Parse markup
                boxes = []
                for line in lines[2:]:  # Skip first two lines
                    parts = line.strip().split('|')
                    if len(parts) != 4:
                        continue
                    class_id = int(parts[0])
                    coords = parts[1].split(',') + parts[2].split(',')
                    boxes.append([int(x) for x in coords] + [class_id])
                
                self.numbers.append((digit, boxes, number))
                self.labels.append(number)
        
        if not self.numbers:
            raise ValueError(f"No numbers found in {dataset_path}")
        
        print(f"Loaded {len(self.numbers)} numbers from SyntheticDigitStrings")
    
    def get_number(self):
        """Get a random number from dataset"""
        if not self.numbers:
            return None
            
        # Choose number that was used least
        idx = min(range(len(self.numbers)), key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        
        # Reset counters if all numbers were used too many times
        if min(self.usage_count.values()) > 20:
            self.usage_count.clear()
            
        return self.numbers[idx]

class LetterNoiseLoader:
    def __init__(self, letters_path):
        """Loads letters to use as noise"""
        self.letters_path = Path(letters_path)
        self.letters = []
        self.letter_classes = []
        
        # Create list of valid folders (letters a-z and A-Z)
        valid_folders = []
        for folder in self.letters_path.iterdir():
            if folder.is_dir() and len(folder.name) == 1:
                if folder.name.isalpha():  # Only letters
                    valid_folders.append(folder.name)
        
        print(f"Found {len(valid_folders)} letter folders: {sorted(valid_folders)}")
        
        # Load letter images
        for folder_name in valid_folders:
            folder_path = self.letters_path / folder_name
            for img_file in folder_path.glob("*.png"):
                try:
                    # Load image with alpha channel
                    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # If image has alpha channel (4 channels)
                        if img.shape[-1] == 4:
                            # Take only alpha channel
                            alpha = img[:, :, 3]
                            # Create white image
                            letter = np.ones_like(alpha) * 255
                            # Where alpha > 0, set to 0 (black)
                            letter[alpha > 0] = 0
                        else:
                            # If no alpha channel, use as is
                            letter = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        self.letters.append(letter)
                        self.letter_classes.append(folder_name)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        print(f"Loaded {len(self.letters)} letters for noise")
    
    def get_random_letter(self):
        """Returns a random letter"""
        if not self.letters:
            return None, None
        idx = random.randint(0, len(self.letters) - 1)
        return self.letters[idx], self.letter_classes[idx]

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

def is_valid_overlap(box1, box2, max_iou=0.15):
    """Check if the overlap between two boxes is valid"""
    iou = calculate_iou(box1, box2)
    return iou < max_iou

def check_alignment(boxes, tolerance=5):
    """Checks if all digits are on the same line"""
    if not boxes:
        return True
    # Take the middle line of each bounding box
    mid_lines = [(box[1] + box[3]) / 2 for box in boxes]
    # Check if all middle lines are within tolerance of the first one
    reference = mid_lines[0]
    return all(abs(mid - reference) <= tolerance for mid in mid_lines)

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
                
                # Draw line in blue color with variations
                line_color = (
                    random.randint(240, 255),  # B
                    random.randint(20, 40),    # G
                    random.randint(0, 10)      # R
                )
                thickness = random.randint(1, 2)  # Reduce line thickness
                for i in range(len(points) - 1):
                    # Draw main line
                    cv2.line(noise_layer, points[i], points[i + 1], line_color, thickness)
                    # Add a thinner line on top for unevenness effect
                    if thickness > 1:
                        darker_color = (
                            min(255, line_color[0] + 20),
                            max(0, line_color[1] - 10),
                            max(0, line_color[2] - 10)
                        )
                        cv2.line(noise_layer, points[i], points[i + 1], darker_color, 1)
    
    # Add lines
    mask = np.any(noise_layer > 0, axis=2).astype(np.float32)
    mask = mask[:, :, np.newaxis]
    mask = np.repeat(mask, 3, axis=2)
    # Slightly reduce line intensity
    noise_layer = noise_layer * 0.9
    image = image * (1 - mask) + noise_layer * mask
    
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
                        # If image has alpha channel (4 channels)
                        if img.shape[-1] == 4:
                            # Take only alpha channel
                            alpha = img[:, :, 3]
                            # Create white image
                            digit = np.ones_like(alpha) * 255
                            # Where alpha > 0, set to 0 (black)
                            digit[alpha > 0] = 0
                        else:
                            # If no alpha channel, use as is
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
            raise ValueError("No available digits found. This should not happen if the dataset was loaded correctly.")
        
        idx = min(available_indices, key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        return self.digits[idx], self.labels[idx]

def add_letter_noise(image, letter_loader, boxes, num_letters=10, probability=0.9):
    """Adds letter noise to the image of the same size as digits"""
    if random.random() > probability or not letter_loader:
        return image
    
    # Determine number of letters to add
    num_to_add = random.randint(num_letters, num_letters + 10)  # 10-20 letters
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Determine letter size based on digit size
    if boxes:
        # Calculate average digit size
        digit_heights = []
        for box in boxes:
            digit_height = box[3] - box[1]  # y2 - y1
            digit_heights.append(digit_height)
        
        if digit_heights:
            avg_digit_height = sum(digit_heights) / len(digit_heights)
            letter_height = int(avg_digit_height * 1.2)  # Letters are 20% larger than digits
        else:
            letter_height = int(h * 0.12)  # Fallback size
    else:
        letter_height = int(h * 0.12)  # Fallback size
    
    # Determine several possible horizontal lines for letters
    possible_lines = []
    if boxes:
        min_y = min(box[1] for box in boxes)
        max_y = max(box[3] for box in boxes)
        h_step = int(letter_height * 1.1)
        # Line above digits
        if min_y - h_step >= 0:
            possible_lines.append(max(0, min_y - h_step))
        # Line with digits (main)
        possible_lines.append(min_y)
        # Line below digits
        if max_y + h_step < h:
            possible_lines.append(min(h - letter_height, max_y + h_step))
        # If there's plenty of space, add another line above and below
        if min_y - 2 * h_step >= 0:
            possible_lines.append(max(0, min_y - 2 * h_step))
        if max_y + 2 * h_step < h:
            possible_lines.append(min(h - letter_height, max_y + 2 * h_step))
    else:
        # If no digits, place in the center
        possible_lines = [int((h - letter_height) // 2)]
    
    # Create list of already occupied areas to avoid overlaps
    occupied_areas = []
    added_count = 0
    
    for _ in range(num_to_add):
        # Get a random letter
        letter_img, letter_class = letter_loader.get_random_letter()
        if letter_img is None:
            continue
        
        # Apply pen effect to the letter
        colored_letter = apply_pen_effect(letter_img)
        
        # Calculate letter width while maintaining proportions
        aspect_ratio = letter_img.shape[1] / letter_img.shape[0]
        width = int(letter_height * aspect_ratio)
        
        # Resize the letter
        resized_letter = cv2.resize(colored_letter, (width, letter_height))
        
        # Place the letter in the selected area
        max_attempts = 50
        for attempt in range(max_attempts):
            # Random position within the selected range
            x = random.randint(0, w - width)
            # Select a random line for letter placement
            letter_y = random.choice(possible_lines)
            
            # Check boundaries
            if x < 0 or x + width > w or letter_y < 0 or letter_y + letter_height > h:
                continue
            
            # Check for overlap with digits
            letter_area = [x, letter_y, x + width, letter_y + letter_height]
            overlap_with_digits = False
            
            for box in boxes:
                if not (letter_area[2] < box[0] or letter_area[0] > box[2] or 
                       letter_area[3] < box[1] or letter_area[1] > box[3]):
                    overlap_with_digits = True
                    break
            
            if overlap_with_digits:
                continue
            
            # Check for overlap with already placed letters
            overlap_with_letters = False
            for area in occupied_areas:
                if not (letter_area[2] < area[0] or letter_area[0] > area[2] or 
                       letter_area[3] < area[1] or letter_area[1] > area[3]):
                    overlap_with_letters = True
                    break
            
            if not overlap_with_letters:
                # Place the letter
                roi = image[letter_y:letter_y + letter_height, x:x + width]
                # Create mask based on brightness
                gray_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
                mask = (gray_letter < 240).astype(np.float32)
                mask = np.dstack([mask, mask, mask])
                # Blend with background
                roi[:] = roi * (1 - mask) + resized_letter * mask
                
                # Add area to occupied list
                occupied_areas.append(letter_area)
                added_count += 1
                break
    
    return image

def apply_high_quality_resize(digit_image, target_height, max_rotation=5):
    """
    Applies high-quality scaling to the digit image with random distortions
    """
    # Determine dimensions while maintaining proportions
    aspect_ratio = digit_image.shape[1] / digit_image.shape[0]
    target_width = int(target_height * aspect_ratio)
    
    # Add padding for rotation
    padding = int(target_height * 0.2)
    padded_height = target_height + 2 * padding
    padded_width = target_width + 2 * padding
    
    # Select interpolation method based on scaling type
    scale_factor = target_height / digit_image.shape[0]
    if scale_factor > 1:
        interpolation = cv2.INTER_CUBIC  # Better for enlargement
    else:
        interpolation = cv2.INTER_AREA   # Better for reduction
    
    # Scale image
    resized = cv2.resize(digit_image, (target_width, target_height), interpolation=interpolation)
    
    # Create image with padding
    padded = np.ones((padded_height, padded_width), dtype=np.uint8) * 255
    padded[padding:padding+target_height, padding:padding+target_width] = resized
    
    # Apply random rotation
    angle = np.random.uniform(-max_rotation, max_rotation)
    rotation_matrix = cv2.getRotationMatrix2D(
        (padded_width/2, padded_height/2), angle, 1.0)
    rotated = cv2.warpAffine(padded, rotation_matrix, (padded_width, padded_height),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    
    # Add slight perspective distortion
    if random.random() < 0.3:  # 30% chance
        height, width = rotated.shape
        # Create points in the correct format for OpenCV
        src_points = np.array([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ], dtype=np.float32)
        
        # Small random shifts of corners
        max_shift = width * 0.05
        dst_points = src_points + np.random.uniform(-max_shift, max_shift, src_points.shape).astype(np.float32)
        
        # Apply perspective transformation
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(rotated, transform_matrix, (width, height),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    else:
        warped = rotated
    
    # Crop padding, finding the boundaries of the digit
    coords = cv2.findNonZero(255 - warped)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = warped[y:y+h, x:x+w]
        
        # Scale to target height
        final_width = int(target_height * (cropped.shape[1] / cropped.shape[0]))
        final = cv2.resize(cropped, (final_width, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        final = cv2.resize(warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply slight blur to remove artifacts
    final = cv2.GaussianBlur(final, (3, 3), 0.5)
    
    return final

def generate_multi_digit_image(digit_selector, touching_loader, target_size=416, min_digits=5, max_digits=10, max_attempts=50, is_dark_bg=False):
    """Generates an image with digits, possibly including numbers from the touching digits dataset"""
    for attempt in range(max_attempts):
        # Create background as paper texture
        image = create_paper_texture((target_size, target_size))
        
        boxes = []  # [x1, y1, x2, y2, class_id]
        base_height = int(target_size * 0.12)
        
        # First, try to place touching digits (if available)
        if touching_loader is not None:
            touching_number = touching_loader.get_number()
            if touching_number:
                digit_img, digit_boxes, number = touching_number
                # Scale image by height with high quality
                resized_digit = apply_high_quality_resize(digit_img, base_height)
                new_width = resized_digit.shape[1]
                
                # Calculate scale for bounding boxes
                scale_x = new_width / digit_img.shape[1]
                scale_y = base_height / digit_img.shape[0]
                
                # Apply pen effect
                colored_digit = apply_pen_effect(resized_digit)
                
                # Place in the center
                start_x = (target_size - new_width) // 2
                start_y = (target_size - base_height) // 2
                
                # Place image
                roi = image[start_y:start_y + base_height, start_x:start_x + new_width]
                # Create mask for digits
                gray_digit = cv2.cvtColor(colored_digit, cv2.COLOR_BGR2GRAY)
                mask = (gray_digit < 240 if not is_dark_bg else gray_digit > 15).astype(np.float32)
                mask = np.dstack([mask, mask, mask])
                # Blend with background
                roi[:] = roi * (1 - mask) + colored_digit * mask
                
                # Update bounding box coordinates
                main_boxes = []
                for box in digit_boxes:
                    x1, y1, x2, y2, class_id = box
                    # Scale coordinates
                    x1 = int(x1 * scale_x) + start_x
                    y1 = int(y1 * scale_y) + start_y
                    x2 = int(x2 * scale_x) + start_x
                    y2 = int(y2 * scale_y) + start_y
                    main_boxes.append([x1, y1, x2, y2, class_id])
                boxes.extend(main_boxes)
                
                # If there's space, add single digits on the sides
                if digit_selector is not None:
                    # Determine available areas on the left and right of the main number
                    main_left = min(box[0] for box in boxes)
                    main_right = max(box[2] for box in boxes)
                    main_y = boxes[0][1]  # Use the same height
                    
                    # Try to add digits on the left
                    left_space = main_left
                    if left_space > base_height:  # If there's at least space for one digit
                        num_left = random.randint(1, min(3, left_space // base_height))
                        current_x = 0
                        for _ in range(num_left):
                            if current_x + base_height > main_left:
                                break
                            digit, label = digit_selector.get_digit()
                            # Use improved scaling
                            resized_digit = apply_high_quality_resize(digit, base_height)
                            width = resized_digit.shape[1]
                            if current_x + width > main_left - 10:  # Leave padding
                                break
                            colored_digit = apply_pen_effect(resized_digit)
                            
                            # Place the digit
                            roi = image[main_y:main_y + base_height, current_x:current_x + width]
                            gray_digit = cv2.cvtColor(colored_digit, cv2.COLOR_BGR2GRAY)
                            mask = (gray_digit < 240 if not is_dark_bg else gray_digit > 15).astype(np.float32)
                            mask = np.dstack([mask, mask, mask])
                            roi[:] = roi * (1 - mask) + colored_digit * mask
                            
                            # Add bounding box
                            box = [current_x, main_y, current_x + width, main_y + base_height, label]
                            boxes.append(box)
                            current_x += width + random.randint(5, 15)  # Add random padding
                    
                    # Try to add digits on the right
                    right_space = target_size - main_right
                    if right_space > base_height:
                        num_right = random.randint(1, min(3, right_space // base_height))
                        current_x = main_right + 10  # Start with padding
                        for _ in range(num_right):
                            digit, label = digit_selector.get_digit()
                            # Use improved scaling
                            resized_digit = apply_high_quality_resize(digit, base_height)
                            width = resized_digit.shape[1]
                            if current_x + width >= target_size:
                                break
                            
                            colored_digit = apply_pen_effect(resized_digit)
                            
                            # Place the digit
                            roi = image[main_y:main_y + base_height, current_x:current_x + width]
                            gray_digit = cv2.cvtColor(colored_digit, cv2.COLOR_BGR2GRAY)
                            mask = (gray_digit < 240 if not is_dark_bg else gray_digit > 15).astype(np.float32)
                            mask = np.dstack([mask, mask, mask])
                            roi[:] = roi * (1 - mask) + colored_digit * mask
                            
                            # Add bounding box
                            box = [current_x, main_y, current_x + width, main_y + base_height, label]
                            boxes.append(box)
                            current_x += width + random.randint(5, 15)  # Add random padding
        
        # If no touching digits or failed to place them, generate only from single digits
        if not boxes and digit_selector is not None:
            num_digits = random.randint(min_digits, max_digits)
            
            # First, collect information about all digits
            digit_info = []
            total_width = 0
            for _ in range(num_digits):
                digit, label = digit_selector.get_digit()
                aspect_ratio = digit.shape[1] / digit.shape[0]
                width = int(base_height * aspect_ratio)
                # Resize the digit
                resized_digit = cv2.resize(digit, (width, base_height))
                # Apply pen effect
                colored_digit = apply_pen_effect(resized_digit)
                digit_info.append((width, base_height, colored_digit, label))
                total_width += width
            
            # Base spacing between digits
            spacing = int(base_height * 0.1)
            total_width += spacing * (num_digits - 1)
            
            # If digits don't fit, try the next attempt
            if total_width > target_size * 0.9:
                continue
            
            # Center horizontally and vertically
            start_x = (target_size - total_width) // 2
            start_y = (target_size - base_height) // 2
            
            # Place digits
            current_x = start_x
            for width, height, colored_digit, label in digit_info:
                # Place the digit
                roi = image[start_y:start_y + height, current_x:current_x + width]
                # Create mask based on brightness
                gray_digit = cv2.cvtColor(colored_digit, cv2.COLOR_BGR2GRAY)
                mask = (gray_digit < 240 if not is_dark_bg else gray_digit > 15).astype(np.float32)
                mask = np.dstack([mask, mask, mask])
                # Blend with background
                roi[:] = roi * (1 - mask) + colored_digit * mask
                
                # Add bounding box
                box = [current_x, start_y, current_x + width, start_y + height, label]
                boxes.append(box)
                
                # Update position for the next digit
                current_x += width + spacing
        
        # Check number of digits
        if len(boxes) >= min_digits:
            print(f"Successfully placed {len(boxes)} digits")
            return image, boxes
            
    print("Failed to place all digits correctly after multiple attempts")
    return image, boxes

def main():
    # ============================================================================
    # DATASET GENERATION CONFIGURATION
    # ============================================================================
    
    # a) Use Touching Digits dataset
    USE_TOUCHING_DIGITS = True
    TOUCHING_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Touching-Digits/SyntheticDigitStrings"
    
    # b) Use digits for this dataset (Single Digits)
    USE_SINGLE_DIGITS = True
    SINGLE_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Single-Digits/dataset"
    
    # c) Add markup for dataset with letters (systemically without markup)
    ADD_LETTER_ANNOTATIONS = False  # Letters are added as noise without markup
    
    # d) Pen effect parameters
    PEN_PRESSURE_VARIATION = True  # Enable pen pressure variation
    PAPER_TEXTURE_INTENSITY = 0.08  # Paper texture intensity (reduced)
    
    # ============================================================================
    # GENERATION PARAMETERS
    # ============================================================================
    num_images = 20000  # Number of images to generate
    target_size = 416  # Image size
    min_digits = 5    # Minimum number of digits
    max_digits = 10   # Maximum number of digits
    
    # ============================================================================
    
    # Create output directories
    output_dir = Path("/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/yolo-dataset-v9.2")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize digit selector
    digit_selector = None
    if USE_SINGLE_DIGITS:
        try:
            digit_selector = DigitSelector(SINGLE_DIGITS_PATH)
            print(f"Single digits loader initialized: {SINGLE_DIGITS_PATH}")
        except Exception as e:
            print(f"Error initializing single digits loader: {e}")
            USE_SINGLE_DIGITS = False
    
    # Initialize touching digits loader
    touching_loader = None
    if USE_TOUCHING_DIGITS:
        try:
            touching_loader = TouchingDigitsLoader(TOUCHING_DIGITS_PATH)
            print(f"Touching digits loader initialized: {TOUCHING_DIGITS_PATH}")
        except Exception as e:
            print(f"Error initializing touching digits loader: {e}")
            USE_TOUCHING_DIGITS = False
    
    # Initialize letter noise loader
    letters_path = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Hnd-Letters/Img"
    try:
        letter_loader = LetterNoiseLoader(letters_path)
        print("Letter noise loader initialized successfully")
    except Exception as e:
        print(f"Error initializing letter loader: {e}")
        letter_loader = None
    
    # Check that at least one digit loader is working
    if not USE_SINGLE_DIGITS and not USE_TOUCHING_DIGITS:
        raise ValueError("At least one digit source must be enabled (USE_SINGLE_DIGITS or USE_TOUCHING_DIGITS)")
    
    # Generate images
    for i in tqdm(range(num_images)):
        image, boxes = generate_multi_digit_image(digit_selector, touching_loader, target_size, min_digits, max_digits)
        
        # Add noise and lines to image
        image = add_noise_and_lines(image, boxes)
        
        # Add letter noise
        image = add_letter_noise(image, letter_loader, boxes)
        
        # Save image
        image_path = images_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save labels in YOLO format (only for digits, not for letters)
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                x_min, y_min, x_max, y_max, class_id = box
                yolo_box = convert_to_yolo_format([x_min, y_min, x_max, y_max], 
                                                image.shape[1], image.shape[0])
                f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

if __name__ == "__main__":
    main() 