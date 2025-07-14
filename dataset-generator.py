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
    # Create base light background (slightly cream tint)
    base_color = np.random.uniform(0.92, 0.98)  # Base white shade
    texture = np.ones((*size, 3), dtype=np.float32) * base_color
    
    # Add slight yellow/cream tint
    yellow_tint = np.random.uniform(0, 0.02)  # Very light yellow tint
    texture[:, :, 0] *= (1.0 - yellow_tint)  # Slightly decrease blue channel
    texture[:, :, 1] *= (1.0 - yellow_tint * 0.5)  # Decrease green channel a bit less
    
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
    
    # Add light vertical or horizontal stripes (imitating paper structure)
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

def generate_random_color():
    """Generates a random color for the pen in BGR format"""
    # Create different shades by varying all components
    variations = [
        lambda: np.array([180, 20, 5]),     # Blue
        lambda: np.array([107, 46, 9]),     # Dark Blue #1
        lambda: np.array([82, 33, 4]),      # Dark Blue #2
        lambda: np.array([33, 13, 0]),      # Black
        # lambda: np.array([255, 30, 10]),    # Bright Blue
        # lambda: np.array([255, 140, 50]),   # Cyan
        lambda: np.array([200, 30, 50]),     # Blue with purple
        # lambda: np.array([220, 120, 30]),    # Marine Blue
        lambda: np.array([230, 50, 20]),     # Cobalt Blue
        lambda: np.array([190, 40, 60]),     # Indigo
        # lambda: np.array([210, 90, 40]),     # Sapphire
        lambda: np.array([250, 10, 5]),      # Ultramarine
        lambda: np.array([147, 62, 66]),     # Blue dust
        lambda: np.array([177, 102, 115]),   # Crimson
        lambda: np.array([148, 64, 67]),     # Deep Purple
    ]
    
    # Select base color
    base_color = variations[np.random.randint(0, len(variations))]()
    
    # Add a small random variation for variety
    noise = np.random.uniform(-20, 20, 3)
    color = np.clip(base_color + noise, 0, 255)
    
    return color / 255.0

def apply_pen_effect_classic_blue(digit_image, color=None):
    """Classic blue pen effect - thin precise lines"""
    if color is None:
        color = np.array([200, 20, 0]) / 255.0  # Default color
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    # Reduce line thickness using erosion
    kernel = np.ones((2, 2), np.uint8)
    digit_eroded = cv2.erode(digit_image, kernel, iterations=1)
    # digit_eroded = digit_image
    
    digit_mask = (digit_eroded == 0).astype(np.float32)
    
    # Make lines thinner
    digit_mask = gaussian_filter(digit_mask, sigma=0.2)
    digit_mask = np.power(digit_mask, 2.0)
    
    # Minimum texture for classic pen
    stroke_texture = gaussian_filter(np.random.randn(*digit_mask.shape) * 0.01, sigma=0.2)
    digit_mask += (digit_mask > 0.1) * stroke_texture
    
    # Create edge gradient mask with multiple blur levels
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.5)
    edge_mask2 = gaussian_filter(digit_mask, sigma=1.0)
    edge_mask3 = gaussian_filter(digit_mask, sigma=2.0)
    
    # Combine masks to create a gradient transition
    edge_gradient = (edge_mask1 - edge_mask2) * 0.6 + (edge_mask2 - edge_mask3) * 0.4
    edge_gradient = np.clip(edge_gradient * 2.0, 0, 1)  # Strengthen effect
    
    # Apply gradient transparency
    alpha = digit_mask * (1.0 - edge_gradient)
    alpha = np.clip(alpha, 0, 1)
    
    # Further enhance contrast in the center of lines
    alpha = np.where(alpha > 0.5, alpha * 1.2, alpha * 0.8)
    alpha = np.clip(alpha, 0, 1)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    return (result * 255).astype(np.uint8)

def apply_pen_effect_gel(digit_image, color=None):
    """Gel pen effect - smooth lines with strong shine"""
    if color is None:
        color = np.array([255, 20, 5]) / 255.0  # Default color
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    digit_mask = (digit_image == 0).astype(np.float32)
    
    # Make lines thinner, but with smooth edges
    digit_mask = gaussian_filter(digit_mask, sigma=0.3)
    digit_mask = np.power(digit_mask, 1.8)
    
    # Create edge gradient mask with multiple blur levels
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.3)
    edge_mask2 = gaussian_filter(digit_mask, sigma=0.8)
    edge_mask3 = gaussian_filter(digit_mask, sigma=1.5)
    
    # Combine masks to create a gradient transition
    edge_gradient = (edge_mask1 - edge_mask2) * 0.7 + (edge_mask2 - edge_mask3) * 0.3
    edge_gradient = np.clip(edge_gradient * 2.5, 0, 1)  # Strengthen effect
    
    # Apply gradient transparency while preserving brightness in the center
    alpha = digit_mask * (1.0 - edge_gradient * 0.8)
    alpha = np.clip(alpha, 0, 1)
    
    # Further enhance contrast in the center for a brighter effect
    alpha = np.where(alpha > 0.6, alpha * 1.3, alpha * 0.7)
    alpha = np.clip(alpha, 0, 1)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    # Add gloss effects considering the new transparency mask
    gloss_mask = (alpha > 0.7) * (np.random.rand(*digit_mask.shape) < 0.03)
    gloss = gaussian_filter(gloss_mask.astype(float), sigma=0.2) * 0.4
    result += np.dstack([gloss] * 3)
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def apply_pen_effect_fountain(digit_image, color=None):
    """Fountain pen effect - characteristic thickening and spreading"""
    if color is None:
        color = np.array([180, 30, 40]) / 255.0  # Default color
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    digit_mask = (digit_image == 0).astype(np.float32)
    
    # Base line with slight blur
    digit_mask = gaussian_filter(digit_mask, sigma=0.3)
    
    # Create uneven pressure effect
    pressure = np.random.randn(*digit_mask.shape) * 0.15 + 1.0
    pressure = gaussian_filter(pressure, sigma=1.5)
    digit_mask *= pressure
    
    # Add characteristic thickening in places where direction changes
    gradient_x = gaussian_filter(digit_mask, sigma=1.0, order=[1,0])
    gradient_y = gaussian_filter(digit_mask, sigma=1.0, order=[0,1])
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Strengthen effect in places with a large gradient
    thickening = gradient_mag * 0.5
    thickening = gaussian_filter(thickening, sigma=0.5)
    digit_mask = np.maximum(digit_mask, thickening)
    
    # Ink spreading effect
    ink_spread = gaussian_filter(digit_mask, sigma=0.7)
    digit_mask = np.maximum(digit_mask, ink_spread * 0.3)
    
    # Create edge gradient mask considering spreading
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.5)
    edge_mask2 = gaussian_filter(digit_mask, sigma=1.2)
    edge_mask3 = gaussian_filter(digit_mask, sigma=2.5)
    
    # Combine masks to create a gradient transition
    edge_gradient = (edge_mask1 - edge_mask2) * 0.5 + (edge_mask2 - edge_mask3) * 0.5
    edge_gradient = np.clip(edge_gradient * 3.0, 0, 1)  # Strengthen effect
    
    # Add random variations to transparency for uneven spreading simulation
    variation = gaussian_filter(np.random.randn(*digit_mask.shape) * 0.1, sigma=1.0)
    edge_gradient += variation * (edge_gradient > 0.2)
    edge_gradient = np.clip(edge_gradient, 0, 1)
    
    # Apply gradient transparency considering spreading
    alpha = digit_mask * (1.0 - edge_gradient)
    alpha = np.clip(alpha, 0, 1)
    
    # Further enhance contrast in the center for a more saturated effect
    alpha = np.where(alpha > 0.5, alpha * 1.4, alpha * 0.6)
    alpha = np.clip(alpha, 0, 1)
    
    # Final smoothing
    alpha = gaussian_filter(alpha, sigma=0.2)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    return (result * 255).astype(np.uint8)

def apply_pen_effect_ballpoint(digit_image, color=None):
    """Ballpoint pen effect - intermittent lines with characteristic gaps"""
    if color is None:
        color = np.array([210, 35, 15]) / 255.0  # Default color
    else:
        color = color / 255.0 if color.max() > 1 else color
        
    digit_mask = (digit_image == 0).astype(np.float32)
    
    # Base thin line
    digit_mask = gaussian_filter(digit_mask, sigma=0.2)
    digit_mask = np.power(digit_mask, 1.8)
    
    # Create uneven pressure effect
    pressure = np.random.randn(*digit_mask.shape) * 0.1 + 0.9
    pressure = gaussian_filter(pressure, sigma=1.0)
    digit_mask *= pressure
    
    # Add microtexture instead of stripes
    texture = np.random.randn(*digit_mask.shape) * 0.15
    texture = gaussian_filter(texture, sigma=0.3)
    digit_mask *= (1.0 + texture * (digit_mask > 0.1))
    
    # Small random gaps
    gaps = np.random.rand(*digit_mask.shape) > 0.98
    gaps = gaussian_filter(gaps.astype(float), sigma=0.2) * 0.3
    digit_mask *= (1.0 - gaps)
    
    # Create edge gradient mask with sharp transitions
    edge_mask1 = gaussian_filter(digit_mask, sigma=0.3)
    edge_mask2 = gaussian_filter(digit_mask, sigma=0.8)
    edge_mask3 = gaussian_filter(digit_mask, sigma=1.5)
    
    # Combine masks to create a gradient transition with emphasis on sharpness
    edge_gradient = (edge_mask1 - edge_mask2) * 0.8 + (edge_mask2 - edge_mask3) * 0.2
    edge_gradient = np.clip(edge_gradient * 2.0, 0, 1)  # First clip values
    edge_gradient = np.power(np.maximum(edge_gradient, 0), 1.5)  # Then apply power to non-negative values
    
    # Add additional texture to the gradient
    edge_texture = gaussian_filter(np.random.randn(*digit_mask.shape) * 0.1, sigma=0.2)
    edge_gradient += edge_texture * (edge_gradient > 0.2)
    edge_gradient = np.clip(edge_gradient, 0, 1)
    
    # Apply gradient transparency
    alpha = digit_mask * (1.0 - edge_gradient)
    
    # Enhance contrast for sharper lines
    alpha = np.where(alpha > 0.4, alpha * 1.5, alpha * 0.5)
    alpha = np.clip(alpha, 0, 1)
    
    result = np.ones((*digit_mask.shape, 3), dtype=np.float32)
    for i in range(3):
        result[:, :, i] = 1.0 - alpha * (1.0 - color[i])
    
    # Ensure all values are within a valid range before conversion
    result = np.clip(result * 255, 0, 255)
    return result.astype(np.uint8)

def apply_pen_effect(digit_image, pressure_variation=True):
    """Applies a random pen effect to an image"""
    # List of available effects
    effects = [
        apply_pen_effect_classic_blue,
        apply_pen_effect_gel,
        apply_pen_effect_fountain,
        apply_pen_effect_ballpoint
    ]
    
    # Select a random effect
    effect_func = random.choice(effects)
    
    # Generate a random color
    color = generate_random_color()
    
    # Apply the effect
    return effect_func(digit_image, color)

class RealDigitSelector:
    """Class for working with real digit images"""
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.digit_paths = defaultdict(list)
        self.load_digit_paths()
    
    def load_digit_paths(self):
        """Loads paths to digit images"""
        for digit_dir in self.dataset_path.glob("[0-9]"):
            digit = int(digit_dir.name)
            for img_path in digit_dir.glob("*.png"):
                self.digit_paths[digit].append(img_path)
    
    def get_digit(self, class_id=None):
        """Returns a random image of the specified digit"""
        if class_id is None:
            class_id = random.randint(0, 9)
        
        if class_id in self.digit_paths and self.digit_paths[class_id]:
            # Select a random path to the image
            img_path = random.choice(self.digit_paths[class_id])
            
            # Load the image
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                # Separate color channels and alpha channel
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                mask = alpha > 0

                result = bgr.copy()
                result[~mask] = [255, 255, 255]
                
                return result, class_id
        
        raise ValueError(f"No images found for digit {class_id}")

class TouchingDigitsLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.numbers = []
        self.labels = []
        self.usage_count = defaultdict(int)
        
        # Load all numbers from the dataset
        print("Loading numbers from SyntheticDigitStrings...")
        number_dirs = list(self.dataset_path.iterdir())
        for number_dir in tqdm(number_dirs, desc="Loading directories"):
            if not number_dir.is_dir():
                continue
                
            number = number_dir.name
            txt_files = list(number_dir.glob('*.txt'))
            if not txt_files:
                continue
                
            # Take only one file from each directory for faster loading
            txt_file = random.choice(txt_files)
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:  # Skip files without annotation
                    continue
                
                # Get the image
                img_path = txt_file.with_suffix('.png')
                if not img_path.exists():
                    continue
                    
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                
                # Convert image to grayscale
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA
                        # Use RGB channels to get grayscale image
                        rgb = img[:, :, :3]
                        digit = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    else:  # RGB
                        digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:  # Already grayscale
                    digit = img
                
                # Normalize values and ensure digits are black (0) on white (255) background
                digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)
                
                # If digits are light on dark background, invert
                if np.mean(digit[digit > 127]) < np.mean(digit[digit <= 127]):
                    digit = cv2.bitwise_not(digit)
                
                # Improve contrast
                digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Parse annotation
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
        """Get a random number from the dataset"""
        if not self.numbers:
            return None
            
        # Select the number that was used the least
        idx = min(range(len(self.numbers)), key=lambda x: self.usage_count[x])
        self.usage_count[idx] += 1
        
        # Reset counters if all numbers have been used too many times
        if min(self.usage_count.values()) > 20:
            self.usage_count.clear()
            
        return self.numbers[idx]

class LetterNoiseLoader:
    def __init__(self, letters_path):
        """Loads letters for use as noise"""
        self.letters_path = Path(letters_path)
        self.letters = []
        self.letter_classes = []
        
        # Create a list of valid folders (letters a-z and A-Z)
        valid_folders = []
        for folder in self.letters_path.iterdir():
            if folder.is_dir() and len(folder.name) == 1:
                if folder.name.isalpha():  # Only letters
                    valid_folders.append(folder.name)
        
        print(f"Found {len(valid_folders)} folders with letters: {sorted(valid_folders)}")
        
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
                            # Create a white image
                            letter = np.ones_like(alpha) * 255
                            # Where alpha > 0, put 0 (black)
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

def draw_pen_line(image, start_point, end_point, color, thickness=2):
    """Draws a line in pen style with irregularities and effects"""
    # Create an empty image for the line
    line_image = np.ones_like(image) * 255
    
    # Draw the base line
    cv2.line(line_image, start_point, end_point, (0, 0, 0), thickness)
    
    # Convert to grayscale
    if len(line_image.shape) == 3:
        line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        line_gray = line_image
    
    # Apply pen effect
    return apply_pen_effect_fountain(line_gray, color)

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
                
                # Generate a random color for the line
                color = generate_random_color() * 255
                
                # Draw line segments
                for i in range(len(points) - 1):
                    # Get line segment in pen style
                    line_segment = draw_pen_line(
                        noise_layer, 
                        points[i], 
                        points[i + 1],
                        color,
                        thickness=random.randint(1, 2)
                    )
                    
                    # Define area for insertion
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    min_x = max(0, min(x1, x2) - 5)
                    max_x = min(image.shape[1], max(x1, x2) + 5)
                    min_y = max(0, min(y1, y2) - 5)
                    max_y = min(image.shape[0], max(y1, y2) + 5)
                    
                    # Create mask for the current segment
                    segment_mask = np.mean(line_segment[min_y:max_y, min_x:max_x], axis=2) < 250
                    segment_mask = segment_mask.astype(np.float32)
                    
                    # Apply segment to noise_layer
                    noise_layer[min_y:max_y, min_x:max_x] = (
                        noise_layer[min_y:max_y, min_x:max_x] * (1 - segment_mask[:, :, np.newaxis]) +
                        line_segment[min_y:max_y, min_x:max_x] * segment_mask[:, :, np.newaxis]
                    )
    
    # Add lines to the image
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
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            # Create mask from alpha channel
                            alpha = img[:, :, 3]
                            # Normalize alpha channel
                            alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
                            # Invert mask (black digits on white background)
                            digit = cv2.bitwise_not(alpha)
                        else:
                            # If no alpha channel, convert to grayscale
                            digit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # Invert if needed (assuming digits are light on dark background)
                            if np.mean(digit) > 127:
                                digit = cv2.bitwise_not(digit)
                        
                        # Ensure digits are black on white background
                        if np.mean(digit[digit > 0]) < 127:
                            digit = cv2.bitwise_not(digit)
                        
                        # Normalize values
                        digit = cv2.normalize(digit, None, 0, 255, cv2.NORM_MINMAX)
                        
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
    """Adds letter noise to an image of the same size as digits"""
    if random.random() > probability or not letter_loader:
        return image
    
    # Determine the number of letters to add
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
    
    # Create a list of already occupied areas to avoid overlaps
    occupied_areas = []
    added_count = 0
    
    for _ in range(num_to_add):
        # Get a random letter
        letter_img, letter_class = letter_loader.get_random_letter()
        if letter_img is None:
            continue

        # Calculate letter width while maintaining proportions
        aspect_ratio = letter_img.shape[1] / letter_img.shape[0]
        width = int(letter_height * aspect_ratio)

        letter_img = cv2.resize(letter_img, (width, letter_height))
        
        # Apply pen effect to the letter
        colored_letter = apply_pen_effect(letter_img)
        
        # Create mask from the colored image
        letter_mask = np.mean(colored_letter, axis=2) < 250
        letter_mask = letter_mask.astype(np.float32)
        
        # Change letter size and mask
        # resized_letter = cv2.resize(colored_letter, (width, letter_height))
        # resized_mask = cv2.resize(letter_mask, (width, letter_height))
        resized_letter = colored_letter
        resized_mask = letter_mask
        
        # Place the letter in the selected area
        max_attempts = 50
        for attempt in range(max_attempts):
            # Random position in the selected range
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
                # Expand mask to 3 channels
                mask = np.dstack([resized_mask] * 3)
                # Blend with background
                roi[:] = roi * (1 - mask) + resized_letter * mask
                
                # Add area to occupied list
                occupied_areas.append(letter_area)
                added_count += 1
                break
    
    return image

def apply_high_quality_resize(digit_image, target_height, max_rotation=5):
    """
    Applies high-quality scaling to a digit image with random distortions
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
    
    # Scale the image
    resized = cv2.resize(digit_image, (target_width, target_height), interpolation=interpolation)
    
    # Create an image with padding
    padded = np.ones((padded_height, padded_width), dtype=np.uint8) * 255
    padded[padding:padding+target_height, padding:padding+target_width] = resized
    
    # With 60% probability, apply a slight right tilt of 2-7 degrees
    if random.random() < 0.8:
        angle = random.uniform(8, 28)  # Positive angle for right tilt
        rotation_matrix = cv2.getRotationMatrix2D(
            (padded_width/2, padded_height/2), -angle, 1.0)  # Negative angle for right tilt
        rotated = cv2.warpAffine(padded, rotation_matrix, (padded_width, padded_height),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=[255])
    else:
        rotated = padded
    
    # Add a slight perspective distortion
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
                                   borderValue=[255])
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
    
    # Apply a slight blur to remove artifacts
    final = cv2.GaussianBlur(final, (3, 3), 0.5)
    
    return final

def apply_color_to_digit(digit_image, color):
    """
    Applies a random pen effect to an image
    digit_image: grayscale image where digits are black (0) on white (255) background
    color: BGR color in format [B, G, R], each channel from 0 to 255
    """
    # Select a random effect
    effects = [
        apply_pen_effect_classic_blue,
        apply_pen_effect_gel,
        apply_pen_effect_fountain,
        apply_pen_effect_ballpoint
    ]
    effect_func = random.choice(effects)

    _, gray_image = cv2.threshold(digit_image, 128, 255, cv2.THRESH_BINARY)
    
    # Apply the effect
    return effect_func(gray_image, color)

def generate_quotation_mark(height, is_opening=True, is_double=False):
    """Generates an image of a quotation mark of a given height
    
    Args:
        height: image height
        is_opening: True for opening quotation mark, False for closing
        is_double: True for double quotes, False for single quotes
    """
    # Create an empty image with a slight width buffer
    width = int(height * 0.3)  # Width is approximately 30% of height
    if is_double:
        width = int(width * 1.6)  # Increase width for double quotes
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Define base parameters for lines
    line_height = int(height * 0.3)  # Height of each line
    tilt_angle = 15  # Tilt angle in degrees
    dx = int(line_height * np.tan(np.radians(tilt_angle)))  # X shift due to tilt
    
    # Define line positions
    if is_opening:
        # Opening quotation mark
        x_base = int(width * (0.7 if not is_double else 0.8))
        x_offset = int(width * 0.15)  # Distance between lines
    else:
        # Closing quotation mark
        x_base = int(width * (0.3 if not is_double else 0.2))
        x_offset = int(width * 0.15)  # Distance between lines
    
    # Calculate coordinates for the first line
    y_top1 = int(height * 0.3)
    y_bottom1 = y_top1 + line_height
    
    # Calculate coordinates for the second line
    y_top2 = y_top1
    y_bottom2 = y_bottom1
    
    # Draw lines considering tilt
    if is_opening:
        # For opening quotation mark, lines go from right to left
        cv2.line(image, (x_base, y_top1), (x_base - dx, y_bottom1), (0, 0, 0), thickness=2)
        cv2.line(image, (x_base - x_offset, y_top2), (x_base - x_offset - dx, y_bottom2), (0, 0, 0), thickness=2)
        
        if is_double:
            # Add a second pair of lines for double quotes
            x_shift = int(width * 0.3)  # Shift for the second pair
            cv2.line(image, (x_base - x_shift, y_top1), (x_base - dx - x_shift, y_bottom1), (0, 0, 0), thickness=2)
            cv2.line(image, (x_base - x_offset - x_shift, y_top2), (x_base - x_offset - dx - x_shift, y_bottom2), (0, 0, 0), thickness=2)
    else:
        # For closing quotation mark, lines go from left to right
        cv2.line(image, (x_base, y_top1), (x_base + dx, y_bottom1), (0, 0, 0), thickness=2)
        cv2.line(image, (x_base + x_offset, y_top2), (x_base + x_offset + dx, y_bottom2), (0, 0, 0), thickness=2)
        
        if is_double:
            # Add a second pair of lines for double quotes
            x_shift = int(width * 0.3)  # Shift for the second pair
            cv2.line(image, (x_base + x_shift, y_top1), (x_base + dx + x_shift, y_bottom1), (0, 0, 0), thickness=2)
            cv2.line(image, (x_base + x_offset + x_shift, y_top2), (x_base + x_offset + dx + x_shift, y_bottom2), (0, 0, 0), thickness=2)
    
    return image

def generate_multi_digit_image(digit_selector, touching_loader, real_loader, target_size=640, min_digits=5, max_digits=10, max_attempts=50):
    """Generates an image with digits and their masks"""
    for attempt in range(max_attempts):
        # Create a background as paper texture
        image = create_paper_texture((target_size, target_size))
        
        boxes = []  # [x1, y1, x2, y2, class_id]
        masks = []  # [(mask, class_id), ...]
        base_height = int(target_size * 0.12)
        num_digits = random.randint(min_digits, max_digits)
        # Collect information about all digits
        digit_info = []
        total_width = 0
        
        # Determine if quotes will be added (40% probability)
        add_quotes = random.random() < 0.4
        # Determine if quotes will be double (30% probability)
        is_double_quotes = random.random() < 0.3 if add_quotes else False
        quote_width = int(base_height * 0.3 * (1.6 if is_double_quotes else 1.0)) if add_quotes else 0
        quote_spacing = int(base_height * 0.2) if add_quotes else 0
        
        # Generate only single digits
        if digit_selector is not None:
            for _ in range(num_digits):
                func = np.random.choice([digit_selector.get_digit, real_loader.get_digit])
                digit, label = func()
                aspect_ratio = digit.shape[1] / digit.shape[0]
                width = int(base_height * aspect_ratio)
                if func == real_loader.get_digit: width = int(width/2)
                resized_digit = cv2.resize(digit, (width, base_height),
                                          interpolation=cv2.INTER_LANCZOS4)
                
                if func == digit_selector.get_digit:
                    # Apply color to digits
                    color = generate_random_color() * 255
                    colored_digit = apply_color_to_digit(resized_digit, color)
                else:
                    colored_digit = resized_digit
                
                # Create mask from the colored image
                digit_mask = np.mean(colored_digit, axis=2) < 250
                digit_mask = digit_mask.astype(np.float32)
                
                digit_info.append((width, base_height, colored_digit, digit_mask, label))
                total_width += width
            
            # Base spacing between digits
            spacing = int(base_height * 0.1)
            total_width += spacing * (num_digits - 1)
            
            # Add quote width and spacing
            if add_quotes:
                total_width += 2 * quote_width + 2 * quote_spacing
            
            if total_width > target_size * 0.9:
                continue
            
            # Center horizontally and vertically
            start_x = (target_size - total_width) // 2
            start_y = (target_size - base_height) // 2
            current_x = start_x
            
            # Add opening quote
            if add_quotes:
                # Generate opening quote
                opening_quote = generate_quotation_mark(base_height, is_opening=True, is_double=is_double_quotes)
                color = generate_random_color() * 255
                colored_quote = apply_color_to_digit(opening_quote, color)
                
                # Ensure sizes match
                roi = image[start_y:start_y + base_height, current_x:current_x + quote_width]
                colored_quote_resized = cv2.resize(colored_quote, (quote_width, base_height))
                quote_mask = np.mean(colored_quote_resized, axis=2) < 250
                quote_mask = quote_mask.astype(np.float32)
                
                # Place opening quote
                roi[:] = roi * (1 - quote_mask[:, :, np.newaxis]) + colored_quote_resized * quote_mask[:, :, np.newaxis]
                current_x += quote_width + quote_spacing
            
            # Place digits
            for width, height, colored_digit, digit_mask, label in digit_info:
                # Create full mask for the current digit
                full_mask = np.zeros((target_size, target_size), dtype=np.float32)
                full_mask[start_y:start_y + height, current_x:current_x + width] = digit_mask
                
                # Place the digit on the image
                roi = image[start_y:start_y + height, current_x:current_x + width]
                roi[:] = roi * (1 - digit_mask[:, :, np.newaxis]) + colored_digit * digit_mask[:, :, np.newaxis]
                
                boxes.append([current_x, start_y, current_x + width, start_y + height, label])
                masks.append((full_mask, label))
                current_x += width + spacing
            
            # Add closing quote
            if add_quotes:
                current_x -= spacing  # Remove last spacing
                current_x += quote_spacing
                # Generate closing quote
                closing_quote = generate_quotation_mark(base_height, is_opening=False, is_double=is_double_quotes)
                color = generate_random_color() * 255
                colored_quote = apply_color_to_digit(closing_quote, color)
                
                # Ensure sizes match
                roi = image[start_y:start_y + base_height, current_x:current_x + quote_width]
                colored_quote_resized = cv2.resize(colored_quote, (quote_width, base_height))
                quote_mask = np.mean(colored_quote_resized, axis=2) < 250
                quote_mask = quote_mask.astype(np.float32)
                
                # Place closing quote
                roi[:] = roi * (1 - quote_mask[:, :, np.newaxis]) + colored_quote_resized * quote_mask[:, :, np.newaxis]
        
        if len(boxes) >= min_digits:
            return image, boxes, masks
    
    print("Failed to place all digits correctly after multiple attempts")
    return image, boxes, []

def convert_mask_to_yolo_segments(mask, image_width, image_height):
    """Converts mask to YOLO segmentation format (list of normalized contour coordinates)"""
    # Find contours of the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour to reduce the number of points
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Convert contour to normalized coordinates
    contour = contour.reshape(-1, 2)
    normalized_contour = contour.astype(np.float32)
    normalized_contour[:, 0] = normalized_contour[:, 0] / image_width
    normalized_contour[:, 1] = normalized_contour[:, 1] / image_height
    
    return normalized_contour.flatten().tolist()

def main():
    # ============================================================================
    # CONFIGURATION FOR DATASET GENERATION
    # ============================================================================
    
    SEGMENTATION = False
    DETECTION = True

    # a) Use Touching Digits dataset
    USE_TOUCHING_DIGITS = False  # Disable using touching digits
    TOUCHING_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Touching-Digits/SyntheticDigitStrings"
    
    # b) Use Single Digits for this dataset (Single Digits)
    USE_SINGLE_DIGITS = True
    SINGLE_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Single-Digits/dataset"
    
    # c) Use real digits
    USE_REAL_DIGITS = True
    REAL_DIGITS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Real-Digits/"

    # d) Add annotations for the letter dataset (system-wise without annotation)
    USE_LETTER_ANNOTATIONS = True  # Letters are added as noise without annotation
    LETTERS_PATH = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Datasets/Hnd-Letters/Img"
    
    # e) Pen effect parameters
    PEN_PRESSURE_VARIATION = True  # Enable pen pressure variation
    PAPER_TEXTURE_INTENSITY = 0.08  # Paper texture intensity (reduced)
    
    # ============================================================================
    # GENERATION PARAMETERS
    # ============================================================================
    num_images = 16000  # Number of generated images
    target_size = 640  # Image size
    min_digits = 5    # Minimum number of digits
    max_digits = 10   # Maximum number of digits
    
    # ============================================================================
    
    # Create output directories
    output_dir = Path("/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/yolo-dataset-v12")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize digit selector
    digit_selector = None
    if USE_SINGLE_DIGITS:
        try:
            digit_selector = DigitSelector(SINGLE_DIGITS_PATH)
            print(f"Single digit loader initialized: {SINGLE_DIGITS_PATH}")
        except Exception as e:
            print(f"Error initializing single digit loader: {e}")
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

        # Initialize touching digits loader
    real_loader = None
    if USE_REAL_DIGITS:
        try:
            real_loader = RealDigitSelector(REAL_DIGITS_PATH)
            print(f"Real digits loader initialized: {REAL_DIGITS_PATH}")
        except Exception as e:
            print(f"Error initializing real digits loader: {e}")
            USE_REAL_DIGITS = False
    
    # Initialize letter noise loader
    letter_loader = None
    if USE_LETTER_ANNOTATIONS:
        try:
            letter_loader = LetterNoiseLoader(LETTERS_PATH)
            print("Letter noise loader initialized successfully")
        except Exception as e:
            print(f"Error initializing letter loader: {e}")
            USE_LETTER_ANNOTATIONS = False
    
    # Check if at least one digit loader is working
    if not USE_SINGLE_DIGITS and not USE_TOUCHING_DIGITS:
        raise ValueError("At least one digit source (USE_SINGLE_DIGITS or USE_TOUCHING_DIGITS) must be enabled")
    
    # Generate images
    for i in tqdm(range(num_images)):
        image, boxes, masks = generate_multi_digit_image(digit_selector, touching_loader, real_loader, target_size, min_digits, max_digits)
        
        # Add noise and lines to the image
        image = add_noise_and_lines(image, boxes)
        
        # Add letter noise
        image = add_letter_noise(image, letter_loader, boxes)
        
        # Save image
        image_path = images_dir / f"image_{i:04d}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save labels in YOLO format with segmentation
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, 'w') as f:
                if SEGMENTATION:
                    for mask, class_id in masks:
                        # Convert mask to YOLO segmentation format
                        segments = convert_mask_to_yolo_segments(mask, image.shape[1], image.shape[0])
                        if segments:
                            # Write in format: class_id x1 y1 x2 y2 ... xn yn
                            f.write(f"{class_id} {' '.join(map(str, segments))}\n")
                if DETECTION:
                    for box in boxes:
                        x_min, y_min, x_max, y_max, class_id = box
                        yolo_box = convert_to_yolo_format([x_min, y_min, x_max, y_max], 
                                                        image.shape[1], image.shape[0])
                        f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

if __name__ == "__main__":
    main()  