import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

def visualize_detection(image, results, conf_threshold=0.5):
    """
    Visualizes detection results on the image
    
    Args:
        image: Source image
        results: Results from YOLO model
        conf_threshold: Confidence threshold for displaying predictions
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Generate random colors for each class
    num_classes = 10  # for digits from 0 to 9
    colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(num_classes)}
    
    # Process results
    for result in results:
        # Get boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()
            
            for box in boxes:
                # Get coordinates, confidence and class
                x1, y1, x2, y2, confidence, class_id = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = int(class_id)
                
                if confidence < conf_threshold:
                    continue
                
                # Get color for class
                color = colors[class_id]
                
                # Draw box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add text with class and confidence
                text = f"{class_id}: {confidence:.2f}"
                cv2.putText(vis_image, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image

def preprocess_image(image):
    """
    Image preprocessing:
    1. Convert to grayscale
    2. Binarization using Otsu's method
    3. Inversion (black text on white background)
    """
    # Convert to grayscale if image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Otsu's binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Convert back to BGR for YOLO
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary_bgr

def test_detection_model(model_path, test_dir, output_dir, conf_threshold=0.25):
    """
    Testing detection model on images
    """
    # Load model
    model = YOLO(model_path)
    
    # Create directory for results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of test images
    test_dir = Path(test_dir)
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} images for testing")
    
    # Process each image
    for img_path in image_files:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Preprocess image
        preprocessed = preprocess_image(image)
        
        # Get predictions
        results = model.predict(preprocessed, conf=conf_threshold)
        
        # Draw results on original image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Draw box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add text with class and confidence
                label = f"{cls}: {conf:.2f}"
                cv2.putText(image, label, (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Visualize results
        vis_image = visualize_detection(preprocessed, results, conf_threshold)
        
        # Save result
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), vis_image)
        print(f"Saved result to {output_path}")

def main():
    # Paths to files and directories
    model_path = "runs/digits_detect-7/weights/epoch36.pt"  # Path to model weights
    test_dir = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/test-data/hard-test-yolo/images"  # Directory with test images
    output_dir = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/results"  # Directory for results
    
    # Start testing
    test_detection_model(model_path, test_dir, output_dir)

if __name__ == "__main__":
    main()