import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def visualize_segmentation(image, results, conf_threshold=0.5):
    """
    Visualizes segmentation results on the image
    
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
        # If there are segmentation masks
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            
            for mask, box in zip(masks, boxes):
                # Get confidence and class
                confidence = box[4]
                class_id = int(box[5])
                
                if confidence < conf_threshold:
                    continue
                
                # Convert mask to image format
                mask = mask.astype(np.uint8)
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # Create colored mask
                color = colors[class_id]
                colored_mask = np.zeros_like(image)
                colored_mask[mask == 1] = color
                
                # Apply mask with transparency
                alpha = 0.5
                vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
                
                # Draw mask contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_image, contours, -1, color, 2)
                
                # Add text with class and confidence
                x1, y1 = int(box[0]), int(box[1])
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

def test_segmentation_model(model_path, test_dir, output_dir, conf_threshold=0.5):
    """
    Testing segmentation model on images
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
        results = model.predict(
            preprocessed, 
            conf=conf_threshold
        )
        
        # Visualize results
        # vis_image = visualize_segmentation(image, results, conf_threshold)
        vis_image = visualize_segmentation(preprocessed, results, conf_threshold)
        
        # Save results
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), vis_image)
        
        # Save binarized image for verification
        # binary_output_path = output_dir / f"bin_{img_path.name}"
        # cv2.imwrite(str(binary_output_path), preprocessed)

def main():
    # Paths to files and directories
    model_path = "runs/digits_segment-7/weights/epoch51.pt"  # Path to model weights
    test_dir = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/test-data/hard-test-yolo/images"  # Directory with test images
    output_dir = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/results"  # Directory for results
    conf_threshold = 0.5  # confidence threshold

    # Start testing
    print("Starting model testing...")
    test_segmentation_model(model_path, test_dir, output_dir, conf_threshold)
    print("Testing completed!")

if __name__ == "__main__":
    main()