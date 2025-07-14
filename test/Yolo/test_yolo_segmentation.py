from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import random

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

def test_model(model_path, test_images_dir, output_dir, conf_threshold=0.5):
    """
    Tests the model on images and saves results
    
    Args:
        model_path: Path to model weights
        test_images_dir: Directory with test images
        output_dir: Directory for saving results
        conf_threshold: Confidence threshold for displaying predictions
    """
    # Load model
    model = YOLO(model_path)
    
    # Create directory for results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of test images
    test_images = list(Path(test_images_dir).glob('*.jpg')) + \
                 list(Path(test_images_dir).glob('*.png'))
    
    print(f"Found {len(test_images)} test images")
    
    # Process each image
    for img_path in test_images:
        print(f"Processing {img_path.name}...")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Get predictions
        results = model.predict(
            source=image,
            conf=conf_threshold,
            save=False,
            save_txt=False,
            save_conf=True,
            show=False,
            stream=True
        )
        
        # Visualize results
        vis_image = visualize_segmentation(image, results, conf_threshold)
        
        # Save result
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), vis_image)
        print(f"Saved result to {output_path}")

def main():
    # Configuration
    model_path = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Yolo/runs/digits_segment-12_with_real_digits2/weights/epoch36.pt"  # path to model weights
    test_images_dir = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/test-data/hard-test-yolo/images"  # directory with test images
    output_dir = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/results"  # directory for results
    conf_threshold = 0.5  # confidence threshold
    
    # Start testing
    print("Starting model testing...")
    test_model(model_path, test_images_dir, output_dir, conf_threshold)
    print("Testing completed!")

if __name__ == "__main__":
    main() 