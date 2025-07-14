from ultralytics import YOLO
from pathlib import Path

def train_yolo_detection(dataset_path: str, epochs: int = 100):
    """Trains YOLO model for detection"""
    # Load pre-trained YOLOv8 model for detection
    model = YOLO('yolov8n.pt')
    
    # Path to dataset configuration
    yaml_path = str(Path(dataset_path) / 'data.yaml')
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=32,
        device='0',  # use GPU if available
        patience=20,  # early stopping
        save=True,  # save best weights
        save_period=1,  # save weights every epoch
        save_json=True,  # save metrics to JSON
        plots=True,  # generate plots
        project='runs',  # project name
        name='digits_detect-12_with_real_digits',  # experiment name
        pretrained=True,  # use pretrained weights
        optimizer='Adam',  # optimizer
        lr0=0.001,  # initial learning rate
        lrf=0.01,  # final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        hsv_h=0.4,  # minimum hue change
        hsv_s=0.4,  # minimum saturation change
        hsv_v=0.4,  # minimum brightness change
        degrees=0.09,  # slight rotation
        translate=0.1,  # slight shift
        scale=0.3,    # moderate scaling
        perspective=0.0,  # disable perspective
        flipud=0.0,   # disable vertical flip
        fliplr=0.0,   # disable horizontal flip
        mosaic=0.12,   # disable mosaic
        mixup=0.0,    # disable mixup
        copy_paste=0.0,  # disable copy-paste
    )
    
    return results

def main():
    # Path to dataset
    dataset = "/home/optimus/Desktop/Work/EasyData/Hand-Written-Digits-Detection/Yolo/Dataset-v12"
    
    # Train the model
    print("Starting model training...")
    results = train_yolo_detection(dataset, epochs=100)
    print("Training completed!")

if __name__ == "__main__":
    main() 