"""
YOLOv8 Nano Vision Model Training for Human Detection

This module fine-tunes the YOLOv8 Nano object detection model for real-time human
detection in disaster response scenarios. YOLOv8n is optimized for speed and 
efficiency on edge devices while maintaining good detection accuracy.

Model Specifications:
- Architecture: YOLOv8 Nano (smallest, fastest variant)
- Pre-trained weights: COCO dataset (80 classes including 'person')
- Target use: Real-time inference on rescue robot hardware
- Typical speed: 30-60 FPS on modern hardware

Training Configuration:
- Epochs: 50 (adjustable based on dataset size and convergence)
- Image size: 512x512 pixels (balance between speed and accuracy)
- Batch size: 16 (adjust based on GPU memory availability)
- Augmentation: Enabled (random flips, crops, color jitter for robustness)

Dataset Requirements (data.yaml):
- Must specify paths to train/val image directories
- Class definitions (typically: human, person, survivor)
- Annotation format: YOLO format (class x_center y_center width height)

Expected data.yaml structure:
    train: path/to/train/images
    val: path/to/val/images
    nc: 1  # number of classes
    names: ['person']

Output:
- runs/detect/train/weights/best.pt: Best model checkpoint
- runs/detect/train/weights/last.pt: Final epoch checkpoint
- Training metrics, loss curves, and sample predictions in runs/detect/train/

Usage:
    python script_name.py
    
Post-Training:
- Export best.pt for deployment: model.export(format='onnx') for optimization
- Integrate with multi-modal fusion system for vision-based human detection

Use Case: Provides the vision component for ResQBot's multi-modal human detection
system, identifying persons in webcam/camera feeds even in challenging disaster
environments (debris, poor lighting, partial occlusion).
"""


from ultralytics import YOLO

# Load YOLOv8 Nano (fast & lightweight)
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='data.yaml',   # your dataset config
    epochs=50,          # number of epochs
    imgsz=512,          # input image size
    batch=16,           # batch size
    augment=True        # apply data augmentation
)
