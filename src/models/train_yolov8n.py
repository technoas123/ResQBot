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
