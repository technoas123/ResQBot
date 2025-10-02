from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs\detect\train\weights\best.pt")

# Run real-time detection using webcam
model.predict(
    source=0,        # 0 = default webcam
    imgsz=512,       # same size as training
    conf=0.5,        # confidence threshold
    show=True        # show live results
)
