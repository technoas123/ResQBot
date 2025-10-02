from ultralytics import YOLO

# Load trained model
model = YOLO(r"runs\detect\train\weights\best.pt")

# Validate on your validation dataset
metrics = model.val(data="data.yaml")

# Print accuracy metrics
print(f"Precision: {metrics.box.map:.3f}")
print(f"Recall: {metrics.box.map50:.3f}")
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
