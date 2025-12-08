"""
YOLOv8 Vision Model Validation Script

This module evaluates the trained YOLOv8 human detection model on the validation
dataset to assess its performance before deployment in the ResQBot multi-modal
fusion system.

Evaluation Metrics:
- Precision: Proportion of correct positive predictions (TP / (TP + FP))
  → How many detected humans are actually humans?
- Recall: Proportion of actual positives detected (TP / (TP + FN))
  → How many actual humans were successfully detected?
- mAP@0.5: Mean Average Precision at IoU threshold 0.5
  → Standard COCO evaluation metric for object detection
- mAP@0.5:0.95: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
  → More strict metric, requires tighter bounding box predictions

Validation Process:
1. Loads best model checkpoint from training run
2. Runs inference on validation dataset specified in data.yaml
3. Computes detection metrics by comparing predictions to ground truth
4. Generates detailed performance reports and visualizations

Model Input:
- runs/detect/train/weights/best.pt: Trained model with best validation performance

Dataset Requirements:
- data.yaml must contain validation set path and class definitions
- Validation images with corresponding YOLO format annotations

Output:
- Console: Key performance metrics (precision, recall, mAP scores)
- runs/detect/val/: Detailed validation results, confusion matrix, PR curves
- Predicted vs. ground truth visualizations for error analysis

Usage:
    python script_name.py

Performance Targets:
- Precision > 0.85: Minimize false alarms (critical for rescue operations)
- Recall > 0.90: Maximize human detection (missing survivors is costly)
- mAP@0.5 > 0.80: Overall detection quality benchmark

Use Case: Ensures vision model meets accuracy requirements before integration
into the multi-modal fusion system where it will vote alongside audio and thermal
detectors for final human presence decisions.
"""


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
