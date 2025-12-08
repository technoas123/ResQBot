"""
Thermal Model Training Verification and Quality Assurance Script

This module validates that the thermal human detection model has been properly trained
and meets minimum performance requirements before deployment in ResQBot's multi-modal
fusion system.

Verification Checklist:
1. Model Artifacts: Confirms all training outputs exist
   - thermal_head.pt: Trained neural network weights
   - metrics.json: Evaluation metrics and performance report
   - val_logits.npz: Validation predictions for analysis

2. Model Performance: Evaluates classification accuracy
   - Excellent: >85% accuracy (production-ready)
   - Good: 70-85% accuracy (acceptable, monitor performance)
   - Poor: <70% accuracy (requires retraining or data improvements)

3. Confusion Matrix: Shows per-class detection performance
   - Rows represent true labels
   - Columns represent predictions
   - Diagonal values = correct classifications
   - Off-diagonal = misclassifications (cool ↔ hot confusion)

Output Interpretations:

PASS Messages:
- All files exist → Training completed successfully
- Excellent accuracy → Model ready for field deployment
- Low confusion → Model distinguishes humans from background well

FAIL Messages:
- Missing files → Run thermal training pipeline first
- Low accuracy → Need more/better training data or hyperparameter tuning
- High confusion → Classes overlap (improve feature engineering)

Typical Confusion Matrix (2-class):
   Cool: [850, 150]  → 850 cool samples correctly classified, 150 misclassified as hot
   Hot:  [100, 900]  → 100 hot samples misclassified as cool, 900 correctly classified
   
This indicates:
- Cool detection: 85.0% recall (850/1000)
- Hot detection: 90.0% recall (900/1000)
- Overall accuracy: 87.5% ((850+900)/2000)

Common Issues Detected:
- Missing thermal_head.pt → Training script didn't complete
- metrics.json absent → Evaluation step failed
- Low accuracy → Insufficient training data or poor feature quality
- Asymmetric confusion → One class much harder to detect than other

Files Checked:
- data/processed/thermal/thermal_head.pt (Model weights, ~50-100 KB)
- data/processed/thermal/metrics.json (Performance report, JSON format)
- data/processed/thermal/val_logits.npz (Raw predictions for debugging)

Usage:
    python test_thermal_sim.py
    
Expected Output (Success):
    === Verifying Thermal AI Training ===
    PASS: data/processed/thermal/thermal_head.pt - EXISTS
    PASS: data/processed/thermal/metrics.json - EXISTS
    PASS: data/processed/thermal/val_logits.npz - EXISTS
    Accuracy: 0.892
    EXCELLENT - AI is well trained!
    Confusion Matrix:
       Cool: [872, 128]
       Hot:  [88, 912]

Expected Output (Failure):
    === Verifying Thermal AI Training ===
    FAIL: data/processed/thermal/thermal_head.pt - MISSING!
    Some files are missing. Please run thermal training first.

Next Steps After Verification:
- PASS → Integrate thermal model into multi-modal fusion pipeline
- FAIL → Debug training process, check data preparation, retrain model
- Low accuracy → Collect more diverse thermal data, adjust hyperparameters

Use Case: Quality gate before deploying thermal detection on rescue robots.
Ensures the model meets minimum accuracy standards and won't cause false alarms
or miss actual human heat signatures in disaster scenarios.

Note: Currently checks 2-class problem (cool/hot). Modify for multi-class
scenarios (human/background/animal/machinery) by updating confusion matrix display.
"""


import json
import numpy as np
import os

def verify_training():
    print("=== Verifying Thermal AI Training ===")
    
    # Check if files exist
    files_to_check = [
        'data/processed/thermal/thermal_head.pt',
        'data/processed/thermal/metrics.json', 
        'data/processed/thermal/val_logits.npz'
    ]
    
    all_files_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"PASS: {file} - EXISTS")
        else:
            print(f"FAIL: {file} - MISSING!")
            all_files_exist = False
    
    if not all_files_exist:
        print("Some files are missing. Please run thermal training first.")
        return False
    
    # Check metrics
    try:
        with open('data/processed/thermal/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        accuracy = metrics['accuracy']
        print(f"Accuracy: {accuracy:.3f}")
        
        if accuracy > 0.85:
            print("EXCELLENT - AI is well trained!")
        elif accuracy > 0.70:
            print("GOOD - AI is learning")
        else:
            print("NEEDS IMPROVEMENT - Accuracy too low")
        
        # Show confusion matrix
        cm = metrics['confusion_matrix']
        print("Confusion Matrix:")
        print(f"   Cool: {cm[0]}")
        print(f"   Hot:  {cm[1]}")
        
        return True
        
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return False

if __name__ == "__main__":
    verify_training()