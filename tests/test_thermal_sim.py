# tests/test_thermal_sim.py
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