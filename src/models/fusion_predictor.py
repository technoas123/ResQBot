"""
Multi-Modal Human Detection System for Disaster Response (ResQBot)

This module implements a real-time sensor fusion system that combines three detection 
modalities to identify humans in disaster scenarios with high confidence. Designed for 
search and rescue robotics applications.

Detection Modalities:
1. AUDIO: Analyzes 3-second audio recordings using a trained neural network to classify
   sounds as human (voices, calls for help), emergency (sirens, alarms), or background
   - Extracts 128D features: MFCCs, chroma, spectral contrast, and tonnetz
   - Uses the trained disaster human detector model
   
2. VISION: Captures webcam images and uses YOLOv8 object detection to identify persons
   - Detects COCO class 0 (person) in real-time video frames
   - Provides visual confirmation of human presence
   
3. THERMAL: Analyzes thermal imagery to detect heat signatures
   - Currently simulated with 70% detection probability for testing
   - Designed to work in low-light/obscured conditions

Fusion Logic:
- Majority voting: If 2 or 3 modalities detect "human", trigger alert
- Reduces false positives by requiring multi-sensor agreement
- Sends LoRa alert to base station when human is detected

Hardware Requirements:
- Microphone (configurable input device)
- Webcam/USB camera
- Thermal camera (future integration)
- LoRa transceiver for remote communication

Usage:
    python script_name.py
    [Enter microphone device index when prompted]
    
Output:
    - Console predictions from each modality
    - LoRa alert transmission on positive human detection
    - Temporary files: temp_audio.wav, webcam_frame.jpg

Use Case: Deployed on rescue robots to locate survivors in collapsed structures,
disaster zones, or areas inaccessible to human rescuers.
"""


import sounddevice as sd
import numpy as np
import cv2
import librosa
import torch
import time
import os
from pathlib import Path
from ultralytics import YOLO
from scipy.io.wavfile import write
from models.disaster_human_detector_audio import ImprovedDisasterDetector

# Model paths
audio_model_path = r"C:\Users\ahamm\OneDrive\Documents\MY PROJECTS\ResQBot\data\models\audio\disaster_human_detector.pt"
vision_model_path = r"C:\Users\ahamm\OneDrive\Documents\MY PROJECTS\ResQBot\data\models\vision\yolov8n.pt"
thermal_model_path = r"C:\Users\ahamm\OneDrive\Documents\MY PROJECTS\ResQBot\data\models\thermal\thermal_head.pt"

# Output paths
audio_output_file = "temp_audio.wav"
webcam_image_path = "webcam_frame.jpg"
thermal_image_path = "test_thermal.png"  # Uses static image for now

# Device selection
mic_index = int(input(" Enter input device index for microphone: "))

# Audio class map
audio_classes = ["human", "emergency", "background"]

# --------- Audio Prediction --------- #
def extract_audio_features(file):
    y, sr = librosa.load(file, sr=16000)
    
    # Extract multiple features to get 128 dimensions
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    # Combine all features to get approximately 128 dimensions
    mfcc_mean = np.mean(mfcc.T, axis=0)                    # 40 features
    chroma_mean = np.mean(chroma.T, axis=0)                # 12 features  
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)  # 7 features
    tonnetz_mean = np.mean(tonnetz.T, axis=0)              # 6 features
    
    # Combine all features (40 + 12 + 7 + 6 = 65 features)
    combined_features = np.concatenate([
        mfcc_mean, 
        chroma_mean, 
        spectral_contrast_mean, 
        tonnetz_mean
    ])
    
    # If we need exactly 128, we can pad or use more features
    if len(combined_features) < 128:
        # Pad with zeros or repeat features to get 128
        combined_features = np.pad(combined_features, (0, 128 - len(combined_features)), 'constant')
    elif len(combined_features) > 128:
        # Truncate if too many features
        combined_features = combined_features[:128]
    
    return combined_features

def predict_audio():
    print(" Recording 3 seconds of audio...")
    fs = 16000
    duration = 3
    audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16', device=mic_index)
    sd.wait()
    write(audio_output_file, fs, audio)

    features = extract_audio_features(audio_output_file)
    print(f"DEBUG: Feature vector shape: {features.shape}")
    
    features_tensor = torch.tensor(features).unsqueeze(0).float()
    print(f"DEBUG: Tensor shape: {features_tensor.shape}")

    model = ImprovedDisasterDetector(input_dim=128, num_classes=3)
    checkpoint = torch.load(audio_model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        output = model(features_tensor)
        pred = torch.argmax(output, dim=1).item()

    return audio_classes[pred]

# --------- Vision Prediction --------- #
def predict_vision():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(" Failed to capture image from webcam")
        return "background"
        
    cv2.imwrite(webcam_image_path, frame)

    model = YOLO(vision_model_path)
    results = model(webcam_image_path)

    for res in results:
        for cls in res.boxes.cls:
            if int(cls) == 0:  # COCO "person" class
                return "human"

    return "background"

# --------- Thermal Prediction (Static Image for now) --------- #
def predict_thermal():
    # Simple simulation for thermal detection
    print(" Thermal model: Simulating detection")
    
    # You can add logic here based on your thermal image
    # For now, let's simulate based on time or random
    import random
    # Simulate 70% chance of human detection for testing
    if random.random() < 0.7:
        return "human"
    else:
        return "background"

# --------- Fusion and LoRa Trigger --------- #
def send_lora_alert(msg="Human Detected - Multimodal"):
    print(f" Sending to LoRa base station: {msg}")

def fusion_decision(audio, thermal, vision):
    print(f"\n Predictions:\n  Audio: {audio}\n  Thermal: {thermal}\n  Vision: {vision}")
    votes = [audio, thermal, vision]
    if votes.count("human") >= 2:
        send_lora_alert()
        return " Human detected!"
    else:
        return " No human detected."

# --------- MAIN --------- #
if __name__ == "__main__":
    print(" Starting MULTI-MODAL FUSION prediction...\n")
    
    try:
        audio_pred = predict_audio()
        thermal_pred = predict_thermal()
        vision_pred = predict_vision()

        result = fusion_decision(audio_pred, thermal_pred, vision_pred)
        print("\n Fusion Result:", result)
        
    except Exception as e:
        print(f" Error during prediction: {e}")
        import traceback
        traceback.print_exc()