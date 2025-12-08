"""
Image Preprocessing Utility for Vision Models

This module provides a simple utility function to load and prepare images for 
computer vision model inference or training pipelines.

Function: load_image()
- Loads an image from disk using OpenCV (BGR color space by default)
- Resizes to a target dimension for consistent model input
- Default size: 224x224 pixels (common for many CNN architectures like ResNet, VGG)

Parameters:
- path: File path to the input image (jpg, png, etc.)
- target_size: Tuple specifying desired (width, height) in pixels

Returns:
- Numpy array of shape (height, width, 3) with BGR color channels
- Pixel values in range [0, 255] as uint8

Usage Examples:
    # Load for standard CNN input
    img = load_image("photo.jpg")  # Returns 224x224x3 array
    
    # Load for YOLOv8 input
    img = load_image("frame.jpg", target_size=(640, 640))
    
    # Load for custom model
    img = load_image("thermal.png", target_size=(512, 512))

Notes:
- No normalization applied (add img/255.0 if needed for neural networks)
- Maintains aspect ratio distortion (consider letterboxing for production)
- BGR format (convert to RGB if required: cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

Use Case: Preprocessing step for vision-based human detection in ResQBot,
ensuring webcam frames or saved images are properly sized before feeding
to YOLOv8 or other vision models in the multi-modal fusion pipeline.
"""

import cv2
import numpy as np

def load_image(path, target_size = (224,224)):
    """" Load and Process Data """
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    return img

