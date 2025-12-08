"""
Detection Result Visualization Utility

This module provides a simple visualization function to display detection results
from the ResQBot multi-modal human detection system. Shows the analyzed image
alongside the model's confidence score for human presence.

Function: plot_detection_result()
- Displays an image with its corresponding detection prediction score
- Uses matplotlib for quick visual inspection during development/testing
- Useful for debugging, demonstration, and qualitative evaluation

Parameters:
- image: Input image as numpy array (RGB or BGR format)
  Typically from webcam, saved file, or disaster scene
- prediction: Float value representing detection confidence [0.0-1.0]
  Values closer to 1.0 indicate higher confidence of human presence
  Can also be class probabilities or fusion scores

Display Format:
- Image rendered without axes (clean presentation)
- Title shows numeric confidence score (2 decimal precision)
- Example: "Detection Score: 0.87" indicates 87% confidence

Usage Examples:
    # Single model prediction
    img = cv2.imread("scene.jpg")
    score = model.predict(img)
    plot_detection_result(img, score)
    
    # Multi-modal fusion result
    fusion_score = 0.85  # 2 out of 3 modalities detected human
    plot_detection_result(webcam_frame, fusion_score)
    
    # Batch visualization during validation
    for img, pred in zip(val_images, predictions):
        plot_detection_result(img, pred)

Note: Missing parentheses on plt.show (should be plt.show()) - will not display
until this is fixed. Add () to make the plot actually render.

Enhancement Suggestions:
- Add bounding box overlay for localized detection
- Color-code title based on threshold (green=human, red=no human)
- Display all three modality scores (audio, thermal, vision)
- Save output instead of/in addition to displaying
- Add timestamp and location metadata

Use Case: Quick visualization tool for inspecting ResQBot's detection performance,
useful during model development, testing, and demonstrating system capabilities
to stakeholders or rescue teams.
"""

import matplotlib.pyplot as plt

def plot_detection_result(image, prediction):
    """ plot image with detection result """
    plt.imshow(image)
    plt.title(f'Detection Score: {prediction:.2f}')
    plt.axis('off')
    plt.show()  # Fixed: added parentheses