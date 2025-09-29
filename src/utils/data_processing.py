import cv2
import numpy as np

def load_image(path, target_size = (224,224)):
    """" Load and Process Data """
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    return img

