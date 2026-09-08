import os
import cv2

def get_image(image: str):
    if not os.path.isfile(image):
        raise FileNotFoundError(f"Image not found: {image}")
    img = cv2.imread(image)

    if img is None:
        raise ValueError(f"Failed to read image: {image}")

    return img

