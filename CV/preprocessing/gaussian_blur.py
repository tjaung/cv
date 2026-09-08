import cv2

def apply_gaussian_blur(image, kernel_size=5):
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('Kernel size must be a positive odd integer')
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
