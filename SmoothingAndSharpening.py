import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, kernel_size):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def apply_laplacian_filter(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Take the absolute value to get magnitude
    laplacian = np.abs(laplacian)
    return laplacian

# Load the image
img = cv2.imread('./pictures/landscape.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Smoothing (Gaussian Blur)
kernel_size_smooth = 5
smoothed_img = apply_gaussian_blur(img, kernel_size_smooth)

# Sharpening (Laplacian Filter)
sharpened_img = apply_laplacian_filter(img)

# Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 3, 2)
plt.imshow(smoothed_img)
plt.title('Smoothing (Gaussian Blur)')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_img, cmap='gray')
plt.title('Sharpening (Laplacian Filter)')

plt.show()
