import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_opening(image, kernel_size):
    # Define a kernel for opening operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image

def apply_closing(image, kernel_size):
    # Define a kernel for closing operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply closing operation
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image

# Load the image
img = cv2.imread('./pictures/landscape.jpg', cv2.IMREAD_GRAYSCALE)

# Opening operation
kernel_size_opening = 4
opened_img = apply_opening(img, kernel_size_opening)

# Closing operation
kernel_size_closing = 8
closed_img = apply_closing(img, kernel_size_closing)

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Base Image')

plt.subplot(1, 3, 2)
plt.imshow(opened_img, cmap='gray')
plt.title('Opening Operation')

plt.subplot(1, 3, 3)
plt.imshow(closed_img, cmap='gray')
plt.title('Closing Operation')

plt.show()
