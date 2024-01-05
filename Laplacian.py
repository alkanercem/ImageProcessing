import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_laplacian_filter(image):

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Take the absolute value to get magnitude
    laplacian = np.abs(laplacian)

    return laplacian

# Load the image
img = cv2.imread('./pictures/landscape.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Laplacian filter
laplacian_img = apply_laplacian_filter(img)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_img, cmap='gray')
plt.title('Applied Laplacian Filter')

plt.show()
