import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):

    # Define the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Contrast stretching operation
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))

    # Convert to uint8
    stretched_image = stretched_image.astype(np.uint8)

    return stretched_image

# Load the image
img = cv2.imread('./pictures/landscape.jpg', cv2.IMREAD_GRAYSCALE)

# Apply contrast stretching
stretched_img = contrast_stretching(img)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(stretched_img, cmap='gray', vmin=0, vmax=255)
plt.title('Contrast Stretching Result')

plt.show()
