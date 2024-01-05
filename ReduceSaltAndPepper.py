import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_median_blur(image, kernel_size):
    # Apply median blur
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image


# Load the image with salt and pepper noise
img = cv2.imread('./pictures/salt_pepper_noise.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply median blur to reduce salt and pepper noise
kernel_size_median_blur = 3
blurred_img = apply_median_blur(img, kernel_size_median_blur)

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 3, 2)
plt.imshow(blurred_img)
plt.title('Median Blur (Salt & Pepper Reduction)')


plt.show()
