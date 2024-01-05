import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_filter(image, kernel_size=3):
    # Convert the image to grayscale
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Apply Sobel filter in x and y directions
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Compute the magnitude of the gradient and normalize
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return gradient_magnitude

# Load the image
img = cv2.imread('./pictures/landscape.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Sobel filter
sobel_result = apply_sobel_filter(img)

# Display
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 2, 2)
plt.imshow(sobel_result, cmap='gray')
plt.title('Applied Sobel Filter')

plt.show()
