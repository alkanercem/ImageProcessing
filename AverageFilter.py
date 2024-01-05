import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_average_filter(image, kernel_size):

    # Average filter kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # Apply filtering
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

# Read the image
img = cv2.imread('./pictures/landscape.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to the RGB format

# Apply the average filter with kernel size
filter_kernel_size = 8
filtered_img = apply_average_filter(img_rgb, filter_kernel_size)

# Display the original and filtered images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Base Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img)
plt.title(f'After Implemented Average Filter (Kernel Size = {filter_kernel_size})')

plt.show()