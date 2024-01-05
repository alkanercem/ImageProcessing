import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_segmentation(image, lower_threshold, upper_threshold):

    # Create masks for each channel using the given thresholds
    mask_red = cv2.inRange(image[:, :, 0], lower_threshold[0], upper_threshold[0])
    mask_green = cv2.inRange(image[:, :, 1], lower_threshold[1], upper_threshold[1])
    mask_blue = cv2.inRange(image[:, :, 2], lower_threshold[2], upper_threshold[2])

    segmentation_mask = cv2.bitwise_and(mask_red, cv2.bitwise_and(mask_green, mask_blue))

    # Implement the segmentation mask to the base image
    segmented_image = cv2.bitwise_and(image, image, mask=segmentation_mask)

    return segmentation_mask, segmented_image

# Load the image
img = cv2.imread('./pictures/landscape.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# RGB thresholds for segmentation
lower_threshold = [100, 50, 50]
upper_threshold = [255, 255, 255]

# Perform RGB segmentation
segmentation_mask, segmented_img = rgb_segmentation(img, lower_threshold, upper_threshold)

# Results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 3, 2)
plt.imshow(segmentation_mask, cmap='gray')
plt.title('Segmentation Mask')

plt.subplot(1, 3, 3)
plt.imshow(segmented_img)
plt.title('Segmented Image')

plt.show()
