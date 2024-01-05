import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./pictures/landscape.jpg', 0)

# Histogram equalization
equ = cv2.equalizeHist(img)

plt.figure(figsize=(8, 4))

# Base image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Plotting the image after histogram equalization
plt.subplot(1, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('After Histogram Equalization ')

plt.show()
