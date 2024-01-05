import cv2
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, kernel_size):
    # Applying Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image

# Read the image
img = cv2.imread('./pictures/landscape.jpg')

# Convert the color format from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Implement the Gaussian blur
filter_size = 9  # Size of the filter kernel (odd number)
blurred_img = apply_gaussian_blur(img, filter_size)

# Plotting the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 2, 2)
plt.imshow(blurred_img)
plt.title(f'Implemented Gaussian Blur (Filter Size = {filter_size})')

plt.show()
