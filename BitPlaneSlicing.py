import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_plane_slice(image_path, bit_level):

    # Read the image (in color)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract a specific bit plane
    bit_plane = (gray_image >> bit_level) & 1

    # Visualization
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {bit_level}')
    plt.axis('off')
    plt.show()

# Example usage
image_path = './pictures/landscape.jpg'
bit_level = 4  # Bit level
bit_plane_slice(image_path, bit_level)
