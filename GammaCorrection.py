import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=1.0):

    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0
    # Apply gamma correction
    corrected_image = np.power(normalized_image, gamma)
    # Rescale the corrected image to the range [0, 255] and convert to uint8
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image

# Read the image
img = cv2.imread('./pictures/landscape.jpg')

# Convert to RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Gamma correction
gamma_value = 2
corrected_img = gamma_correction(img, gamma_value)

# Display
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Base Image')

plt.subplot(1, 2, 2)
plt.imshow(corrected_img)
plt.title(f'Gamma Correction (γ={gamma_value})')

plt.show()
