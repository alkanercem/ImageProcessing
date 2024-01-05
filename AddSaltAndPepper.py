import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):

    noisy_image = np.copy(image)

    # Salt noise
    salt_noise = np.random.rand(*image.shape[:2])
    noisy_image[salt_noise < salt_prob] = 255

    #  Pepper noise
    pepper_noise = np.random.rand(*image.shape[:2])
    noisy_image[pepper_noise < pepper_prob] = 0

    return noisy_image

# Load the original image
original_img = cv2.imread('./pictures/landscape.jpg')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Parameters for salt and pepper noise
salt_probability = 0.04
pepper_probability = 0.04

# Add salt and pepper noise
noisy_img = add_salt_and_pepper_noise(original_img, salt_probability, pepper_probability)

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Base Image')

plt.subplot(1, 2, 2)
plt.imshow(noisy_img)
plt.title('Image with Salt and Pepper Noise')

plt.show()
