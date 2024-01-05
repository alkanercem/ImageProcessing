import cv2
import numpy as np
import matplotlib.pyplot as plt

def contraharmonic_mean_filter(image, kernel_size, Q):

    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
    result_image = np.zeros_like(image, dtype=np.float32)

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1].astype(np.float32)
            numerator = np.sum(np.power(window, Q + 1))
            denominator = np.sum(np.power(window, Q))

            if denominator != 0:
                result_image[i - pad_size, j - pad_size] = numerator / denominator

    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    return result_image

# Example usage with different Q values
img = cv2.imread('./pictures/ContraharmonicMean.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Contraharmonic Mean Filter with different Q values
filtered_img_q1 = contraharmonic_mean_filter(img, kernel_size=3, Q=1)
filtered_img_q_neg1 = contraharmonic_mean_filter(img, kernel_size=3, Q=-3)

# Display the images using matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Base Image')

plt.subplot(1, 3, 2)
plt.imshow(filtered_img_q1, cmap='gray')
plt.title('Q = 1')

# The Best
plt.subplot(1, 3, 3)
plt.imshow(filtered_img_q_neg1, cmap='gray')
plt.title('Q = -1')

plt.show()
