import cv2
import numpy as np


def dft_magnitude_spectrum(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute DFT
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

    return magnitude_spectrum


def dct_features(image, block_size=8):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute DCT
    dct = cv2.dct(np.float32(image))

    # Extract features from DCT coefficients
    features = dct[:block_size, :block_size].flatten()

    return features
