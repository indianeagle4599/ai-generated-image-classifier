import cv2
import numpy as np
import matplotlib.pyplot as plt


def dft_magnitude_pooling(image_path, pool_size=4):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform 2D DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate magnitude spectrum
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Perform pooling on magnitude spectrum
    h, w = magnitude_spectrum.shape
    pooled = cv2.resize(magnitude_spectrum, dsize=(w // pool_size, h // pool_size))

    # Upsample pooled magnitude to original size
    pooled_upsampled = cv2.resize(pooled, (w, h), interpolation=cv2.INTER_NEAREST)

    # Reconstruct the image from pooled magnitude
    pooled_dft = np.zeros_like(dft_shift)
    pooled_dft[:, :, 0] = pooled_upsampled * np.cos(
        np.angle(dft_shift[:, :, 0] + 1j * dft_shift[:, :, 1])
    )
    pooled_dft[:, :, 1] = pooled_upsampled * np.sin(
        np.angle(dft_shift[:, :, 0] + 1j * dft_shift[:, :, 1])
    )

    pooled_dft_shift = np.fft.ifftshift(pooled_dft)
    reconstructed = cv2.idft(pooled_dft_shift)
    reconstructed = cv2.magnitude(reconstructed[:, :, 0], reconstructed[:, :, 1])

    # Normalize the reconstructed image
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    return img, magnitude_spectrum, pooled_upsampled, reconstructed


def display_results(original, magnitude, pooled, reconstructed):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(np.log(1 + magnitude), cmap="gray")
    ax2.set_title("DFT Magnitude Spectrum")
    ax2.axis("off")

    ax3.imshow(np.log(1 + pooled), cmap="gray")
    ax3.set_title("Pooled DFT Magnitude")
    ax3.axis("off")

    ax4.imshow(reconstructed, cmap="gray")
    ax4.set_title("Reconstructed Image")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
image_path = "dataset/example/train/00a1c6c39f034eb689aa830470acbcb2.jpg"
original, magnitude, pooled, reconstructed = dft_magnitude_pooling(image_path)
display_results(original, magnitude, pooled, reconstructed)
