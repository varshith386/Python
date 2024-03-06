#Out7_a ---->>>FFT  Dataset-2

import numpy as np
from skimage import io
import imageio
import os
import matplotlib.pyplot as plt

import time


start_time = time.time()

def fft_registration(fixed_image, moving_image, central_weight=2.0):
    # Compute the cross-correlation in the frequency domain using FFT
    fft_fixed = np.fft.fft2(fixed_image)
    fft_moving = np.fft.fft2(moving_image)
    cross_correlation = np.fft.fftshift(np.fft.ifft2(fft_fixed * np.conj(fft_moving)))

    # Apply a Gaussian window to prioritize the central region
    rows, cols = cross_correlation.shape[0], cross_correlation.shape[1]
    r_center, c_center = rows // 2, cols // 2
    gauss_window = np.exp(
        -((np.arange(rows) - r_center) ** 2 + (np.arange(cols) - c_center) ** 2) /
        (2 * (central_weight * min(r_center, c_center)) ** 2)
    )
    cross_correlation *= gauss_window[:, np.newaxis]

    # Find the translation by locating the peak in the weighted cross-correlation
    translation = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)

    # Apply the translation to the moving image
    aligned_image = np.roll(moving_image, shift=(translation[0] - fixed_image.shape[0] // 2,
                                                 translation[1] - fixed_image.shape[1] // 2),
                            axis=(0, 1))

    return aligned_image


def register_images(fixed_image, moving_images, output_directory, central_weights):
    aligned_images = []
    for i, moving_image in enumerate(moving_images):
        for central_weight in central_weights:
            aligned_image = fft_registration(fixed_image, moving_image, central_weight)

            # Save the aligned image
            output_filename = os.path.join(output_directory, f'aligned_frame_{i}_weight_{central_weight}.png')
            imageio.imwrite(output_filename, aligned_image)

            # Append to the list of aligned images
            aligned_images.append(aligned_image)

    return aligned_images

# Load images
image_folder = r'C:\Users\hp\OneDrive\Desktop\College\sip_endsem\data\div-images\train'
output_aligned_directory1 = r'C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out7_a'

image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')])

# Create a list to store images
image_list = [io.imread(filename) for filename in image_files]

# Convert to NumPy arrays
numpy_image_list = [np.array(image) for image in image_list]

# Fixed image initialization
fixed_image = numpy_image_list[0]  # Fixed
moving_images = numpy_image_list[1:]  # Moving

# Define central weights for prioritizing the central region during registration
central_weights = [1.0, 2.0, 3.0]  # Add more weights as needed

# Apply FFT with central prioritization on the fixed image multiple times
aligned_images = register_images(fixed_image, moving_images, output_aligned_directory1, central_weights)



def overlap_measure(im1, im2):
    intersection = np.sum(im1 & im2)
    union = np.sum(im1 | im2)
    
    if union == 0:
        return 0  # Avoid division by zero
    
    overlap_coefficient = 2 * intersection / union
    return overlap_coefficient


def jaccard_index(im1, im2):
    intersection = np.sum(im1 & im2)
    union = np.sum(im1 | im2)
    
    if union == 0:
        return 0  # Avoid division by zero
    
    jaccard = intersection / union
    return jaccard

def mse(im1, im2):
    # Ensure the images have the same shape
    min_height = min(im1.shape[0], im2.shape[0])
    min_width = min(im1.shape[1], im2.shape[1])

    im1 = im1[:min_height, :min_width]
    im2 = im2[:min_height, :min_width]

    # Calculate MSE
    mse_value = np.mean((im1 - im2)**2)
    return mse_value

def print_evaluation_metrics3a(overlap_measure, jaccard_index, euclidean_distance, mse_value):
    print("-----FFT-----")
    print("Overlap measure is", overlap_measure)
    print("Jaccard Index is", jaccard_index)
    print("Euclidean measure is", euclidean_distance)
    print("MSE is", mse_value)

def plot_images3a(fixed_image, aligned_images, mean_aligned_image, weight):
    # Plot the reference image
    plt.subplot(1, 3, 1)
    plt.imshow(fixed_image, cmap='gray')
    plt.title('Reference Image')

    # Plot the registered imagec
    plt.subplot(1, 3, 2)
    plt.imshow(aligned_images, cmap='gray')
    plt.title('Registered Image')

    # Plot the mean of registered images
    plt.subplot(1, 3, 3)
    plt.imshow(mean_aligned_image, cmap='gray')
    plt.title('Mean of Registered Images')

    plt.suptitle("FFT")
    # Display the plots
    plt.show()

mean_aligned_image = np.mean(aligned_images, axis=0)
threshold = 0.5  # Adjust the threshold as needed
binary_mean = (mean_aligned_image > threshold).astype(np.uint8)
binary_ref_img = (fixed_image > threshold).astype(np.uint8)

#Overlap measure
o_measure = overlap_measure(binary_mean, binary_ref_img)
#JaccardIndex
J_measure = jaccard_index(binary_mean, binary_ref_img)
# euclidean measure
euclidean_distance = np.linalg.norm((binary_mean - binary_ref_img)/100)
#mse
mse_value = mse(binary_mean, binary_ref_img)


print_evaluation_metrics3a(o_measure, J_measure, euclidean_distance, mse_value)
plot_images3a(fixed_image, aligned_images[1], mean_aligned_image, central_weights[2])

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to run the code: {elapsed_time} seconds")
