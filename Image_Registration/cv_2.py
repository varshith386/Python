#Out 6_a ---->>> CV2 Dataset-2

import os
import cv2
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time


start_time = time.time()
def create_transform_matrix(displacement):
    transform_matrix = np.eye(3)

    if displacement.ndim == 1:  # Handle a single displacement
        transform_matrix[0, 2] = displacement[0]
        transform_matrix[1, 2] = displacement[1]
    elif displacement.ndim == 2:  # Take the mean of multiple displacements
        mean_displacement = np.mean(displacement, axis=0)
        transform_matrix[0, 2] = mean_displacement[0]
        transform_matrix[1, 2] = mean_displacement[1]

    return transform_matrix

# Loading images
image_folder = r'C:\Users\hp\OneDrive\Desktop\College\sip_endsem\data\div-images\train'
im_folder2= ""
image1_path = os.path.join(image_folder, '004.png')
image1 = io.imread(image1_path, as_gray=True).astype(np.uint8)
image_1_path = r"C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out6_a\aligned_frame_0_optical_flow.png"
image_2_path = r"C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out6_a\aligned_frame_41_optical_flow.png"
image_1 = io.imread(image_1_path, as_gray=True).astype(np.uint8)
image_2 = io.imread(image_2_path, as_gray=True).astype(np.uint8)


image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')])

# Create a list to store images
image_list = [io.imread(filename) for filename in image_files]

# Output directory
output_path = r'C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out6_a.png'
imageio.mimwrite(output_path, image_list, duration=0.1)
#out directory
output_aligned_directory = r'C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out6_a'


# Use the first image as the reference
reference_frame = image_list[0]

# Initialize the reference frame as grayscale if it's not already
if len(reference_frame.shape) == 3:
    reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
else:
    reference_frame_gray = reference_frame

# Plot the reference image
plt.subplot(1, 3, 1)
plt.imshow(reference_frame_gray, cmap='gray')
plt.title('Reference Image')

# List to store registered images
registered_images = []

# Find feature points in the reference image
reference_points = cv2.goodFeaturesToTrack(reference_frame_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

for i, moving_frame in enumerate(image_list[1:]):
    # Use the moving frame directly if it's a single-channel image
    moving_gray = moving_frame if len(moving_frame.shape) == 2 else cv2.cvtColor(moving_frame, cv2.COLOR_BGR2GRAY)

    # Find feature points in the moving frame
    moving_points, status, error = cv2.calcOpticalFlowPyrLK(reference_frame_gray, moving_gray, reference_points, None)

    # Extract the displacement values from the flow
    displacement = moving_points[status == 1] - reference_points[status == 1]

    # Create an affine transformation matrix from the displacement
    transform_matrix = create_transform_matrix(displacement)

    # Warp the moving frame using the affine transformation
    displaced_frame = cv2.warpAffine(moving_frame, transform_matrix[:2, :], (moving_frame.shape[1], moving_frame.shape[0]))

    # Append the registered image to the list
    registered_images.append(displaced_frame)
    
    # Save the registered image
    output_filename = os.path.join(output_aligned_directory, f'aligned_frame_{i}_optical_flow.png')
    imageio.imwrite(output_filename, displaced_frame)
    
    registered_images.append(displaced_frame)

    # Update feature points for the next iteration
    reference_points = moving_points

# Calculate and plot the mean of registered images
mean_registered_image = np.mean(registered_images, axis=0)


def mae(img1, img2):
    
    gray_image1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray_image2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # Ensure both images have the same shape for SSIM calculation
    if gray_image1.shape != gray_image2.shape:
        # Resize images to a common shape
        common_shape = (max(gray_image1.shape[0], gray_image2.shape[0]),
                        max(gray_image1.shape[1], gray_image2.shape[1]))

        gray_image1 = cv2.resize(gray_image1, common_shape, interpolation=cv2.INTER_NEAREST)
        gray_image2 = cv2.resize(gray_image2, common_shape, interpolation=cv2.INTER_NEAREST)

# Read images using OpenCV
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image_2_path)
    # Ensure the images have the same shape
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    img1 = img1[:min_height, :min_width]
    img2 = img2[:min_height, :min_width]

    # Calculate MAE
    mae_value = np.mean(np.abs(img1 - img2))
    return mae_value
# Assuming you have two images: image1 and imag

# Calculate MAE
# mae_value = mae(image_1, image_2)

# Print the MAE
# print("Mean Absolute Error (MAE) is:", mae_value)


def mse(img1, img2):
    
    gray_image1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray_image2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # Ensure both images have the same shape for SSIM calculation
    if gray_image1.shape != gray_image2.shape:
        # Resize images to a common shape
        common_shape = (max(gray_image1.shape[0], gray_image2.shape[0]),
                        max(gray_image1.shape[1], gray_image2.shape[1]))

        gray_image1 = cv2.resize(gray_image1, common_shape, interpolation=cv2.INTER_NEAREST)
        gray_image2 = cv2.resize(gray_image2, common_shape, interpolation=cv2.INTER_NEAREST)

    # Ensure the images have the same shape
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    img1 = img1[:min_height, :min_width]
    img2 = img2[:min_height, :min_width]

    # Calculate MSE
    mse_value = np.mean((img1 - img2)**2)
    return mse_value



def calculate_ssim(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    gray_image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

    # Ensure both images have the same shape for SSIM calculation
    if gray_image1.shape != gray_image2.shape:
        # Resize images to a common shape
        common_shape = (max(gray_image1.shape[0], gray_image2.shape[0]),
                        max(gray_image1.shape[1], gray_image2.shape[1]))

        gray_image1 = cv2.resize(gray_image1, common_shape, interpolation=cv2.INTER_NEAREST)
        gray_image2 = cv2.resize(gray_image2, common_shape, interpolation=cv2.INTER_NEAREST)

    # Calculate SSIM
    ssim_value, _ = ssim(gray_image1, gray_image2, full=True)

    return ssim_value

# Calculate SSIM using the entire sequence
# ssim_value = calculate_ssim(binary_mean, binary_ref_img)
# print("SSIM measure is", ssim_value)

def print_evaluation_metrics2a(mae, mse, euclidean_distance, calculate_ssim):
    print("-----CV2-----")
    print("Mae  is", mae)
    print("Mse is", mse)
    print("Euclidean measure is", euclidean_distance)
    print("Ssim is", calculate_ssim)
    
def plot_images2a(reference_img, registered_img, mean_img):
    # Plot the reference image
    plt.subplot(1, 3, 1)
    plt.imshow(reference_img, cmap='gray')
    plt.title('Reference Image')

    # Plot the registered image
    plt.subplot(1, 3, 2)
    registered_imgpath=r"C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out6_a\aligned_frame_41_optical_flow.png"
    registered_img = io.imread(registered_imgpath, as_gray=True).astype(np.uint8)
    plt.imshow(registered_img, cmap='gray')
    plt.title('Registered Image')

    # Plot the mean of registered images
    plt.subplot(1, 3, 3)
    plt.imshow(mean_img, cmap='gray')
    plt.title('Mean of Registered Images')

    plt.suptitle("cv2")
    # Display the plots
    plt.show()

threshold = 0.5  # Adjust the threshold as needed
binary_mean = (mean_registered_image > threshold).astype(np.uint8)
binary_ref_img = (reference_frame_gray > threshold).astype(np.uint8)

# mae_value = mae(image_1, image_2)
#Overlap measure
calculate_ssim_ = calculate_ssim(binary_mean, binary_ref_img)

# JaccardIndex
mae_ = mae(binary_mean, binary_ref_img)

# Euclidean measure
euclidean_distance_msr = np.linalg.norm((image_1 - image_2) / 100)

# MSE
mse_value_msr = mse(image_1, image_2)

# Call the function to print the metrics
print_evaluation_metrics2a(mae_, mse_value_msr, euclidean_distance_msr, calculate_ssim_)

# Call the plot function after processing all frames
plot_images2a(reference_frame_gray, displaced_frame, mean_registered_image)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to run the code: {elapsed_time} seconds")
