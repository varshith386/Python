#Out 6 ---->>> CV2

import os
import cv2
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt

import time


start_time = time.time()

def create_transform_matrix(displacement):
    transform_matrix = np.eye(3)

    if displacement.ndim == 1:
        transform_matrix[0, 2] = displacement[0]
        transform_matrix[1, 2] = displacement[1]
    elif displacement.ndim == 2: 
        mean_displacement = np.mean(displacement, axis=0)
        transform_matrix[0, 2] = mean_displacement[0]
        transform_matrix[1, 2] = mean_displacement[1]

    return transform_matrix

# Loading images
image_folder = r'C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Input\train'
image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')])

# Create a list to store images
image_list = [io.imread(filename) for filename in image_files]

# Output directory
output_aligned_directory = r"C:\Users\hp\OneDrive\Desktop\College\sip_endsem\Output\out6"
os.makedirs(output_aligned_directory, exist_ok=True)


reference_frame = image_list[0]


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
    

    output_filename = os.path.join(output_aligned_directory, f'aligned_frame_{i}_optical_flow.png')
    imageio.imwrite(output_filename, displaced_frame)

 
    reference_points = moving_points

#Mean of reg img
mean_registered_image = np.mean(registered_images, axis=0)



def jaccard_index(img1, img2):
    intersection = np.sum(img1 & img2)
    union = np.sum(img1 | img2)
    
    if union == 0:
        return 0 
    
    jaccard = intersection / union
    return jaccard

def mse(img1, img2):

    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    img1 = img1[:min_height, :min_width]
    img2 = img2[:min_height, :min_width]

    mse_value = np.mean((img1 - img2)**2)
    return mse_value

def overlap_measure(img1, img2):
    intersection = np.sum(img1 & img2)
    union = np.sum(img1 | img2)
    
    if union == 0:
        return 0 
    
    overlap_coefficient = 2 * intersection / union
    return overlap_coefficient


def print_evaluation_metrics2(overlap_measure, jaccard_index, euclidean_distance, mse_value):
    print("-----CV2-----")
    print("Overlap measure is", overlap_measure)
    print("Jaccard Index is", jaccard_index)
    print("Euclidean measure is", euclidean_distance)
    print("MSE is", mse_value)
    
def plot_images2(reference_img, registered_img, mean_img):
    # Plot the reference image
    plt.subplot(1, 3, 1)
    plt.imshow(reference_img, cmap='gray')
    plt.title('Reference Image')

    # Plot the registered image
    plt.subplot(1, 3, 2)
    plt.imshow(registered_img, cmap='gray')
    plt.title('Registered Image')

    # Plot the mean of registered images
    plt.subplot(1, 3, 3)
    plt.imshow(mean_img, cmap='gray')
    plt.title('Mean of Registered Images')

    plt.suptitle("cv2")

    plt.show()

threshold = 0.5 
binary_mean = (mean_registered_image > threshold).astype(np.uint8)
binary_ref_img = (reference_frame_gray > threshold).astype(np.uint8)


#Overlap measure
o_msr = overlap_measure(binary_mean, binary_ref_img)

# JaccardIndex
J_msr = jaccard_index(binary_mean, binary_ref_img)

# Euclidean measure
euclidean_distance_msr = np.linalg.norm((binary_mean - binary_ref_img) / 100)

# MSE
mse_value_msr = mse(binary_mean, binary_ref_img)

# Call the function to print the metrics
print_evaluation_metrics2(o_msr, J_msr, euclidean_distance_msr, mse_value_msr)

# Call the plot function after processing all frames
plot_images2(reference_frame_gray, displaced_frame, mean_registered_image)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to run the code: {elapsed_time} seconds")
