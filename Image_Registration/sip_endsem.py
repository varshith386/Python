#Out 5 ---->>>   StackReg

#For few images
from pystackreg import StackReg
from skimage import io
from matplotlib import pyplot as plt
import imageio
import numpy as np
import os
from skimage.transform import resize

# Loading images to train model
image_folder = r'C:\Users\hp\OneDrive\Desktop\train'
im_folder2= ""

# Reading images frm folder
image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')])

# Create a list to store images

image_list = [io.imread(filename) for filename in image_files]

# for i, image in enumerate(image_list):
#     print(f"Image {i + 1} dimensions: {image.shape}")

#Creating img stack

output_path = r'C:\Users\hp\OneDrive\Desktop\Pyhton\images\out5.png'
imageio.mimwrite(output_path, image_list, duration=0.1)

#out directory
output_aligned_directory = r'C:\Users\hp\OneDrive\Desktop\out5'
output_aligned_directory1 = r'C:\Users\hp\OneDrive\Desktop\out6'

os.makedirs(output_aligned_directory, exist_ok=True)

# Load the entire stack
img_stack = io.imread(output_path)

# Initialize StackReg
sr = StackReg(StackReg.RIGID_BODY)

#Registering each img n storing it in stack
out_previous = sr.register_transform_stack(img_stack, reference='previous')

out_previous_uint8 = (out_previous * 255).astype(np.uint8)

# Loading ref img
ref_img_path = r"C:\Users\hp\OneDrive\Desktop\train\ct93.png"
ref_img = io.imread(ref_img_path)

for i, aligned_frame in enumerate(out_previous_uint8):
    output_filename = os.path.join(output_aligned_directory, f'aligned_frame_{i}.png')
    imageio.imwrite(output_filename, aligned_frame)


def overlap_measure(image1, image2):
    intersection = np.sum(image1 & image2)
    union = np.sum(image1 | image2)
    
    if union == 0:
        return 0  
    
    overlap_coefficient = 2 * intersection / union
    return overlap_coefficient


def jaccard_index(image1, image2):
    intersection = np.sum(image1 & image2)
    union = np.sum(image1 | image2)
    
    if union == 0:
        return 0  
    
    jaccard = intersection / union
    return jaccard

def resize_images(image1, image2):
    common_size = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))


    resized_image1 = resize(image1, common_size, anti_aliasing=True)
    resized_image2 = resize(image2, common_size, anti_aliasing=True)

    return resized_image1, resized_image2


def mse(image1, image2):

    image1 = io.imread(r'C:\Users\hp\OneDrive\Desktop\train\ct93.png', as_gray=True)
    image2 = io.imread(r'C:\Users\hp\OneDrive\Desktop\train\ct937.png', as_gray=True)

    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    image1 = image1[:min_height, :min_width]
    image2 = image2[:min_height, :min_width]

    # Calculate MSE
    mse_value = np.mean((image1 - image2)**2)
    return mse_value


def print_evaluation_metrics1(overlap_measure, jaccard_index, euclidean_distance, mse_value):
    print("-----StackReg------")
    print("Overlap measure is", overlap_measure)
    print("Jaccard Index is", jaccard_index)
    print("Euclidean measure is", euclidean_distance)
    print("MSE is", mse_value)
    
    
def plot_images1(reference_img, registered_img, mean_img):
    # Plot the reference image
    plt.subplot(1, 3, 1)
    plt.imshow(ref_img, cmap='gray')
    plt.title('Reference Image')

    # Plot the registered image
    plt.subplot(1, 3, 2)
    plt.imshow(out_previous_uint8[0], cmap='gray')
    plt.title('Registered Image')

    # Plot the mean of registered images
    plt.subplot(1, 3, 3)
    plt.imshow(out_previous_uint8.mean(axis=0), cmap='gray')
    plt.title('Mean of Registered Images')

    plt.suptitle("Stackreg")
    # Display the plots
    plt.show()

mean=np.mean(out_previous) 
threshold = 0.5 
binary_mean = (mean > threshold).astype(np.uint8)
binary_ref_img = (ref_img > threshold).astype(np.uint8)

#overlap measure
overlap_measure_ = overlap_measure(binary_mean, binary_ref_img)
# euclidean measure
euclidean_dist = np.linalg.norm((binary_mean - binary_ref_img)/100)
#Jaccard Index
jaccard_index_value = jaccard_index(binary_mean, binary_ref_img)
#MSE
mse_val = mse(binary_mean, binary_ref_img)

print_evaluation_metrics1(overlap_measure_, jaccard_index_value, euclidean_dist, mse_val)
plot_images1(ref_img, out_previous_uint8[0], out_previous_uint8.mean(axis=0))