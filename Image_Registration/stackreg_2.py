
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
image_folder = r'C:\Users\hp\OneDrive\Desktop\data\div-images\train'
im_folder2= ""
image1_path = os.path.join(image_folder, '004.png')
image1 = io.imread(image1_path, as_gray=True).astype(np.uint8)


# print("Image1 shape:", image1.shape)
# print("Image1 content:", image1)


# Reading images frm folder
image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png')])

# Create a list to store images

image_list = [io.imread(filename, as_gray=True).astype(np.uint8) for filename in image_files]


# Print the dimensions of the converted images
# for i, image_gray in enumerate(image_list):
#     print(f"Converted Image {i + 1} dimensions: {image_gray.shape}")

#Creating img stack

output_path = r'C:\Users\hp\OneDrive\Desktop\Pyhton\images\out5_a.png'
imageio.mimwrite(output_path, image_list, duration=0.1)
#out directory
output_aligned_directory = r'C:\Users\hp\OneDrive\Desktop\out5_a'
output_aligned_directory1 = r'C:\Users\hp\OneDrive\Desktop\out6'

os.makedirs(output_aligned_directory, exist_ok=True)

img_stack = io.imread(output_path)
# Initialize StackReg
sr = StackReg(StackReg.RIGID_BODY)


#Registering each img n storing it in stack
out_previous = sr.register_transform_stack(img_stack, reference='previous')

out_previous_uint8 = (out_previous * 255).astype(np.uint8)

# Loading ref img
ref_img_path = r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\004.png"
ref_img = io.imread(ref_img_path)

for i, aligned_frame in enumerate(out_previous_uint8):
    output_filename = os.path.join(output_aligned_directory, f'aligned_frame_{i}.png')
    imageio.imwrite(output_filename, aligned_frame)


def iou(image1, image2):
    # Convert images to boolean
    binary_image1 = (image1 > 0).astype(int)
    binary_image2 = (image2 > 0).astype(int)

    # Calculate IoU without using logical operations
    intersection = np.sum(binary_image1 * binary_image2)
    union = np.sum(binary_image1 + binary_image2 - binary_image1 * binary_image2)

    if union == 0:
        return 0  # Avoid division by zero

    iou_value = intersection / union
    return iou_value

image1 = io.imread(r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\004.png", as_gray=True)
image2 = io.imread(r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\064.png", as_gray=True)

def sad(image1, image2):

    # Convert images to boolean
    binary_image1 = (image1 > 0).astype(int)
    binary_image2 = (image2 > 0).astype(int)

    # Calculate Sum of Absolute Differences (SAD)
    sad_value = np.sum(np.abs(binary_image1 - binary_image2))

    return sad_value

def mse(image1, image2):
    # Ensure the images have the same shape
    image1 = io.imread(r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\004.png", as_gray=True)
    image2 = io.imread(r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\064.png", as_gray=True)

    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    image1 = image1[:min_height, :min_width]
    image2 = image2[:min_height, :min_width]

    # Calculate MSE
    mse_value = np.mean((image1 - image2)**2)
    return mse_value

def print_evaluation_metrics1a(tversky_value, sad, euclidean_distance, mse_value):
    
    image1 = io.imread(r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\004.png", as_gray=True)
    image2 = io.imread(r"C:\Users\hp\OneDrive\Desktop\data\div-images\train\064.png", as_gray=True)
    iou_value = iou(image1, image2)
    
    print("------------------Dataset-2------------------")
    print("-----StackReg------")
    print("Intersection over Union (IoU):", iou_value)
    print("Sad  is", sad)
    print("Euclidean measure is", euclidean_distance)
    print("MSE is", mse_value)
        
def plot_images1a(reference_img, registered_img, mean_img):
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
threshold = 0.5  # Adjust the threshold as needed
binary_mean = (mean > threshold).astype(np.uint8)
binary_ref_img = (ref_img > threshold).astype(np.uint8)

#overlap measure
iou_value_ = iou(binary_mean, binary_ref_img)
# euclidean measure
euclidean_dist = np.linalg.norm((binary_mean - binary_ref_img)/100)
#Jaccard Index
sad_ = sad(binary_mean, binary_ref_img)
#MSE
mse_val = mse(binary_mean, binary_ref_img)

print_evaluation_metrics1a(iou_value_, sad_, euclidean_dist, mse_val)
plot_images1a(ref_img, out_previous_uint8[0], out_previous_uint8.mean(axis=0))