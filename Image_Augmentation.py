import os
import glob
import numpy as np
import cv2
#from imgaug import augmenters as iaa

# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, img_to_array, load_img
from tensorflow.keras import layers
from tensorflow.keras import Sequential 
#from keras.preprocessing.image import array_to_img, img_to_array, load_img

# Input and output directories
input_dir = r"F:\SEMESTER (4-1)\THESIS AND PROJECT\ALL_DATA\1804045"
output_dir = r"F:\SEMESTER (4-1)\THESIS AND PROJECT\augmented_all\1804045"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the preprocessing function
def preprocess_image(image):
    resized_image = cv2.resize(image, (50, 50))
    return resized_image

# Define the padding function
def pad_and_resize_image(image):
    padding_pixels = 7  # Number of pixels to add as padding

    # Create white padding using numpy
    padded_image = np.ones((image.shape[0] + padding_pixels*2, image.shape[1] + padding_pixels*2, 3), dtype=np.uint8) * 255
    padded_image[padding_pixels:-padding_pixels, padding_pixels:-padding_pixels, :] = image

    # Resize the padded image
    resized_image = cv2.resize(padded_image, (64,64))  #new image size = 50x50

    return resized_image


datagen = ImageDataGenerator(
        rotation_range = 15,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = False,
        vertical_flip=False,
        brightness_range = (0.1, 1.5))
# Load input images
image_files = glob.glob(os.path.join(input_dir, "*.png"))

for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)
     # Get the base filename
    base_filename = os.path.basename(image_file)
    #base_filename = os.path.splitext(base_filename)[0]
    #image = preprocess_image(image)
    image = pad_and_resize_image(image)
    
    # Apply random augmentations
    augmented_images = []
    # Converting the input sample image to an array
    x = img_to_array(image)
    
    # Reshaping the input image
    x = x.reshape((1, ) + x.shape) 
    

    i = 0
    for batch in datagen.flow(x, batch_size = 1,
                              save_to_dir = output_dir, 
                              save_prefix =f"{base_filename}_aug_{i}", save_format ='png'):
        i += 1
        if i > 4:
            break
