#################### 1st part ##############################
import cv2
import os

# Load the image
loc = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\BANGLA LETTERS\bangla-1904045_1.jpg'
image = cv2.imread(loc)
#image = cv2.resize(image, (0,0), fx = 0.35, fy = 0.5)
#cv2.imshow('Frame', image)

# Check if the image was successfully loaded
if image is None:
    print("Error: Unable to load the image file.")
    exit()

# Crop and save each row as a separate image
# Replace the coordinates with the actual values
start_row = 5
end_row = image.shape[0]
start_col = 5
end_col = 210
row = []
for i in range(10):
    row1 = image[start_row:end_row, start_col:end_col]
    start_col = 5 + end_col
    end_col = end_col + 215
    row.append(row1)

    #cv2.imshow(f"{i}", row1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
output_dir = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\CROPPED BANGLA\1804045'
os.makedirs(output_dir, exist_ok=True)
cell = []
for i in range(10):
    start_row = 2
    end_row = 212
    start_col= 2
    end_col = 200
    image = row[i]    
    for j in range(6):
        row2 = image[start_row:end_row, start_col:end_col]
        start_row = 2+end_row
        end_row = 215 + end_row
        
        output_path = os.path.join(output_dir, f"letter_{j+1+6*i}.png")
        cv2.imwrite(output_path, row2)
        #print("ok")


######################### 2nd part #################################

import cv2
import os

# Load the image
loc = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\BANGLA LETTERS\bangla-1904045_2.jpg'
image = cv2.imread(loc)
#image = cv2.resize(image, (0,0), fx = 0.35, fy = 0.5)
#cv2.imshow('Frame', image)

# Check if the image was successfully loaded
if image is None:
    print("Error: Unable to load the image file.")
    exit()

# Crop and save each row as a separate image
# Replace the coordinates with the actual values
start_row = 5
end_row = image.shape[0]
start_col = 0
end_col = 213
row = []
for i in range(10):
    row1 = image[start_row:end_row, start_col:end_col]
    start_col = 1 + end_col
    end_col = end_col + 216
    row.append(row1)

    cv2.imshow(f"{i}", row1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
output_dir = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\CROPPED BANGLA\1804045'
os.makedirs(output_dir, exist_ok=True)
cell = []
for i in range(10):
    start_row = 4
    end_row = 222
    start_col= 2
    end_col = 210
    image = row[i]    
    for j in range(6):
        row2 = image[start_row:end_row, start_col:end_col]
        start_row = 2+end_row
        end_row = 225 + end_row
        
        output_path = os.path.join(output_dir, f"letter_{60+j+1+6*i}.png")
        cv2.imwrite(output_path, row2)
        #print("ok")


############################### 3rd Part ###########################################
import cv2
import os

# Load the image
loc = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\BANGLA LETTERS\bangla-1904045_3.jpg'
image = cv2.imread(loc)
#image = cv2.resize(image, (0,0), fx = 0.35, fy = 0.5)
#cv2.imshow('Frame', image)

# Check if the image was successfully loaded
if image is None:
    print("Error: Unable to load the image file.")
    exit()

# Crop and save each row as a separate image
# Replace the coordinates with the actual values
start_row = 1
end_row = image.shape[0]
start_col = 0
end_col = 230
row = []
for i in range(10):
    row1 = image[start_row:end_row, start_col:end_col]
    start_col = 2 + end_col
    end_col = end_col + 232
    row.append(row1)

    #cv2.imshow(f"{i}", row1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
output_dir = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\CROPPED BANGLA\1804045'
os.makedirs(output_dir, exist_ok=True)
cell = []
for i in range(10):
    start_row = 4
    end_row = 226
    start_col= 2
    end_col = 220
    image = row[i]    
    for j in range(3):
        row2 = image[start_row:end_row, start_col:end_col]
        start_row = 2+end_row
        end_row = 228 + end_row
        
        output_path = os.path.join(output_dir, f"letter_{120+j+1+3*i}.png")
        cv2.imwrite(output_path, row2)
        #print("ok")


######################### Applying Harrish Corner Detection to get only the letter part ##################################################
# Importing all the modules
import cv2
import numpy as np
import os

# Creating a function to crop the letters
def letter_crop(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Converting Image into Gray Scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 48, 3, 0.04)
    dst = cv2.dilate(dst, None)
    threshold = 0.01 * dst.max()
    #Finding the corners
    corner_image = np.zeros_like(dst)
    corner_image[dst > threshold] = 255
    # Creating a contour path based on the corners
    contours, _ = cv2.findContours(corner_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # Finding the corners
    x, y, w, h = cv2.boundingRect(largest_contour)
    #Cropping the letter
    cropped_image = image[y:y+h+3, x:x+w+7]
    return cropped_image



location = r"F:\SEMESTER (4-1)\THESIS AND PROJECT\CROPPED BANGLA\1804045"  #folder where the images are stored
output_dir = r"F:\SEMESTER (4-1)\THESIS AND PROJECT\HARRISH DETECTED BANGLA LETTERS\1804045" #folder to where the cropped image will be stored
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(location):
    file_path = os.path.join(location, filename)
    image_path = file_path
    if os.path.isfile(file_path):
        letter = letter_crop(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, letter)
