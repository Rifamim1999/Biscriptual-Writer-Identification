import cv2
import os

# Load the image
loc = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\SCAN\84.jpg'
image = cv2.imread(loc)
#image = cv2.resize(image, (0,0), fx = 0.35, fy = 0.5)
cv2.imshow('Frame', image)

# Check if the image was successfully loaded
if image is None:
    print("Error: Unable to load the image file.")
    exit()

# Crop and save each row as a separate image
# Replace the coordinates with the actual values
start_row = 1
end_row = 1423
start_col = 0
end_col = 309
row = []
for i in range(3):
    row1 = image[start_row:end_row, start_col:end_col]
    start_col = 1 + end_col
    end_col = end_col + 309
    row.append(row1)

    cv2.imshow(f"{i}", row1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
output_dir = r'F:\SEMESTER (4-1)\THESIS AND PROJECT\DATA COLLECTION\folder_name'
os.makedirs(output_dir, exist_ok=True)
cell = []
for i in range(3):
    start_row = 4
    end_row = 61
    start_col= 0
    end_col = 314
    image = row[i]    
    for j in range(26):
        row2 = image[start_row:end_row, start_col:end_col]
        start_row = 2+end_row
        end_row = 54 + end_row
        
        output_path = os.path.join(output_dir, f"letter_{j+1+26*i}.png")
        cv2.imwrite(output_path, row2)
        print("ok")
