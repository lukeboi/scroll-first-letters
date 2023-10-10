# https://chat.openai.com/c/06a0b4eb-7e31-4104-b56d-194d0a02f470
from PIL import Image
import numpy as np

# Step 1: List of filenames (Edit this list to include your greyscale image filenames)
image_filenames = [
    "tmp/20230827161846/5_max/max_image_20.png",
    "tmp/20230827161846/5_max/max_image_25.png",
    "tmp/20230827161846/5_max/max_image_30.png",
    "tmp/20230827161846/5_max/max_image_35.png",
    "tmp/20230827161846/5_max/max_image_40.png",
    "tmp/20230827161846/5_max/max_image_45.png",
    # Add more filenames here
]

# Check if there are any files to process
if len(image_filenames) == 0:
    print("No images found.")
    exit()

# Step 2: Load the first image to get dimensions and initialize max_image array
first_image = Image.open(image_filenames[0]).convert("L")
image_shape = np.array(first_image).shape
max_image = np.array(first_image, dtype=np.uint8)

# Step 3: Compute the maximum pixel value at each location
for filename in image_filenames[1:]:
    img = Image.open(filename).convert("L")
    img_array = np.array(img)
    if img_array.shape != image_shape:
        print(f"Skipping image {filename} due to different dimensions.")
        continue
    max_image = np.maximum(max_image, img_array)

# Convert grayscale max_image to RGB format
# We expand the dimensions of max_image from (H, W) to (H, W, 3) and populate all 3 channels with max_image.
max_image_rgb = np.stack([max_image]*3, axis=2)

# Step 4: Save the resulting image in RGB format
result_image = Image.fromarray(max_image_rgb.astype('uint8'), 'RGB')
result_image.save("custom_max_image.png")

print("Max image saved as custom_max_image.png")
