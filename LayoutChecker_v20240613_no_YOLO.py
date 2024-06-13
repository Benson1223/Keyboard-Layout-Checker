#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from pdf2image import convert_from_path
import glob
import os
import cv2
import numpy as np
import re
from tensorflow import keras


# In[3]:


# Get the current working directory
Main_folder_path = os.getcwd()

# Define the folder paths dictionary
PDF_Folder_Paths = {}
PNG_Folder_Paths = {}
KEY_Folder_Paths = {}

# Function to search for a folder matching the given patterns and update the dictionary
def find_and_update_folder(folder_key, *patterns):
    # Use glob to find folders that match the given patterns
    matches = glob.glob(os.path.join(Main_folder_path, *patterns))
    
    if matches:
        # If a matching folder is found, update the dictionary with the folder path
        PDF_Folder_Paths[folder_key] = matches[0]

# Search for Design folder
find_and_update_folder('Design', "Design")

# Search for Reference folder
find_and_update_folder('Reference', "Reference")

# Search for PDK folder
find_and_update_folder('PDK', "PDK")

# Print the result for each folder
if 'Design' in PDF_Folder_Paths:
    print("Design folder path processing complete.")
else:
    print("Design folder not found.")

if 'Reference' in PDF_Folder_Paths:
    print("Reference folder path processing complete.")
else:
    print("Reference folder not found.")

if 'PDK' in PDF_Folder_Paths:
    print("PDK folder path processing complete.")
else:
    print("PDK folder not found.")


# In[5]:


def generate_keys_from_cropped_image(cropped_image_path):
    # Load the binary image using OpenCV
    image_cv2 = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image_cv2, 180, 255, cv2.THRESH_BINARY)###ORI 127
    
    # Find contours in the binary image using OpenCV
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an index to number the saved KEY images
    index = 1

    # Prepare the KEY images folder path
    KEY_Folder_name = os.path.splitext(os.path.basename(cropped_image_path))[0] + " Key"
    KEY_Folder_Path = os.path.join(os.path.dirname(cropped_image_path), KEY_Folder_name)
    if not os.path.exists(KEY_Folder_Path):
        os.makedirs(KEY_Folder_Path)

    # Create a list to store the valid contours
    valid_contours = []

    # Iterate over each contour and create keys
    for i, contour in enumerate(contours):
        
        # Find the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the KEY image from the original image
        key = image_cv2[y:y+h, x:x+w]
        
        # Ignore contours with unwanted shape
        if key.shape[0] < 100 or key.shape[1] < 100 or key.shape[0] > 1500 or key.shape[1] > 1500:
            continue
        
        # Calculate the total number of pixels in the contour
        total_pixels = key.shape[0] * key.shape[1]

        # Ignore contours with a small or large number of pixels
        if total_pixels < 60000 or total_pixels > 900000:
            continue
        
        # Check for overlap with other valid contours
        overlap = False
        
        for valid_contour in valid_contours:
            if (x >= valid_contour[0] and y >= valid_contour[1] and x + w <= valid_contour[0] + valid_contour[2] and y + h <= valid_contour[1] + valid_contour[3]):
                overlap = True
                break
        
        # If no overlap, save the contour and image
        if not overlap:
            valid_contours.append((x, y, w, h))
            # Generate the filename using the index
            filename = f"key_{index}.png"
            # Save the cropped KEY image as PNG in the KEY folder
            cv2.imwrite(os.path.join(KEY_Folder_Path, filename), key)
            # Increment the index for the next KEY image
            index += 1

    return KEY_Folder_Path


# In[6]:


# Function to adjust crop coordinates to stay within image boundaries
def adjust_crop_coordinates(image_width, image_height, crop_coords):
    crop_left, crop_upper, crop_right, crop_lower = crop_coords
    crop_left = max(0, crop_left)
    crop_upper = max(0, crop_upper)
    crop_right = min(image_width, crop_right)
    crop_lower = min(image_height, crop_lower)
    return crop_left, crop_upper, crop_right, crop_lower

# Function to create PNG and KEY images from a PDF file (both 1 page and 2 pages)
def create_images_from_pdf(pdf_file, file_name, PNG_Folder_Path):
    Image.MAX_IMAGE_PIXELS = None
    # Convert the first and second page of the PDF to images with specified settings
    #images = convert_from_path(pdf_file, dpi=600, first_page=1, last_page=2)
    images = convert_from_path(pdf_file, dpi=600, first_page=1, last_page=2, poppler_path=os.path.join(os.getcwd(),'bin'))
    
    if len(images) == 1:
        # Get the image dimensions
        image_width, image_height = images[0].size

        # Convert the first image to a binary image using thresholding
        image_cv = np.array(images[0])
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(image_cv, 180, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image using OpenCV
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize a variable to hold the contour with the specified area range
        contour_p1 = None

        # Iterate over each contour
        for contour in contours:
            # Find the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the KEY image from the original image
            key = binary_image[y:y+h, x:x+w]

            # Calculate the total number of pixels in the contour
            total_pixels = key.shape[0] * key.shape[1]

            # Check if the area is within the specified range
            if 20000000 < total_pixels < 40000000:
                contour_p1 = contour
                break

        if contour_p1 is not None:
            # Find the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour_p1)

            # Crop the image from the original image based on the contour
            image_p1 = images[0].crop((x, y, x+w, y+h)) #ORI no 100

            # Save the image in the PNG folder
            PNG_File_Name_p1 = f"{file_name}_p1.png"
            image_p1_path = os.path.join(PNG_Folder_Path, PNG_File_Name_p1)
            image_p1.save(image_p1_path)

            # Use the cropped image to generate key images
            key_folder_path = generate_keys_from_cropped_image(image_p1_path)
            PNG_Folder_Paths[file_name] = PNG_Folder_Path
            KEY_Folder_Paths[os.path.basename(key_folder_path)] = key_folder_path

        else:
            # Crop coordinates for the first image (left, upper, right, lower)
            crop_coords_p1 = (500, 2000, 11000, 5000) #ORI(500, 2000, 11000, 5000)
            # Adjust crop coordinates to stay within image boundaries
            crop_coords_p1 = adjust_crop_coordinates(image_width, image_height, crop_coords_p1)

            # Crop the first image (image_p1) to focus on the relevant content
            image_p1 = images[0].crop(crop_coords_p1)

            # Save the first image (image_p1) in the PNG folder
            PNG_File_Name_p1 = f"{file_name}_p1.png"
            image_p1_path = os.path.join(PNG_Folder_Path, PNG_File_Name_p1)
            image_p1.save(image_p1_path)

            # Use the cropped image to generate key images
            key_folder_path = generate_keys_from_cropped_image(image_p1_path)
            PNG_Folder_Paths[file_name] = PNG_Folder_Path
            KEY_Folder_Paths[os.path.basename(key_folder_path)] = key_folder_path
            
    else:
        # Get the image dimensions for the first and second images
        image_width_p1, image_height_p1 = images[0].size
        image_width_p2, image_height_p2 = images[1].size
        
        # Convert the first image to a binary image using thresholding
        image_cv_p1 = np.array(images[0])
        image_cv_p1 = cv2.cvtColor(image_cv_p1, cv2.COLOR_BGR2GRAY)
        _, binary_image_p1 = cv2.threshold(image_cv_p1, 180, 255, cv2.THRESH_BINARY)
        
        # Convert the second image to a binary image using thresholding
        image_cv_p2 = np.array(images[1])
        image_cv_p2 = cv2.cvtColor(image_cv_p2, cv2.COLOR_BGR2GRAY)
        _, binary_image_p2 = cv2.threshold(image_cv_p2, 180, 255, cv2.THRESH_BINARY)

        # Find contours in the binary images using OpenCV
        contours_p2, hierarchy_p2 = cv2.findContours(binary_image_p2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to hold the contours with the specified area range
        contour_p2 = None

        # Iterate over each contour for page 2
        for contour in contours_p2:
            # Find the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the KEY image from the original image
            key = image_cv_p2[y:y+h, x:x+w]

            # Calculate the total number of pixels in the contour
            total_pixels = key.shape[0] * key.shape[1]

            # Check if the area is within the specified range
            if 20000000 < total_pixels < 40000000:
                contour_p2 = contour
                break

        if contour_p2 is not None:
            # Find the bounding rectangle of the contour
            x_p2, y_p2, w_p2, h_p2 = cv2.boundingRect(contour_p2)

            # Crop the image from the original image based on the contour
            image_p1 = images[0].crop((x_p2, y_p2, x_p2+w_p2, y_p2+h_p2))
            image_p2 = images[1].crop((x_p2, y_p2, x_p2+w_p2, y_p2+h_p2))

            # Save the image in the PNG folder
            PNG_File_Name_p1 = f"{file_name}_p1.png"
            image_p1_path = os.path.join(PNG_Folder_Path, PNG_File_Name_p1)
            image_p1.save(image_p1_path)
            PNG_File_Name_p2 = f"{file_name}_p2.png"
            image_p2_path = os.path.join(PNG_Folder_Path, PNG_File_Name_p2)
            image_p2.save(image_p2_path)

            # Use the cropped image of the second page to generate key images
            key_folder_path = generate_keys_from_cropped_image(image_p2_path)
            PNG_Folder_Paths[file_name] = PNG_Folder_Path
            KEY_Folder_Paths[os.path.basename(key_folder_path)] = key_folder_path
            
        else:
            #print("No contour found within the specified area range for page 2.")
            # Crop coordinates for the second image (left, upper, right, lower)
            crop_coords_p2 = (500, 2000, 11000, 5000)
            crop_coords_p2 = adjust_crop_coordinates(image_width_p2, image_height_p2, crop_coords_p2)
            image_p1 = images[0].crop(crop_coords_p2)
            PNG_File_Name_p1 = f"{file_name}_p1.png"
            image_p1_path = os.path.join(PNG_Folder_Path, PNG_File_Name_p1)
            image_p1.save(image_p1_path)
            
            image_p2 = images[1].crop(crop_coords_p2)
            PNG_File_Name_p2 = f"{file_name}_p2.png"
            image_p2_path = os.path.join(PNG_Folder_Path, PNG_File_Name_p2)
            image_p2.save(image_p2_path)

            # Use the cropped image of the second page to generate key images
            key_folder_path = generate_keys_from_cropped_image(image_p2_path)
            PNG_Folder_Paths[file_name] = PNG_Folder_Path
            KEY_Folder_Paths[os.path.basename(key_folder_path)] = key_folder_path


# In[7]:


# Initialize dictionaries to store PNG and KEY folder paths
PNG_Folder_Paths = {}
KEY_Folder_Paths = {}

# Iterate through each folder and PDF files
for PDF_Folder_Name, PDF_Folder_Path in PDF_Folder_Paths.items():
    # Get a list of all PDF files in the current folder
    pdf_files = glob.glob(os.path.join(PDF_Folder_Path, '*.pdf'))
    for pdf_file in pdf_files:
        # Extract the file name without extension to use as the folder name for PNG images
        file_name = os.path.splitext(os.path.basename(pdf_file))[0]

        # Create a directory to save the PNG images
        PNG_Folder_Path = os.path.join(PDF_Folder_Path, file_name)
        if not os.path.exists(PNG_Folder_Path):
            os.makedirs(PNG_Folder_Path)
            
        create_images_from_pdf(pdf_file, file_name, PNG_Folder_Path)

# Print the running result
if PNG_Folder_Paths:
    print("Entire layout image processing complete.")
else:
    print("No PDF folders found to process images.")

if KEY_Folder_Paths:
    print("Key image processing complete.")
else:
    print("No PNG folders found to process Key images.")


# In[9]:


# Initialize the dictionary to store image pairs
image_pairs = {}

# Helper function to extract country and version information from the file name
def extract_country_and_version(file_name):
    # Design and Reference format: "*_<country> V<version>"
    ##ORI
    design_ref_match = re.search(r"_(\D{2,8}) V(\d+\.\d+)", file_name)
    if design_ref_match:
        country = design_ref_match.group(1)
        version = float(design_ref_match.group(2))
        return country, version
    
    design_ref_match3 = re.search(r" (\D{2,8}) V(\d+\.\d+)", file_name)
    if design_ref_match3:
        country = design_ref_match3.group(1)
        version = design_ref_match3.group(2)
        return country, version

    # Design and Reference format: "* <country> V<version>"
    design_ref_match2 = re.search(r"(\b[\w\s']{2,8}\b) V(\d+\.\d+)", file_name)
    if design_ref_match2:
        country = design_ref_match2.group(1)
        version = design_ref_match2.group(2)
        return country, version
    
    
    # PDK format: "PDK.<version>_*_<country>"
    # ORI 
    # pdk_match = re.search(r"\.(\d+)_.*_(\D{2,8})\_", file_name)
    
    # if pdk_match:
    #     country = pdk_match.group(2)
    #     version = float(pdk_match.group(1))
    #     print(country)
    #     print(version)
    #     return country, version
    
    #NEW
    pdk_match = re.search(r"\-(\d+\.\d+)_.*_([A-Z\-]+)(?:_|$)", file_name)
    if pdk_match:
        country = pdk_match.group(2)
        version = float(pdk_match.group(1))
        return country, version
    
    return None, None

# Iterate through each folder and PNG files
for PNG_Folder_Name, PNG_Folder_Path in PNG_Folder_Paths.items():
    # Extract country and version information from the folder name
    country, version = extract_country_and_version(PNG_Folder_Name)
    print(country)
    print(version)

    if country and version is not None:
        # If the country is not already in the image_pairs dictionary, add it with empty references
        if country not in image_pairs:
            image_pairs[country] = {'design': {'latest': {'folder_path': None, 'version': None}, 'second_newest': {'folder_path': None, 'version': None}},
                                    'reference': {'folder_path': None, 'version': None},
                                    'pdk': {'latest': {'folder_path': None, 'version': None}, 'second_newest': {'folder_path': None, 'version': None}}}

        # Update the image_pairs dictionary with the appropriate reference, design, and pdk images
        if "Reference" in PNG_Folder_Name and (image_pairs[country]['reference']['folder_path'] is None or version > image_pairs[country]['reference']['version']):
            image_pairs[country]['reference']['folder_path'] = PNG_Folder_Path
            image_pairs[country]['reference']['version'] = version
        elif "PDK" in PNG_Folder_Name:
            if image_pairs[country]['pdk']['latest']['folder_path'] is None:
                image_pairs[country]['pdk']['latest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['pdk']['latest']['version'] = version
            elif version > image_pairs[country]['pdk']['latest']['version']:
                image_pairs[country]['pdk']['second_newest']['folder_path'] = image_pairs[country]['pdk']['latest']['folder_path']
                image_pairs[country]['pdk']['second_newest']['version'] = image_pairs[country]['pdk']['latest']['version']
                image_pairs[country]['pdk']['latest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['pdk']['latest']['version'] = version
            elif image_pairs[country]['pdk']['second_newest']['folder_path'] is None or version > image_pairs[country]['pdk']['second_newest']['version']:
                image_pairs[country]['pdk']['second_newest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['pdk']['second_newest']['version'] = version
        
        elif "PDM" in PNG_Folder_Name:
            if image_pairs[country]['pdk']['latest']['folder_path'] is None:
                image_pairs[country]['pdk']['latest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['pdk']['latest']['version'] = version
            elif version > image_pairs[country]['pdk']['latest']['version']:
                image_pairs[country]['pdk']['second_newest']['folder_path'] = image_pairs[country]['pdk']['latest']['folder_path']
                image_pairs[country]['pdk']['second_newest']['version'] = image_pairs[country]['pdk']['latest']['version']
                image_pairs[country]['pdk']['latest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['pdk']['latest']['version'] = version
            elif image_pairs[country]['pdk']['second_newest']['folder_path'] is None or version > image_pairs[country]['pdk']['second_newest']['version']:
                image_pairs[country]['pdk']['second_newest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['pdk']['second_newest']['version'] = version
        
        else:
            if image_pairs[country]['design']['latest']['folder_path'] is None:
                image_pairs[country]['design']['latest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['design']['latest']['version'] = version
            elif version > image_pairs[country]['design']['latest']['version']:
                image_pairs[country]['design']['second_newest']['folder_path'] = image_pairs[country]['design']['latest']['folder_path']
                image_pairs[country]['design']['second_newest']['version'] = image_pairs[country]['design']['latest']['version']
                image_pairs[country]['design']['latest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['design']['latest']['version'] = version
            elif image_pairs[country]['design']['second_newest']['folder_path'] is None or version > image_pairs[country]['design']['second_newest']['version']:
                image_pairs[country]['design']['second_newest']['folder_path'] = PNG_Folder_Path
                image_pairs[country]['design']['second_newest']['version'] = version

# Print the running result
if image_pairs:
    print("Image pair processing complete.")
else:
    print("Image pair processing fail.")


# In[22]:





# In[12]:


image_pairs


# In[14]:


# Helper function to generate overlay images for a given pair of images (image1 and image2)
def Generate_overlay_images_multi(image2, image1):
    image1_copy = image1
    image2_copy = image2
    _, image1 = cv2.threshold(image1_copy, 180, 255, cv2.THRESH_BINARY)
    _, image2 = cv2.threshold(image2_copy, 180, 255, cv2.THRESH_BINARY)
    
    # Use only some parts of the two binary images from image1
    # The parts are defined by slicing the image1 array using integer indexes
    image1_parts = [
        image1[:int(0.25 * image1.shape[0]), :int(0.65 * image1.shape[1])],
        image1[int(0.25 * image1.shape[0]):, :int(0.65 * image1.shape[1])],
        image1[:int(0.25 * image1.shape[0]), int(0.65 * image1.shape[1]):int(0.8 * image1.shape[1])],
        image1[int(0.25 * image1.shape[0]):, int(0.65 * image1.shape[1]):int(0.8 * image1.shape[1])],
        image1[:int(0.25 * image1.shape[0]), int(0.8 * image1.shape[1]):],
        image1[int(0.25 * image1.shape[0]):, int(0.8 * image1.shape[1]):],
    ]

    # The cv2.copyMakeBorder function adds padding around the image to create an extended_image2
    extended_image2 = cv2.copyMakeBorder(image2, 400, 400, 400, 400, cv2.BORDER_CONSTANT, value=255)

    # Define a function to create overlay images by comparing image1_part with extended_image2
    def create_overlay_image(image1_part, extended_image2, shift_x, shift_y):
        overlay_size = (extended_image2.shape[1] + abs(shift_x), extended_image2.shape[0] + abs(shift_y))
        overlay_image = Image.new("RGB", overlay_size, (255, 255, 255))

        # Loop through each pixel in image1_part
        for x in range(image1_part.shape[1]):
            for y in range(image1_part.shape[0]):
                try:
                    pixel1 = image1_part[y, x]
                    shifted_x = x + shift_x
                    shifted_y = y + shift_y
                    if 0 <= shifted_x < extended_image2.shape[1] and 0 <= shifted_y < extended_image2.shape[0]:
                        pixel2 = extended_image2[shifted_y, shifted_x]
                    else:
                        pixel2 = 255  # Set pixel to white if out of bounds
                except IndexError:
                    pixel2 = 255  # Set pixel to white if out of bounds

                # Compare the pixel values of image1_part and the corresponding pixel in extended_image2
                if pixel1 == 0 and pixel2 == 0:  # Both pixels are black
                    overlay_image.putpixel((x, y), (0, 0, 0))  # Set pixel color to black
                elif pixel1 == 0 and pixel2 != 0:  # Only pixel in image1_part is black
                    overlay_image.putpixel((x, y), (0, 255, 0))  # Set pixel color to green
                elif pixel2 == 0 and pixel1 != 0:  # Only pixel in extended_image2 is black
                    overlay_image.putpixel((x, y), (255, 0, 0))  # Set pixel color to red

        return overlay_image

    # Use cv2.matchTemplate to find the best matching location for each image part in extended_image2
    match_results = []

    # Loop through each image part and find the best matching location using cv2.matchTemplate
    for image_part in image1_parts:
        match_result = cv2.matchTemplate(extended_image2, image_part, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        match_results.append(max_loc)

    # Get the x and y shifts for each image part based on the matching locations
    shifts_x = [result[0] for result in match_results]
    shifts_y = [result[1] for result in match_results]
    
    # Initialize lists to store cropped parts of image1 and image2 for each overlay image
    cropped_parts_image1 = []
    cropped_parts_image2 = []
    cropped_overlay_images = []
    
    for idx, image_part in enumerate(image1_parts):
        shift_x = shifts_x[idx]
        shift_y = shifts_y[idx]
        overlay_image = create_overlay_image(image_part, extended_image2, shift_x, shift_y)
        cropped_overlay_image = overlay_image.crop((0, 0, image_part.shape[1], image_part.shape[0]))
        cropped_overlay_images.append(cropped_overlay_image)

        # Get the corresponding part from the original image1
        if idx == 0:  # upper_left
            cropped_part_image1 = image1[:image_part.shape[0], :image_part.shape[1]]
        elif idx == 1:  # lower_left
            cropped_part_image1 = image1[image1.shape[0] - image_part.shape[0]:, :image_part.shape[1]]
        elif idx == 2:  # upper_middle
            cropped_part_image1 = image1[:image_part.shape[0], int(0.65 * image1.shape[1]):int(0.8 * image1.shape[1])]
        elif idx == 3:  # lower_middle
            cropped_part_image1 = image1[image1.shape[0] - image_part.shape[0]:, int(0.65 * image1.shape[1]):int(0.8 * image1.shape[1])]
        elif idx == 4:  # upper_right
            cropped_part_image1 = image1[:image_part.shape[0], int(0.8 * image1.shape[1]):]
        elif idx == 5:  # lower_right
            cropped_part_image1 = image1[image1.shape[0] - image_part.shape[0]:, int(0.8 * image1.shape[1]):]

        # Get the corresponding part from the extended_image2
        cropped_part_image2 = extended_image2[abs(shift_y):abs(shift_y) + cropped_overlay_image.height,
                                              abs(shift_x):abs(shift_x) + cropped_overlay_image.width]

        cropped_parts_image1.append(cropped_part_image1)
        cropped_parts_image2.append(cropped_part_image2)
        
    return cropped_overlay_images, cropped_parts_image1, cropped_parts_image2

# Helper function to generate overlay images for whole images (image1 and image2)
def Generate_overlay_images_single(image2, image1):    
    image1_copy = image1
    image2_copy = image2
    _, image1 = cv2.threshold(image1_copy, 180, 255, cv2.THRESH_BINARY)
    _, image2 = cv2.threshold(image2_copy, 180, 255, cv2.THRESH_BINARY)
    
    # The cv2.copyMakeBorder function adds padding around the image to create an extended_image2
    extended_image2 = cv2.copyMakeBorder(image2, 400, 400, 400, 400, cv2.BORDER_CONSTANT, value=255)
    
    # Define a function to create overlay images by comparing image1 with extended_image2
    def create_overlay_image(image1, extended_image2, shift_x, shift_y):
        overlay_size = (extended_image2.shape[1] + abs(shift_x), extended_image2.shape[0] + abs(shift_y))
        overlay_image = Image.new("RGB", overlay_size, (255, 255, 255))

        # Loop through each pixel in image1
        for x in range(image1.shape[1]):
            for y in range(image1.shape[0]):
                try:
                    pixel1 = image1[y, x]
                    shifted_x = x + shift_x
                    shifted_y = y + shift_y
                    if 0 <= shifted_x < extended_image2.shape[1] and 0 <= shifted_y < extended_image2.shape[0]:
                        pixel2 = extended_image2[shifted_y, shifted_x]
                    else:
                        pixel2 = 255  # Set pixel to white if out of bounds
                except IndexError:
                    pixel2 = 255  # Set pixel to white if out of bounds

                # Compare the pixel values of image1 and the corresponding pixel in extended_image2
                if pixel1 == 0 and pixel2 == 0:  # Both pixels are black
                    overlay_image.putpixel((x, y), (0, 0, 0))  # Set pixel color to black
                elif pixel1 == 0 and pixel2 != 0:  # Only pixel in image1 is black
                    overlay_image.putpixel((x, y), (0, 255, 0))  # Set pixel color to green
                elif pixel2 == 0 and pixel1 != 0:  # Only pixel in extended_image2 is black
                    overlay_image.putpixel((x, y), (255, 0, 0))  # Set pixel color to red

        return overlay_image

    match_result = cv2.matchTemplate(extended_image2, image1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

    # Get the x and y shifts for image based on the matching locations
    shift_x = max_loc[0]
    shift_y = max_loc[1]
    
    # Crop the overlay images to make them show just the area with the objects
    overlay_image = create_overlay_image(image1, extended_image2, shift_x, shift_y)
    cropped_overlay_image = overlay_image.crop((0, 0, image1.shape[1], image1.shape[0]))
        
    return cropped_overlay_image


# In[16]:


def analyze_and_save_contours(overlay_image, part_name, save_path, cropped_parts_image1, cropped_parts_image2):
    image_np = np.array(overlay_image)
    bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = 0
    
    # Create a list to store the valid contours
    valid_contours = []

    # Iterate over each contour and create keys
    for i, contour in enumerate(contours):
        
        # Find the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the KEY image from the original image
        key = image_np[y:y+h, x:x+w]

        # Ignore contours with unwanted shape
        if key.shape[0] < 100 or key.shape[1] < 100 or key.shape[0] > 1500 or key.shape[1] > 1500:
            continue
        
        # Calculate the total number of pixels in the contour
        total_pixels = key.shape[0] * key.shape[1]

        # Ignore contours with a small or large number of pixels
        if total_pixels < 60000 or total_pixels > 900000:
            continue
        
        # Check for overlap with other valid contours
        overlap = False
        for valid_contour in valid_contours:
            if (x >= valid_contour[0] and y >= valid_contour[1] and x + w <= valid_contour[2] and y + h <= valid_contour[3]):
                overlap = True
                break

        if not overlap:
            # Define the region of interest (ROI) as the center portion of the image
            roi_left = x + w // 15
            roi_right = x + 14 * w // 15
            roi_top = y + h // 20
            roi_bottom = y + 14 * h // 15

            # Resize 'key' image to match the size of 'cropped_roi_part_image1'
            key_resized = cv2.resize(key, (roi_right - roi_left, roi_bottom - roi_top))

            # Calculate green_area and red_area within the ROI
            roi_green_mask = np.all(key[roi_top - y:roi_bottom - y, roi_left - x:roi_right - x] == [0, 255, 0], axis=-1)
            roi_red_mask = np.all(key[roi_top - y:roi_bottom - y, roi_left - x:roi_right - x] == [0, 0, 255], axis=-1)

            green_area = np.count_nonzero(roi_green_mask)
            red_area = np.count_nonzero(roi_red_mask)

            if green_area + red_area > 200: #ORI 200
                index += 1

                # Convert 'cropped_roi_part_image1' and 'cropped_roi_part_image2' to 3 channels
                cropped_roi_part_image1_rgb = cv2.cvtColor(cropped_parts_image1[idx][roi_top:roi_bottom, roi_left:roi_right], cv2.COLOR_GRAY2RGB)
                cropped_roi_part_image2_rgb = cv2.cvtColor(cropped_parts_image2[idx][roi_top:roi_bottom, roi_left:roi_right], cv2.COLOR_GRAY2RGB)

                # Save contour images using np.hstack with the 'key_resized' image
                combined_image = np.hstack((cropped_roi_part_image1_rgb, key_resized, cropped_roi_part_image2_rgb))
                contour_image_path = os.path.join(save_path, f"{part_name}_contour_{index}.png")

                # Convert to PIL image and save
                contour_image_pil = Image.fromarray(combined_image)
                contour_image_pil.save(contour_image_path)

                # Append the current contour's bounding box to the list of valid contours
                valid_contours.append([x, y, x + w, y + h])


# In[17]:


# Iterate over the image pairs and save them along with the overlay images
for country, pair in image_pairs.items():
    
    design_latest_path = pair['design']['latest']['folder_path']
    design_second_newest_path = pair['design']['second_newest']['folder_path']
    reference_path = pair['reference']['folder_path']
    pdk_latest_path = pair['pdk']['latest']['folder_path']
    pdk_second_newest_path = pair['pdk']['second_newest']['folder_path']
    ###ORI
    # # Generate overlay images for latest design p1 and latest design p2
    # if design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p1.png")) and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
    #     latest_design_p1 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
    #     latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     overlay_images_design_latest_single = Generate_overlay_images_single(latest_design_p1, latest_design_p2)
    #     overlay_images_design_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_latest_design.png"))
        
    #     overlay_images_design_latest_multi, _, _ = Generate_overlay_images_multi(latest_design_p1, latest_design_p2)
    #     # Save the overlay images
    #     for idx, overlay_image in enumerate(overlay_images_design_latest_multi):
    #         part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
    #         overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_latest_design_{part_name}.png"))

    ###NEW Generate overlay images for latest design p1 and latest design p2
    if design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p1.png")) and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
        latest_design_p1 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
        latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        overlay_images_design_latest_single = Generate_overlay_images_single(latest_design_p1, latest_design_p2)
        overlay_images_design_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_latest_design.png"))
        
        overlay_images_design_latest_multi, cropped_parts_latest_design_p1, cropped_parts_latest_design_p2 = Generate_overlay_images_multi(latest_design_p1, latest_design_p2)
        # Save the overlay images
        save_contours_path = os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_design_p1_and_design_p2_error_key")
        if not os.path.exists(save_contours_path):
            os.makedirs(save_contours_path)

        for idx, overlay_image in enumerate(overlay_images_design_latest_multi):
            part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
            overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_latest_design_{part_name}.png"))
            analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_latest_design_p1, cropped_parts_latest_design_p2)

        if not os.listdir(save_contours_path):
            os.rmdir(save_contours_path)


    
    ###NEW Generate overlay images for latest PDK p1 and latest PDK p2
    # if pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png")) and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(pdk_latest_path)}_p2.png")):
    #     latest_pdk_p1 = cv2.imread(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
    #     latest_pdk_p2 = cv2.imread(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     overlay_images_pdk_latest_single = Generate_overlay_images_single(latest_pdk_p1, latest_pdk_p2)
    #     overlay_images_pdk_latest_single.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_latest_pdk.png"))
        
    #     overlay_images_pdk_latest_multi, cropped_parts_latest_pdk_p1, cropped_parts_latest_pdk_p2 = Generate_overlay_images_multi(latest_pdk_p1, latest_pdk_p2)
    #     # Save the overlay images
    #     save_contours_path = os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_pdk_p1_and_pdk_p2_error_key")
    #     if not os.path.exists(save_contours_path):
    #         os.makedirs(save_contours_path)

    #     for idx, overlay_image in enumerate(overlay_images_pdk_latest_multi):
    #         part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
    #         overlay_image.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_latest_pdk_{part_name}.png"))
    #         analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_latest_pdk_p1, cropped_parts_latest_pdk_p2)
        
   
    
    ###ORI
    # # Generate overlay images for second newest design p2 and latest design p2
    # if design_second_newest_path and os.path.exists(os.path.join(design_second_newest_path, f"{os.path.basename(design_second_newest_path)}_p2.png")) and design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
    #     second_newest_design_p2 = cv2.imread(os.path.join(design_second_newest_path, f"{os.path.basename(design_second_newest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     overlay_images_design_second_latest_single = Generate_overlay_images_single(second_newest_design_p2, latest_design_p2)
    #     overlay_images_design_second_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_second_latest.png"))
        
    #     overlay_images_design_second_latest_multi, _, _ = Generate_overlay_images_multi(second_newest_design_p2, latest_design_p2)
    #     # Save the overlay images
    #     for idx, overlay_image in enumerate(overlay_images_design_second_latest_multi):
    #         part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
    #         overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_second_latest_{part_name}.png"))

    ###NEW Generate overlay images for second newest design p2 and latest design p2
    if design_second_newest_path and os.path.exists(os.path.join(design_second_newest_path, f"{os.path.basename(design_second_newest_path)}_p2.png")) and design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
        second_newest_design_p2 = cv2.imread(os.path.join(design_second_newest_path, f"{os.path.basename(design_second_newest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        overlay_images_design_second_latest_single = Generate_overlay_images_single(second_newest_design_p2, latest_design_p2)
        overlay_images_design_second_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_second_latest.png"))
        
        overlay_images_design_second_latest_multi, cropped_parts_second_newest_design_p2, cropped_parts_latest_design_p2 = Generate_overlay_images_multi(second_newest_design_p2, latest_design_p2)
        # Save the overlay images
        save_contours_path = os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_second_latest_error_key")
        if not os.path.exists(save_contours_path):
            os.makedirs(save_contours_path)

        for idx, overlay_image in enumerate(overlay_images_design_second_latest_multi):
            part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
            overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_second_latest_{part_name}.png"))
            analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_second_newest_design_p2, cropped_parts_latest_design_p2)

        if not os.listdir(save_contours_path):
            os.rmdir(save_contours_path)

    # Generate overlay images for reference p2 and latest design p2
    if reference_path and os.path.exists(os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2.png")) and design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
        reference_p2 = cv2.imread(os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        overlay_images_reference_latest_single = Generate_overlay_images_single(reference_p2, latest_design_p2)
        overlay_images_reference_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_reference_design.png"))

        # Generate overlay images and cropped parts
        overlay_images_reference_design_multi, cropped_parts_reference, cropped_parts_design = Generate_overlay_images_multi(reference_p2, latest_design_p2)

        save_contours_path = os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_reference_design_error_key")
        
        if not os.path.exists(save_contours_path):
            os.makedirs(save_contours_path)

        # Save the overlay images and analyze contours
        for idx, overlay_image in enumerate(overlay_images_reference_design_multi):
            part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
            overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_reference_design_{part_name}.png"))
            analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_reference, cropped_parts_design)
        
        if not os.listdir(save_contours_path):
            os.rmdir(save_contours_path)
        
    # Generate overlay images for reference p2 and latest PDK p1    
    if reference_path and os.path.exists(os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2.png")) and pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png")):
        reference_p2 = cv2.imread(os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        pdk_p1 = cv2.imread(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
        overlay_images_reference_pdk_single = Generate_overlay_images_single(reference_p2, pdk_p1)
        overlay_images_reference_pdk_single.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_reference_pdk.png"))

        # Generate overlay images and cropped parts
        overlay_images_reference_pdk_multi, cropped_parts_reference, cropped_parts_pdk = Generate_overlay_images_multi(reference_p2, pdk_p1)

        save_contours_path = os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_reference_pdk_error_key")
        
        if not os.path.exists(save_contours_path):
            os.makedirs(save_contours_path)

        # Save the overlay images and analyze contours
        for idx, overlay_image in enumerate(overlay_images_reference_pdk_multi):
            part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
            overlay_image.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_reference_pdk_{part_name}.png"))
            analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_reference, cropped_parts_pdk)
        
        if not os.listdir(save_contours_path):
            os.rmdir(save_contours_path)
        
    # Generate overlay images for latest design p2 and latest PDK p1
    if design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")) and pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png")):
        latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
        pdk_p1 = cv2.imread(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
        overlay_images_latest_pdk_single = Generate_overlay_images_single(latest_design_p2, pdk_p1)
        overlay_images_latest_pdk_single.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_design_pdk.png"))

        # Generate overlay images and cropped parts
        overlay_images_design_pdk_multi, cropped_parts_latest, cropped_parts_pdk = Generate_overlay_images_multi(latest_design_p2, pdk_p1)

        save_contours_path = os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_design_pdk_error_key")
        
        if not os.path.exists(save_contours_path):
            os.makedirs(save_contours_path)

        # Save the overlay images and analyze contours
        for idx, overlay_image in enumerate(overlay_images_design_pdk_multi):
            part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
            overlay_image.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_design_pdk_{part_name}.png"))
            analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_latest, cropped_parts_pdk)
        
        if not os.listdir(save_contours_path):
            os.rmdir(save_contours_path)
    ###ORI        
    # # Generate overlay images for second newest PDK p1 and latest PDK p1
    # if pdk_second_newest_path and os.path.exists(os.path.join(pdk_second_newest_path, f"{os.path.basename(pdk_second_newest_path)}_p1.png")) and pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png")):
    #     second_newest_pdk_p2 = cv2.imread(os.path.join(pdk_second_newest_path, f"{os.path.basename(pdk_second_newest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
    #     latest_pdk_p2 = cv2.imread(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
    #     overlay_images_pdk_second_latest_single = Generate_overlay_images_single(second_newest_pdk_p2, latest_pdk_p2)
    #     overlay_images_pdk_second_latest_single.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_second_latest.png"))
        
    #     overlay_images_pdk_second_latest_multi, _, _ = Generate_overlay_images_multi(second_newest_pdk_p2, latest_pdk_p2)
    #     # Save the overlay images
    #     for idx, overlay_image in enumerate(overlay_images_pdk_second_latest_multi):
    #         part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
    #         overlay_image.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_second_latest_{part_name}.png"))
    # Generate overlay images for second newest PDK p1 and latest PDK p1
            
    ###NEW Generate overlay images for second newest PDK p1 and latest PDK p1
    if pdk_second_newest_path and os.path.exists(os.path.join(pdk_second_newest_path, f"{os.path.basename(pdk_second_newest_path)}_p1.png")) and pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png")):
        second_newest_pdk_p1 = cv2.imread(os.path.join(pdk_second_newest_path, f"{os.path.basename(pdk_second_newest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
        latest_pdk_p1 = cv2.imread(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
        overlay_images_pdk_second_latest_single = Generate_overlay_images_single(second_newest_pdk_p1, latest_pdk_p1)
        overlay_images_pdk_second_latest_single.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_second_latest.png"))
        
        overlay_images_pdk_second_latest_multi, cropped_parts_second_newest_pdk_p1, cropped_parts_latest_pdk_p1 = Generate_overlay_images_multi(second_newest_pdk_p1, latest_pdk_p1)
        # Save the overlay images
        save_contours_path = os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_second_latest_error_key")
        if not os.path.exists(save_contours_path):
            os.makedirs(save_contours_path)

        for idx, overlay_image in enumerate(overlay_images_pdk_second_latest_multi):
            part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
            overlay_image.save(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_overlay_second_latest_{part_name}.png"))
            analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_second_newest_pdk_p1, cropped_parts_latest_pdk_p1)
        
        if not os.listdir(save_contours_path):
            os.rmdir(save_contours_path)

    print(f"Layout overlay image generation processing for {country} complete.")


# Print the running result
print("All layout overlay image generation processing complete.")


# In[230]:


####ORI    
# Generate overlay images for latest design p1 and latest design p2
    # if design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p1.png")) and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
    #     latest_design_p1 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p1.png"), cv2.IMREAD_GRAYSCALE)
    #     latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     overlay_images_design_latest_single = Generate_overlay_images_single(latest_design_p1, latest_design_p2)
    #     overlay_images_design_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_latest_design.png"))
        
    #     overlay_images_design_latest_multi, _, _ = Generate_overlay_images_multi(latest_design_p1, latest_design_p2)
    #     # Save the overlay images
    #     for idx, overlay_image in enumerate(overlay_images_design_latest_multi):
    #         part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
    #         overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_latest_design_{part_name}.png"))

    # # Generate overlay images for second newest design p2 and latest design p2
    # if design_second_newest_path and os.path.exists(os.path.join(design_second_newest_path, f"{os.path.basename(design_second_newest_path)}_p2.png")) and design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png")):
    #     second_newest_design_p2 = cv2.imread(os.path.join(design_second_newest_path, f"{os.path.basename(design_second_newest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     latest_design_p2 = cv2.imread(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2.png"), cv2.IMREAD_GRAYSCALE)
    #     overlay_images_design_second_latest_single = Generate_overlay_images_single(second_newest_design_p2, latest_design_p2)
    #     overlay_images_design_second_latest_single.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_second_latest.png"))
        
    #     overlay_images_design_second_latest_multi, _, _ = Generate_overlay_images_multi(second_newest_design_p2, latest_design_p2)
    #     # Save the overlay images
    #     for idx, overlay_image in enumerate(overlay_images_design_second_latest_multi):
    #         part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
    #         overlay_image.save(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_overlay_second_latest_{part_name}.png"))


  


# In[19]:


###NEW compare other language with US/CH
# Iterate over the image pairs and save them along with the overlay images(Design)
for country, pair in image_pairs.items():
    if country != 'US':  # Check if the country is not CH or US
        fra_latest_path = pair['design']['latest']['folder_path']
        fra_version = pair['design']['latest']['version']
        # print(fra_latest_path)
        # print(fra_version)
        # Check if US or CH exists, and find the corresponding version
        # if 'CH' in image_pairs:
        #     c_latest_path = image_pairs['CH']['design']['latest']['folder_path']
        #     c_version = image_pairs['CH']['design']['latest']['version']
        #     # print(c_latest_path)
        #     if c_latest_path:
        #         # Get base filenames without country and version
        #         fra_base_filename = os.path.splitext(os.path.basename(fra_latest_path))[0].split('_')[0]
        #         c_base_filename = os.path.splitext(os.path.basename(c_latest_path))[0].split('_')[0]

        #         # Check if base filenames are similar
        #         # if fra_base_filename == c_base_filename:#ORI
        #         # Generate overlay images for FRA and CH/US p2
        #         fra_p2_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_p2.png")
        #         c_p2_path = os.path.join(c_latest_path, f"{os.path.basename(c_latest_path)}_p2.png")
                    
        #         if os.path.exists(fra_p2_path) and os.path.exists(c_p2_path):
        #             fra_p2 = cv2.imread(fra_p2_path, cv2.IMREAD_GRAYSCALE)
        #             c_p2 = cv2.imread(c_p2_path, cv2.IMREAD_GRAYSCALE)
                        
        #             overlay_images_fra_c_single = Generate_overlay_images_single(fra_p2, c_p2)
        #             overlay_images_fra_c_single.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_CH_overlay_latest.png"))
                        
        #             overlay_images_fra_c_multi, cropped_parts_fra_p2, cropped_parts_c_p2 = Generate_overlay_images_multi(fra_p2, c_p2)
                        
        #             # Save the overlay images and analyze contours
        #             save_contours_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_CH_error_key")
        #             if not os.path.exists(save_contours_path):
        #                 os.makedirs(save_contours_path)

        #             for idx, overlay_image in enumerate(overlay_images_fra_c_multi):
        #                 part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
        #                 overlay_image.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_CH_overlay_{part_name}.png"))
        #                 analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_fra_p2, cropped_parts_c_p2)
                            
        #             print("Complete Compare CH (Design)")
        #             if not os.listdir(save_contours_path):
        #                 os.rmdir(save_contours_path)
                            

            
        if 'US' in image_pairs:
            c_latest_path = image_pairs['US']['design']['latest']['folder_path']
            c_version = image_pairs['US']['design']['latest']['version']
            # print(c_latest_path)
            if c_latest_path:
                # Get base filenames without country and version
                fra_base_filename = os.path.splitext(os.path.basename(fra_latest_path))[0].split('_')[0]
                c_base_filename = os.path.splitext(os.path.basename(c_latest_path))[0].split('_')[0]
                print(fra_base_filename)
                print(c_base_filename)

                # Check if base filenames are similar
                # if fra_base_filename == c_base_filename: #ORI
                # Generate overlay images for FRA and CH/US p2
                fra_p2_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_p2.png")
                c_p2_path = os.path.join(c_latest_path, f"{os.path.basename(c_latest_path)}_p2.png")
                    
                if os.path.exists(fra_p2_path) and os.path.exists(c_p2_path):
                    fra_p2 = cv2.imread(fra_p2_path, cv2.IMREAD_GRAYSCALE)
                    c_p2 = cv2.imread(c_p2_path, cv2.IMREAD_GRAYSCALE)
                        
                    overlay_images_fra_c_single = Generate_overlay_images_single(fra_p2, c_p2)
                    overlay_images_fra_c_single.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_US_overlay_latest.png"))
                        
                    overlay_images_fra_c_multi, cropped_parts_fra_p2, cropped_parts_c_p2 = Generate_overlay_images_multi(fra_p2, c_p2)
                        
                    # Save the overlay images and analyze contours
                    save_contours_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_US_error_key")
                    if not os.path.exists(save_contours_path):
                        os.makedirs(save_contours_path)

                    for idx, overlay_image in enumerate(overlay_images_fra_c_multi):
                        part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
                        overlay_image.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_US_overlay_{part_name}.png"))
                        analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_fra_p2, cropped_parts_c_p2)
                            
                    print("Complete Compare US (Design)")
                    if not os.listdir(save_contours_path):
                        os.rmdir(save_contours_path)

        # else:
        #     print("Neither US nor CH is available.")
                            

# Iterate over the image pairs and save them along with the overlay images(PDK)
for country, pair in image_pairs.items():
    if country != 'US':  # Check if the country is not CH or US
        fra_latest_path = pair['pdk']['latest']['folder_path']
        fra_version = pair['pdk']['latest']['version']
        # print(fra_latest_path)
        # print(fra_version)
        # Check if US or CH exists, and find the corresponding version
        # if 'CH' in image_pairs:
        #     c_latest_path = image_pairs['CH']['pdk']['latest']['folder_path']
        #     c_version = image_pairs['CH']['pdk']['latest']['version']
        #     # print(c_latest_path)
        #     if c_latest_path:
        #         # Get base filenames without country and version
        #         fra_base_filename = os.path.splitext(os.path.basename(fra_latest_path))[0].split('_')[0]
        #         c_base_filename = os.path.splitext(os.path.basename(c_latest_path))[0].split('_')[0]

        #         # # Check if base filenames are similar
        #         # if fra_base_filename == c_base_filename:  #ORI
        #             # Generate overlay images for FRA and CH/US p2
        #         fra_p2_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_p1.png")
        #         c_p2_path = os.path.join(c_latest_path, f"{os.path.basename(c_latest_path)}_p1.png")
                    
        #         if os.path.exists(fra_p2_path) and os.path.exists(c_p2_path):
        #             fra_p2 = cv2.imread(fra_p2_path, cv2.IMREAD_GRAYSCALE)
        #             c_p2 = cv2.imread(c_p2_path, cv2.IMREAD_GRAYSCALE)
                        
        #             overlay_images_fra_c_single = Generate_overlay_images_single(fra_p2, c_p2)
        #             overlay_images_fra_c_single.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_CH_overlay_latest.png"))
                        
        #             overlay_images_fra_c_multi, cropped_parts_fra_p2, cropped_parts_c_p2 = Generate_overlay_images_multi(fra_p2, c_p2)
                        
        #             # Save the overlay images and analyze contours
        #             save_contours_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_CH_error_key")
        #             if not os.path.exists(save_contours_path):
        #                 os.makedirs(save_contours_path)

        #             for idx, overlay_image in enumerate(overlay_images_fra_c_multi):
        #                 part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
        #                 overlay_image.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_CH_overlay_{part_name}.png"))
        #                 analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_fra_p2, cropped_parts_c_p2)
                        
        #             print("Complete Compare CH (PDK)")
        #             if not os.listdir(save_contours_path):
        #                 os.rmdir(save_contours_path)

            
        if 'US' in image_pairs:
            c_latest_path = image_pairs['US']['pdk']['latest']['folder_path']
            c_version = image_pairs['US']['pdk']['latest']['version']
            # print(c_latest_path)
            if c_latest_path:
                # Get base filenames without country and version
                fra_base_filename = os.path.splitext(os.path.basename(fra_latest_path))[0].split('_')[0]
                c_base_filename = os.path.splitext(os.path.basename(c_latest_path))[0].split('_')[0]
                # print(fra_base_filename)
                # print(c_base_filename)

                # # Check if base filenames are similar
                # if fra_base_filename == c_base_filename: #ORI
                # Generate overlay images for FRA and CH/US p2
                fra_p2_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_p1.png")
                c_p2_path = os.path.join(c_latest_path, f"{os.path.basename(c_latest_path)}_p1.png")
                    
                if os.path.exists(fra_p2_path) and os.path.exists(c_p2_path):
                    fra_p2 = cv2.imread(fra_p2_path, cv2.IMREAD_GRAYSCALE)
                    c_p2 = cv2.imread(c_p2_path, cv2.IMREAD_GRAYSCALE)
                        
                    overlay_images_fra_c_single = Generate_overlay_images_single(fra_p2, c_p2)
                    overlay_images_fra_c_single.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_US_overlay_latest.png"))
                        
                    overlay_images_fra_c_multi, cropped_parts_fra_p2, cropped_parts_c_p2 = Generate_overlay_images_multi(fra_p2, c_p2)
                        
                    # Save the overlay images and analyze contours
                    save_contours_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_US_error_key")
                    if not os.path.exists(save_contours_path):
                        os.makedirs(save_contours_path)

                    for idx, overlay_image in enumerate(overlay_images_fra_c_multi):
                        part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
                        overlay_image.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_US_overlay_{part_name}.png"))
                        analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_fra_p2, cropped_parts_c_p2)
                        
                    print("Complete Compare US (PDK)")
                    if not os.listdir(save_contours_path):
                        os.rmdir(save_contours_path)








        # # Find the corresponding version of CH or US
        # for c in ['CH', 'US']:
        #     c_latest_path = image_pairs[c]['design']['latest']['folder_path']
        #     c_version = image_pairs[c]['design']['latest']['version']
        #     print(c_latest_path)
        #     print(c_version)

        #     if c_latest_path:
        #         # Generate overlay images for FRA and CH/US p2
        #         fra_p2_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_p2.png")
        #         c_p2_path = os.path.join(c_latest_path, f"{os.path.basename(c_latest_path)}_p2.png")
                
        #         if os.path.exists(fra_p2_path) and os.path.exists(c_p2_path):
        #             fra_p2 = cv2.imread(fra_p2_path, cv2.IMREAD_GRAYSCALE)
        #             c_p2 = cv2.imread(c_p2_path, cv2.IMREAD_GRAYSCALE)
                    
        #             overlay_images_fra_c_single = Generate_overlay_images_single(fra_p2, c_p2)
        #             overlay_images_fra_c_single.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_{c}_overlay_latest.png"))
                    
        #             overlay_images_fra_c_multi, cropped_parts_fra_p2, cropped_parts_c_p2 = Generate_overlay_images_multi(fra_p2, c_p2)
                    
        #             # Save the overlay images and analyze contours
        #             save_contours_path = os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_{c}_error_key")
        #             if not os.path.exists(save_contours_path):
        #                 os.makedirs(save_contours_path)

        #             for idx, overlay_image in enumerate(overlay_images_fra_c_multi):
        #                 part_name = ['upper_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'lower_right'][idx]
        #                 overlay_image.save(os.path.join(fra_latest_path, f"{os.path.basename(fra_latest_path)}_{c}_overlay_{part_name}.png"))
        #                 analyze_and_save_contours(overlay_image, part_name, save_contours_path, cropped_parts_fra_p2, cropped_parts_c_p2)

        #             if not os.listdir(save_contours_path):
        #                 os.rmdir(save_contours_path)
                    
        #         break  # Stop searching for the corresponding version after finding one


# In[21]:


# Function to resize images using OpenCV
def resize_image(image, target_size):
    original_size = (image.shape[1], image.shape[0])
    
    if target_size[0] > original_size[0] or target_size[1] > original_size[1]:
        # Use cubic interpolation for upsampling
        interpolation_method = cv2.INTER_CUBIC
    else:
        # Use area interpolation for downsampling
        interpolation_method = cv2.INTER_AREA

    resized_image = cv2.resize(image, target_size, interpolation=interpolation_method)
    return resized_image

def generate_overlay_and_comparison_key_images(image1_key_folder, image2_key_folder, source_type1, source_type2):
    # Load all images from image1_key_folder and resize them
    image1_key_files = [f for f in os.listdir(image1_key_folder) if f.endswith(".png")]
    image1_key_list = []

    for image_file in image1_key_files:
        if not image_file.endswith(".png"):
            continue
        image_path = os.path.join(image1_key_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_resized = resize_image(image, (400, 400))
        image1_key_list.append(image_resized)

    # Load all images from image2_key_folder and resize them
    image2_key_files = [f for f in os.listdir(image2_key_folder) if f.endswith(".png")]
    image2_key_list = []

    for image_file in image2_key_files:
        if not image_file.endswith(".png"):
            continue
        image_path = os.path.join(image2_key_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_resized = resize_image(image, (400, 400))
        image2_key_list.append(image_resized)

    # Prepare the images for comparison
    image1_key_set = np.array(image1_key_list)
    image2_key_set = np.array(image2_key_list)

    # Check the similarity between each image in image1_key_set and image2_key_set
    model = keras.applications.MobileNet(include_top=False, weights='imagenet', input_shape=(400, 400, 3))
    image1_key_set_features = model.predict(np.tile(image1_key_set[..., np.newaxis], (1, 1, 1, 3)))
    image2_key_set_features = model.predict(np.tile(image2_key_set[..., np.newaxis], (1, 1, 1, 3)))

    # Find the matched image in image2_key_set for each image in image1_key_set
    for i in range(len(image1_key_list)):
        image1_key = cv2.imread(os.path.join(image1_key_folder, image1_key_files[i]))
        image1_key_features = image1_key_set_features[i]

        # Compute the similarity score between each image1_key and the images in image2_key_set
        similarities = np.sum(image1_key_features * image2_key_set_features, axis=(1, 2, 3))
        matched_index = np.argmax(similarities)

        matched_image1_key = image1_key
        matched_image2_key = cv2.imread(os.path.join(image2_key_folder, image2_key_files[matched_index]))

        # Get the dimensions of the image1_key and matched_image2_key
        image1_key_height, image1_key_width = image1_key.shape[:2]
        matched_image2_key_height, matched_image2_key_width = matched_image2_key.shape[:2]

        # Calculate the center coordinates of both images
        image1_key_center_x = image1_key_width // 2
        image1_key_center_y = image1_key_height // 2
        matched_image2_key_center_x = matched_image2_key_width // 2
        matched_image2_key_center_y = matched_image2_key_height // 2

        # Calculate the dimensions of the overlapping area
        overlap_width = min(image1_key_center_x, matched_image2_key_center_x)
        overlap_height = min(image1_key_center_y, matched_image2_key_center_y)

        # Calculate the cropping coordinates for image1_key
        image1_key_x1 = image1_key_center_x - overlap_width
        image1_key_x2 = image1_key_center_x + overlap_width
        image1_key_y1 = image1_key_center_y - overlap_height
        image1_key_y2 = image1_key_center_y + overlap_height

        # Calculate the cropping coordinates for matched_image2_key
        matched_image2_key_x1 = matched_image2_key_center_x - overlap_width
        matched_image2_key_x2 = matched_image2_key_center_x + overlap_width
        matched_image2_key_y1 = matched_image2_key_center_y - overlap_height
        matched_image2_key_y2 = matched_image2_key_center_y + overlap_height

        # Crop the images to the overlapping area
        cropped_image1_key = image1_key[image1_key_y1:image1_key_y2, image1_key_x1:image1_key_x2]
        cropped_matched_image2_key = matched_image2_key[matched_image2_key_y1:matched_image2_key_y2, matched_image2_key_x1:matched_image2_key_x2]

        # Create an overlay key image to visualize the differences between the images
        overlay_key = Image.new("RGB", (overlap_width * 2, overlap_height * 2), (255, 255, 255))

        # Iterate over each pixel in the overlapping area
        for x in range(overlap_width * 2):
            for y in range(overlap_height * 2):
                # Get the pixel values from both images
                pixel1 = cropped_image1_key[y, x]
                pixel2 = cropped_matched_image2_key[y, x]

                # Compare the pixel values
                if np.all(pixel1 == [0, 0, 0]) and np.all(pixel2 == [0, 0, 0]):  # Both pixels are black
                    overlay_key.putpixel((x, y), (0, 0, 0))  # Set pixel color to black
                elif np.all(pixel1 == [0, 0, 0]) and not np.all(pixel2 == [0, 0, 0]):  # Only pixel in image1_key is black
                    overlay_key.putpixel((x, y), (0, 255, 0))  # Set pixel color to green
                elif np.all(pixel2 == [0, 0, 0]) and not np.all(pixel1 == [0, 0, 0]):  # Only pixel in image2_key is black
                    overlay_key.putpixel((x, y), (255, 0, 0))  # Set pixel color to red

        # Create folders to store overlay key images and comparison key images
        overlay_key_folder = os.path.join(image1_key_folder, f"{source_type1}_{source_type2}_overlay_key")
        if not os.path.exists(overlay_key_folder):
            os.makedirs(overlay_key_folder)

        comparison_key_folder = os.path.join(image1_key_folder, f"{source_type1}_{source_type2}_comparison_key")
        if not os.path.exists(comparison_key_folder):
            os.makedirs(comparison_key_folder)            
            
        # Create the comparison key image by concatenating the two images side by side
        comparison_key_image = np.hstack((cropped_image1_key, cropped_matched_image2_key, overlay_key))

        # Convert the comparison_key_image to a Pillow Image object
        comparison_key_pillow = Image.fromarray(comparison_key_image)

        # Save the overlay key image in the overlay_key_folder
        overlay_key_filename = os.path.join(overlay_key_folder, f"overlay_key_{int(image1_key_files[i].split('_')[1].split('.')[0])}_ref_{matched_index}.png")
        overlay_key.save(overlay_key_filename)

        # Save the comparison key image in the comparison_key_folder using Pillow
        comparison_key_filename = os.path.join(comparison_key_folder, f"comparison_key_{int(image1_key_files[i].split('_')[1].split('.')[0])}_ref_{matched_index}.png")
        comparison_key_pillow.save(comparison_key_filename)


# In[24]:


# Iterate over the image pairs and save them along with the overlay images
for country, pair in image_pairs.items():

    design_latest_path = pair['design']['latest']['folder_path']
    design_second_newest_path = pair['design']['second_newest']['folder_path']
    reference_path = pair['reference']['folder_path']
    pdk_latest_path = pair['pdk']['latest']['folder_path']
    pdk_second_newest_path = pair['pdk']['second_newest']['folder_path']

# Generate overlay images and comparison images for reference p2 keys and latest design p2 keys
if reference_path and os.path.exists(os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2 Key")) and design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2 Key")):
    generate_overlay_and_comparison_key_images(
        os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2 Key"),
        os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2 Key"),
        "Design",
        "Reference"
    )

# Generate overlay images and comparison images for reference p2 keys and latest pdk p1 keys
if reference_path and os.path.exists(os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2 Key")) and pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1 Key")):
    generate_overlay_and_comparison_key_images(
        os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1 Key"),
        os.path.join(reference_path, f"{os.path.basename(reference_path)}_p2 Key"),
        "PDK",
        "Reference"
    )

# Generate overlay images and comparison images for design p2 keys and latest pdk p1 keys
if design_latest_path and os.path.exists(os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2 Key")) and pdk_latest_path and os.path.exists(os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1 Key")):
    generate_overlay_and_comparison_key_images(
        os.path.join(pdk_latest_path, f"{os.path.basename(pdk_latest_path)}_p1 Key"),
        os.path.join(design_latest_path, f"{os.path.basename(design_latest_path)}_p2 Key"),
        "PDK",
        "Design"
    )
        
    print(f"Processing Key overlay images and comparison images for {country} complete.")
        
# Print running result
print("Key overlay image and comparison image generation processing complete.")


# In[253]:


input("Press Enter To Exit")


