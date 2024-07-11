import os
import re
import cv2
import numpy as np

def count_jpg_files_in_subfolders(parent_folder):
    folder_jpg_count = {}

    # Duyệt qua tất cả các thư mục con
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            jpg_count = 0

            # Duyệt qua tất cả các tệp trong thư mục con
            for file in os.listdir(subfolder_path):
                if file.lower().endswith('.jpg'):
                    jpg_count += 1

            folder_jpg_count[subfolder] = jpg_count

    # Sắp xếp các thư mục dựa trên số lượng tệp JPG từ lớn đến bé
    sorted_folders = sorted(folder_jpg_count.items(), key=lambda x: x[1], reverse=True)

    # In ra kết quả
    for folder, count in sorted_folders:
        print(f'{folder} - {count}')

def rename_folders(base_folder_path):
    # Get list of all folders in the base directory
    folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]

    # Iterate over the folders and rename them
    for folder in folders:
        if folder.startswith('pins_'):
            new_name = folder[len('pins_'):]
            old_path = os.path.join(base_folder_path, folder)
            new_path = os.path.join(base_folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed folder '{folder}' to '{new_name}'")

def read_images_from_folder(folder_path):
    # List to store the images
    images = []

    # Supported image file extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Get list of all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]

    # Iterate over the files and read images
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to read image: {image_file}")

    return images

def create_ones_array_from_images(images):
    # Number of images
    num_images = len(images)

    # Example: Create a 1D array of ones with length equal to the number of images
    ones_array = np.ones(num_images)

    return ones_array

def extract_names_from_filenames(folder_path):
    # List to store extracted names
    names = []

    # Regular expression pattern to match the required format
    pattern = re.compile(r'^(.+?)\d+_\d+\.jpg$')

    # Get list of all files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Iterate over the files and extract names
    for image_file in image_files:
        match = pattern.match(image_file)
        if match:
            name_with_number = match.group(1).strip()  # Extract name and remove trailing spaces
            # Remove any extra spaces
            name = ' '.join(name_with_number.split())
            names.append(name)

    return names

# Example usage
# folder_path = 'D:/Anaconda/Face-Recognition-Based-Attendance-System/test_dataset/pack2/trained'
# images = read_images_from_folder(folder_path)
# ones_array = [1] * len(images)
# print(ones_array)
folder_path = 'D:/Anaconda/105_classes_pins_dataset'
#names = extract_names_from_filenames(folder_path)
#rename_folders(folder_path)
#print(names)
count_jpg_files_in_subfolders(folder_path)