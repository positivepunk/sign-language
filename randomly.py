import os
import random
import shutil

# Set the default input folder name
input_folder = os.path.join('sampledigits')

# Set the default output folder name
output_folder = 'sampledf'

# Get a list of subfolders (targets) in the input folder
subfolders = os.listdir(input_folder)

# Set the desired number of samples for each class
num_samples_per_class = 500

for subfolder in subfolders:
    # Construct the image folder path for the current subfolder
    image_folder = os.path.join(input_folder, subfolder)

    # Set the output subfolder name
    subfolder_name = subfolder

    # Set the output folder path
    output_folder_path = os.path.join(output_folder, subfolder_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Get a list of image file paths in the current subfolder
    image_files = os.listdir(image_folder)

    # Randomly select the desired number of samples from the current class
    selected_image_files = random.sample(image_files, num_samples_per_class)

    for image_file in selected_image_files:
        # Copy the selected image files to the output folder
        source_path = os.path.join(image_folder, image_file)
        destination_path = os.path.join(output_folder_path, image_file)
        shutil.copy(source_path, destination_path)

    print("Done with", subfolder)
