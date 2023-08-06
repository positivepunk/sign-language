import os
import shutil
from sklearn.model_selection import train_test_split


# Set the path to your dataset directory
dataset_path = 'sampledf'

# Set the path to the directory where you want to save the train and test datasets
train_path = 'split/train'
test_path = 'split/test'

# Create the train and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# List all the folders in the dataset directory (assuming each folder represents a class)
class_folders = sorted(os.listdir(dataset_path))

# Iterate over each class folder
for class_folder in class_folders:
    class_folder_path = os.path.join(dataset_path, class_folder)

    # Get the list of images in the class folder
    images = os.listdir(class_folder_path)

    # Split the images into train and test sets
    train_images, test_images = train_test_split(images, train_size=600, test_size=200, random_state=42)

    # Create the train and test class folders in the train and test directories
    train_class_folder_path = os.path.join(train_path, class_folder)
    test_class_folder_path = os.path.join(test_path, class_folder)
    os.makedirs(train_class_folder_path, exist_ok=True)
    os.makedirs(test_class_folder_path, exist_ok=True)

    # Move the train images to the train class folder
    for train_image in train_images:
        src_path = os.path.join(class_folder_path, train_image)
        dest_path = os.path.join(train_class_folder_path, train_image)
        shutil.copy(src_path, dest_path)

    # Move the test images to the test class folder
    for test_image in test_images:
        src_path = os.path.join(class_folder_path, test_image)
        dest_path = os.path.join(test_class_folder_path, test_image)
        shutil.copy(src_path, dest_path)
