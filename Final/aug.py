import keras.preprocessing.image
import os

from keras.utils import load_img, img_to_array

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='constant',
cval=255
)


# Set the default input folder name
input_folder = os.path.join('newv')

# Set the default output folder name
output_folder = 'newva'

# Get a list of subfolders (targets) in the input folder
subfolders = os.listdir(input_folder)

for subfolder in subfolders:
    # Construct the image folder path for the current subfolder
    image_folder = os.path.join(input_folder, subfolder)

    # Set the output subfolder name
    subfolder_name = subfolder

    # Set the prefix name based on the subfolder name
    prefix_name = subfolder

    # Set the output folder path
    output_folder_path = os.path.join(output_folder, subfolder_name)

    # Get a list of image file paths in the current subfolder
    image_files = os.listdir(image_folder)[:40]  # Get the first 150 files

    for image_file in image_files:
        # Load the image
        img = load_img(os.path.join(image_folder, image_file))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder_path, exist_ok=True)

        # Apply augmentation and save the transformed images
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_folder_path, save_prefix=prefix_name, save_format='jpg'):
            i += 1
            if i > 5:
                break

    print("Done with", subfolder)
