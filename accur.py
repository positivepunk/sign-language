import tensorflow as tf
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('Model/f/keras_model100.h5')

# Directory containing the test data
test_data_directory = 'split/test'

# Get the class names from the directory names
class_names = os.listdir(test_data_directory)

# Initialize x_test and y_test lists
x_test = []
y_test = []

# Iterate over the class names
for class_name in class_names:
    class_directory = os.path.join(test_data_directory, class_name)
    for filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, filename)
        # Load the image using cv2.imread
        image = cv2.imread(image_path)
        x_test.append(image)
        y_test.append(class_name)

# Directory containing the train data
train_data_directory = 'sampled500'

# Get the class names from the directory names
class_names = os.listdir(train_data_directory)

# Initialize x_train and y_train lists
x_train = []
y_train = []

# Iterate over the class names
for class_name in class_names:
    class_directory = os.path.join(train_data_directory, class_name)
    for filename in os.listdir(class_directory):
        image_path = os.path.join(class_directory, filename)
        # Load the image using cv2.imread
        image = cv2.imread(image_path)
        x_train.append(image)
        y_train.append(class_name)

# Convert the lists to numpy arrays
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)

# Normalize the pixel values
x_test = x_test / 255.0
x_train = x_train / 255.0

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Evaluate the model on the train data
train_loss, train_accuracy = model.evaluate(x_train, y_train)

# Print the performance metrics
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)
