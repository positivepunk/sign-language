from keras.models import model_from_json
from PIL import Image
import numpy as np

sz = 300

# Load the saved model and its weights
with open("Model/model-rgb.json", "r") as json_file:
    loaded_model_json = json_file.read()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("Model/model-rgb.h5")

# Load class labels
with open("Model/labels.txt", "r") as labels_file:
    class_labels = labels_file.read().splitlines()


# Function to classify an input image
def classify_image(input_image_path):
    # Preprocess the input image
    input_image = Image.open(input_image_path).resize((sz, sz))
    input_image = np.array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0

    # Make predictions
    predictions = classifier.predict(input_image)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class_index]

    return predicted_class_name


# Test the classifier with an input image
input_image_path = "new/Data/X/Image_1683038975.68814.jpg"  # Replace with the actual path to the input image
predicted_class = classify_image(input_image_path)
print("Predicted class:", predicted_class)
