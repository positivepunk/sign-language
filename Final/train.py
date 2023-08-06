# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import save_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 300

# Step 1 - Building the CNN
# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=26, activation='softmax'))  # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])  # categorical_crossentropy for more than 2

# Step 2 - Preparing the train/test data and training the model
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('split/train',
                                                 target_size=(sz, sz),
                                                 batch_size=10,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('split/test',
                                            target_size=(sz, sz),
                                            batch_size=10,
                                            class_mode='categorical')

# Calculate the steps_per_epoch and validation_steps based on the number of images
steps_per_epoch = 600  # 600 training images, batch_size = 10
validation_steps = 200  # 200 testing images, batch_size = 10

classifier.fit_generator(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=test_set,
    validation_steps=validation_steps)

# Saving the model
save_model(classifier, "model-rgb.h5")
print('Model Saved')
