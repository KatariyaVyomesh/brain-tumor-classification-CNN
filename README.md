# brain-tumor-classification-CNN
Deep Learning model for brain tumor detection using CNN on medical images. Classification images as 'Cancer' or 'No-Cancer'.

---------------------------------------------------------------------------------------------

# Brain Tumor Classification using CNN

This project is a deep learning-based approach to classify brain MRI images as either having a tumor (cancer) or not. The model uses a Convolutional Neural Network (CNN) to analyze medical images and predict if a brain tumor is present. It’s trained on images stored in separate folders for positive (tumor) and negative (no tumor) cases.

-----------------------------------------------------------------------------------------------
## Dataset Structure
Ensure the dataset is structured as follows:
brain_tumor_data/ ├── yes/       
# Contains 155 images labeled as "cancer" in JPG format 
├── no/        
# Contains 98 images labeled as "no cancer" in JPG format

-----------------------------------------------------------------------------------------------
# Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- PIL (Python Imaging Library)
Install the required packages using:pip install tensorflow numpy pillow

----------------------------------------------------------------------------------------------

# model Architecture
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras, consisting of:
-> 3 Convolutional layers with ReLU activation and MaxPooling
-> A Flatten layer to transform image data for classification
-> 2 Dense layers with a dropout layer to reduce overfitting
-> Binary output (cancer or no cancer) with a sigmoid activation

-------------------------------------------------------------------------------------------
# Training the Model
Run the following code to train the model:

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Define paths
train_dir = 'brain_tumor_data'  # Folder containing "yes" and "no" subfolders
# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
# Create training and validation data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
# Save the trained model
model.save('brain_tumor_classifier_model.h5')

-----------------------------------------------------------------------------------------

# Prediction on New Images
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('brain_tumor_classifier_model.h5')

def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Prediction: Cancer detected")
    else:
        print("Prediction: No cancer detected")


-----------------------------------------------------------------------------------------------
# Predict on a new image
To make predictions on a new image, load the trained model and use the following code:

from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
# Load the model
model = tf.keras.models.load_model('brain_tumor_classifier_model.h5')
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Prediction: Cancer detected")
    else:
        print("Prediction: No cancer detected")
# Predict on a new image
image_path = 'path_to_image.jpg'  # Replace with the path to your test image
predict_image(image_path, model)

-----------------------------------------------------------------------------------------------
# Results

The model's accuracy and performance will vary based on the training data and configuration. Experiment with hyperparameters like epochs, batch size, and network layers to improve results.

-----------------------------------------------------------------------------------------------
# License
This project is licensed under the MIT License - see the LICENSE file for details.
---
This README.md file provides an overview, installation instructions, dataset structure, code for training and prediction, and a section for the license. Adjust paths and hyperparameters based on your specific setup and requirements.
