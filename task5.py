# Import libraries
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define paths
dataset_dir = 'D:\\New Folder\\food-101'  # Replace with actual path
image_dir = os.path.join(dataset_dir, 'images')

# Preprocess image data using ImageDataGenerator
# Reduce target size to 128x128 to speed up training
image_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2  # Split 20% for validation
)

# Training data generator with reduced image size (128x128)
train_data = image_gen.flow_from_directory(
    image_dir,
    target_size=(128, 128),  # Smaller size for faster processing
    batch_size=64,  # Larger batch size to speed up training
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_data = image_gen.flow_from_directory(
    image_dir,
    target_size=(128, 128),  # Smaller image size for validation as well
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# List of food classes (101 food items)
food_classes = list(train_data.class_indices.keys())

# Load MobileNetV2 base model with more frozen layers for faster training
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet', alpha=0.35)
base_model.trainable = False  # Freeze entire base model

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduce the output dimensions
    Dense(256, activation='relu'),  # Reduce dense layer size for faster computation
    Dropout(0.3),  # Add dropout to prevent overfitting
    Dense(len(food_classes), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Optimizer with moderate learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (reduce the epochs to 5 for faster iteration)
history = model.fit(train_data, epochs=3 , validation_data=val_data, verbose=1)

# Save the trained model in TensorFlow's recommended format (.keras)
model.save('food_recognition_model.keras')

# Plot accuracy and loss graphs
if history.history:
    # Plot accuracy graph
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Create a dictionary to map food items to calorie content
calorie_dict = {
    'apple_pie': 300,
    'baby_back_ribs': 400,
    'baklava': 500,
    # Add more food items and their approximate calorie values here
}

# Function to estimate calories
def estimate_calories(food_item):
    return calorie_dict.get(food_item, "Unknown")

# Load and predict on new images
def predict_food_and_calories(image_path):
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        return None, None

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))  # Smaller image size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict the class of the food
    prediction = model.predict(img_array)
    predicted_class = food_classes[np.argmax(prediction)]

    # Estimate calories
    estimated_calories = estimate_calories(predicted_class)

    return predicted_class, estimated_calories

# Example usage
image_path = 'D:\\New Folder\\food-101\\images\\apple_pie\\134.jpg'  # Replace with the path to an image for testing
predicted_food, calories = predict_food_and_calories(image_path)

if predicted_food and calories:
    print(f"Predicted Food: {predicted_food}, Estimated Calories: {calories}")
else:
    print("Prediction failed due to image path issues or other errors.")
