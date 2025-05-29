import tkinter as tk
from tkinter import filedialog
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load your pre-trained model ('CNN.model')
model = load_model('CNN.model')  # Replace with the correct path to your CNN model
data_dir = "training"
class_names = os.listdir(data_dir)

# Function to classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the selected image
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # FIXED
        
        if img_array is None:
            result_label.config(text="Error: Could not load image")
            return
        
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)  # Apply Gaussian Blur to reduce noise
        edges = cv2.Canny(img_array, threshold1=50, threshold2=150)

        # Convert single-channel edge-detected image to 3 channels (to match input_shape)
        edges_colored = cv2.merge([edges, edges, edges])

        input_shape = (50, 50)  # Ensure this matches your model's input shape
        new_array = cv2.resize(edges_colored, input_shape)  # FIXED

        # Normalize and reshape image for model input
        new_array = new_array / 255.0  # Normalize pixel values
        new_array = np.expand_dims(new_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(new_array)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]

        result_label.config(text=f'Result: {class_label}')

# Create a tkinter window
root = tk.Tk()
root.title("Leaf Pest Classifier")

# Create a label for the title
title_label = tk.Label(root, text="Classification", font=("Helvetica", 20))
title_label.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a button to select an image
classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack()

# Create a quit button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Start the tkinter main loop
root.mainloop()
