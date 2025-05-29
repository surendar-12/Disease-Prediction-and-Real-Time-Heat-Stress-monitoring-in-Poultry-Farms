import tkinter as tk
from tkinter import filedialog
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle
import urllib.request
import json
from time import sleep

# Load pre-trained models
cnn_model = load_model('CNN.model')  # Replace with your CNN model path
knn_model = pickle.load(open('knn_model.sav', 'rb'))  # Load saved KNN model

# Load class names for CNN
data_dir = "data"
class_names = os.listdir(data_dir)

# Function to classify an image using CNN
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            cnn_result_label.config(text="Error: Could not load image")
            return
        
        # Preprocess image
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        edges = cv2.Canny(img_array, threshold1=50, threshold2=150)
        edges_colored = cv2.merge([edges, edges, edges])

        input_shape = (50, 50)  # Ensure this matches your model input shape
        new_array = cv2.resize(edges_colored, input_shape)
        new_array = new_array / 255.0  # Normalize
        new_array = np.expand_dims(new_array, axis=0)  # Add batch dimension

        # Make CNN prediction
        predictions = cnn_model.predict(new_array)
        class_index = np.argmax(predictions)
        cnn_result = class_names[class_index]

        # Display CNN result
        cnn_result_label.config(text=f'CNN Result: {cnn_result}')
        
        # Fetch ThingSpeak data
        fetch_thingspeak_data()

# Function to fetch values from ThingSpeak and classify using KNN
def fetch_thingspeak_data():
    knn_result_label.config(text="Waiting for ThingSpeak data...")

    while True:
        try:
            # Fetch latest data from ThingSpeak
            conn = urllib.request.urlopen("https://api.thingspeak.com/channels/565129/feeds.json?results=1")
            response = conn.read()
            data = json.loads(response)
            conn.close()

            # Extract latest value
            e = float(data['feeds'][0]['field5'])
            print(f"Extracted value e: {e}")
            extracted_value_label.config(text=f'Temp Value: {e}')

            # Predict using KNN
            knn_result = knn_model.predict([[e]])[0]
            print(f"KNN Result: {knn_result}")

            # Display KNN result
            knn_result_label.config(text=f'KNN Result: {knn_result}')
            break

        except Exception as ex:
            knn_result_label.config(text="Error fetching ThingSpeak data. Retrying...")
            sleep(5)  # Wait before retrying

# Create a tkinter GUI window
root = tk.Tk()
root.title("Hen Disease Classification")

# GUI elements
title_label = tk.Label(root, text="HEN’s Diseases Classification System", font=("Helvetica", 20))
title_label.pack(pady=20)

cnn_result_label = tk.Label(root, text="CNN Result: ", font=("Helvetica", 16))
cnn_result_label.pack(pady=10)

extracted_value_label = tk.Label(root, text="Extracted Value e: ", font=("Helvetica", 16))
extracted_value_label.pack(pady=10)

knn_result_label = tk.Label(root, text="KNN Result: ", font=("Helvetica", 16))
knn_result_label.pack(pady=10)

classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack(pady=10)

quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack(pady=10)

# Start GUI main loop
root.mainloop()