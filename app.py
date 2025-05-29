import os
import cv2
import numpy as np
import pickle
import urllib.request
import json
from time import sleep
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load pre-trained models
cnn_model = load_model('CNN.model')  # Replace with actual CNN model path
knn_model = pickle.load(open('knn_model.sav', 'rb'))  # Replace with actual KNN model path

# Load class names for CNN
data_dir = "data"
class_names = os.listdir(data_dir)

# Define symptoms and remedies for classes
disease_info = {
    "cocci": {
        "symptoms": ["Diarrhea", "Weight loss", "Blood in stool"],
        "remedy": "Administer anticoccidial drugs and provide clean, dry bedding."
    },
    "healthy": {
        "symptoms": ["No visible signs of illness"],
        "remedy": "Maintain a balanced diet and ensure proper hygiene."
    },
    "ncd": {  # Newcastle Disease
        "symptoms": ["Respiratory distress", "Twisted neck", "Paralysis"],
        "remedy": "Vaccination is key. Isolate infected birds and provide supportive care."
    },
    "salmo hen disease": {
        "symptoms": ["Fever", "Loss of appetite", "Green diarrhea"],
        "remedy": "Use antibiotics like amoxicillin. Improve sanitation and hygiene."
    }
}

# Function to process image and classify using CNN
def classify_cnn(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return "Error: Could not load image"

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
    class_name = class_names[class_index]

    # Fetch symptoms and remedies
    disease_data = disease_info.get(class_name.lower(), {"symptoms": ["Unknown"], "remedy": "No remedy available"})

    return class_name, disease_data

# Function to fetch ThingSpeak data and classify using KNN
def classify_knn():
    try:
        # Fetch latest data from ThingSpeak
        conn = urllib.request.urlopen("https://api.thingspeak.com/channels/565129/feeds.json?results=1")
        response = conn.read()
        data = json.loads(response)
        conn.close()

        # Extract value
        e = float(data['feeds'][0]['field5'])
        print(f"Extracted Value e: {e}")

        # Predict using KNN
        knn_result = knn_model.predict([[e]])[0]
        return e, knn_result

    except Exception as ex:
        return None, "Error fetching ThingSpeak data"

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    cnn_result, disease_data = classify_cnn(file_path)
    extracted_value, knn_result = classify_knn()

    return jsonify({
        'cnn_result': cnn_result,
        'symptoms': disease_data["symptoms"],
        'remedy': disease_data["remedy"],
        'extracted_value': extracted_value,
        'knn_result': knn_result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

