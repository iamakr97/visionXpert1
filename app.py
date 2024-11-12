from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load ImageNet model for object detection
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the rose disease classification model
rose_classification_model = tf.keras.models.load_model(
    './models/rose_disease.keras')

# Load the Honey-bee classification model
honeybee_classification_model = tf.keras.models.load_model(
    './models/honey_bee.keras')


if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Image preprocessing for MobileNetV2 (ImageNet)
def prepare_image_for_mobilenet(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Image preprocessing for custom models
def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# Object classification route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img_array = prepare_image_for_mobilenet(filepath)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions, top=1)

    result = {
        "filename": file.filename,
        "predictions": [
            {
                "class_id": pred[0],
                "class_name": pred[1],
                "confidence": float(pred[2])
            } for pred in decoded_predictions[0]
        ]
    }

    # Delete the uploaded file after processing
    try:
        os.remove(filepath)
    except OSError as e:
        print(f"Error deleting file {filepath}: {e}")

    return jsonify(result), 200


# Route for Rose disease classification
@app.route('/rose-disease-classification', methods=['POST'])
def rose_disease_classification():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    predictions = rose_classification_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    result = {
        "filename": file.filename,
        "predicted_class": int(predicted_class),
        "confidence": confidence
    }

    # Delete the uploaded file after processing
    try:
        os.remove(filepath)
    except OSError as e:
        print(f"Error deleting file {filepath}: {e}")

    return jsonify(result), 200


# Route for Honeybee classification
@app.route('/honeybee-classification', methods=['POST'])
def honeybee_classification():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    predictions = honeybee_classification_model.predict(img_array)


    class_names = [
        "Varroa, Small Hive Beetles",
        "Ant problems",
        "Few varroa, hive beetles",
        "Healthy",
        "Hive being robbed",
        "Missing queen"
    ]

    # Get predicted class index and confidence
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class_index])
    predicted_class_name = class_names[predicted_class_index]

    # Build the result dictionary
    result = {
        "filename": file.filename,
        "predicted_class_index": int(predicted_class_index),
        "predicted_class_name": predicted_class_name,
        "confidence": confidence,
    }

    # Delete the uploaded file after processing
    try:
        os.remove(filepath)
    except OSError as e:
        print(f"Error deleting file {filepath}: {e}")

    return jsonify(result), 200



# Home route
@app.route('/', methods=['GET'])
def index():
    return "Flask app is running"


if __name__ == '__main__':
    app.run()
