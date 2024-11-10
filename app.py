from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the fish disease classification model
fish_classification_model = tf.keras.models.load_model('./model2 _RGB.keras')


if not os.path.exists('uploads'):
    os.makedirs('uploads')


def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)

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

# Route for fish disease classification


@app.route('/fish-classification', methods=['POST'])
def fish_classification():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    predictions = fish_classification_model.predict(img_array)
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


@app.route('/', methods=['GET'])
def index():
    return "Flask app is running"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
