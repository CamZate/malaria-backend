from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)  # allow frontend JS to access backend

# Load the model once when the server starts
MODEL_PATH = "malaria_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# Define label names (1 = Uninfected, 0 = Infected)
CLASS_NAMES = ["Infected", "Uninfected"]

@app.route("/")
def home():
    return "Malaria Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)[0][0]
    pred_class = CLASS_NAMES[int(pred > 0.5)]
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return jsonify({
        "prediction": pred_class,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets the PORT env var
    app.run(host="0.0.0.0", port=port)

