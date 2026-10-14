from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tomato_model.keras"
LABELS_PATH = BASE_DIR / "tomato_model_labels.json"

IMAGE_SIZE = (224, 224)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ML model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as f:

    labels = json.load(f)


print("Classes:")

for i, label in enumerate(labels):
    print(f"  {i}: {label}")


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    # IMPORTANT:
    #
    # Do NOT divide by 255.
    #
    # Your training model contains:
    #
    # mobilenet_v2.preprocess_input(x)
    #
    # inside the model itself.

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    image_array = preprocess_image(
        image
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    predicted_label = labels[
        predicted_index
    ]

    # Create all predictions
    all_predictions = []

    for i, probability in enumerate(
        predictions
    ):

        all_predictions.append({
            "label": labels[i],
            "confidence": float(
                probability
            )
        })

    # Highest probability first
    all_predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "predictions": all_predictions
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "No image selected."
            }), 400

        # Open image
        image = Image.open(
            file.stream
        )

        result = predict_image(
            image
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       TOMATO PLANT HEALTH AI")
    print("=" * 60)
    print()
    print(
        "Open your browser at:"
    )
    print(
        "http://127.0.0.1:5000"
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )