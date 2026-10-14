import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageGrab
import tkinter as tk
from tkinter import filedialog


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(label_path):
    """Load class labels in the exact order used during training."""

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        try:
            return [data[str(i)] for i in range(len(data))]
        except (KeyError, TypeError):
            return list(data.values())

    raise ValueError("Unsupported labels JSON format.")


# ============================================================
# GET IMAGE
# ============================================================

def get_image_from_clipboard():
    """Try to get an image directly from the Windows clipboard."""

    try:
        clipboard = ImageGrab.grabclipboard()
    except Exception:
        return None

    if isinstance(clipboard, Image.Image):
        return clipboard

    return None


def choose_image_file():
    """Open a file picker."""

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select a tomato leaf image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not file_path:
        return None

    return Path(file_path)


def get_input_image(image_argument):
    """
    Priority:
    1. --image argument
    2. Clipboard image
    3. File picker
    """

    # --------------------------------------------------------
    # Image path supplied
    # --------------------------------------------------------

    if image_argument:

        image_path = Path(image_argument)

        if not image_path.exists():
            raise FileNotFoundError(
                f"\nImage not found:\n{image_path}\n"
            )

        print(f"\nUsing image:")
        print(image_path)

        image = Image.open(image_path).convert("RGB")

        return image, image_path

    # --------------------------------------------------------
    # Clipboard
    # --------------------------------------------------------

    print("\nChecking clipboard...")

    clipboard_image = get_image_from_clipboard()

    if clipboard_image is not None:

        print("Image found in clipboard.")

        clipboard_path = Path("_clipboard_leaf.png")

        clipboard_image.convert("RGB").save(
            clipboard_path
        )

        return clipboard_image.convert("RGB"), clipboard_path

    # --------------------------------------------------------
    # File picker
    # --------------------------------------------------------

    print("No image found in clipboard.")
    print("Opening image selector...")

    selected_path = choose_image_file()

    if selected_path is None:
        raise ValueError(
            "\nNo image selected.\n"
            "Copy a leaf image or select an image file."
        )

    print(f"\nSelected image:")
    print(selected_path)

    image = Image.open(
        selected_path
    ).convert("RGB")

    return image, selected_path


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Prepare image for the trained model.

    IMPORTANT:
    Do NOT divide by 255 here.

    MobileNetV2 preprocess_input() is already INSIDE
    the trained model.
    """

    image = image.resize(
        IMAGE_SIZE
    )

    # Keep pixel values in 0-255 range.
    #
    # This is intentional because the model itself contains:
    #
    # tf.keras.applications.mobilenet_v2.preprocess_input(x)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# TFLITE PREDICTION
# ============================================================

def predict_tflite(
    model_path,
    image_array
):
    """Run prediction using TensorFlow Lite."""

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_index = (
        input_details[0]["index"]
    )

    output_index = (
        output_details[0]["index"]
    )

    input_dtype = (
        input_details[0]["dtype"]
    )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    if input_dtype == np.float32:

        # IMPORTANT:
        # Send 0-255 values because the model contains
        # MobileNetV2 preprocess_input().
        input_data = image_array.astype(
            np.float32
        )

    else:

        scale, zero_point = (
            input_details[0]["quantization"]
        )

        if scale == 0:

            input_data = image_array.astype(
                input_dtype
            )

        else:

            input_data = (
                image_array / scale
            ) + zero_point

            input_data = np.round(
                input_data
            ).astype(input_dtype)

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        input_data
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_index
    )

    predictions = np.squeeze(
        predictions
    )

    # --------------------------------------------------------
    # Dequantize output if necessary
    # --------------------------------------------------------

    output_dtype = (
        output_details[0]["dtype"]
    )

    if output_dtype in (
        np.uint8,
        np.int8
    ):

        scale, zero_point = (
            output_details[0]["quantization"]
        )

        if scale != 0:

            predictions = (
                predictions.astype(
                    np.float32
                ) - zero_point
            ) * scale

    return predictions


# ============================================================
# KERAS PREDICTION
# ============================================================

def predict_keras(
    model_path,
    image_array
):
    """Run prediction using the Keras model."""

    model = tf.keras.models.load_model(
        model_path
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    return np.squeeze(
        predictions
    )


# ============================================================
# CONVERT TO PROBABILITIES
# ============================================================

def get_probabilities(
    predictions
):
    """
    The model already uses softmax, so normally the output
    is already probabilities.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # Model uses Dense(..., activation="softmax"),
    # so this should already sum to approximately 1.
    total = np.sum(predictions)

    if (
        np.all(predictions >= 0)
        and
        np.all(predictions <= 1)
        and
        np.isclose(
            total,
            1.0,
            atol=0.05
        )
    ):
        return predictions

    # Safety fallback
    return tf.nn.softmax(
        predictions
    ).numpy()


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    image_path,
    predicted_label,
    confidence,
    labels,
    probabilities
):

    print()
    print("=" * 65)
    print("              TOMATO LEAF ML DETECTOR")
    print("=" * 65)

    print(
        f"Image      : {image_path}"
    )

    print(
        f"Prediction : {predicted_label}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    print("=" * 65)

    print()
    print("All predictions:")
    print("-" * 65)

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    for index in sorted_indices:

        index = int(index)

        if index < len(labels):

            label = labels[index]

        else:

            label = f"Class {index}"

        probability = float(
            probabilities[index]
        )

        print(
            f"{label:<45}"
            f"{probability * 100:7.2f}%"
        )

    print()
    print("=" * 65)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tomato leaf disease detection "
            "using the trained MobileNetV2 model."
        )
    )

    parser.add_argument(
        "--image",
        required=False,
        default=None,
        help=(
            "Image path. If omitted, clipboard "
            "or file picker is used."
        )
    )

    parser.add_argument(
        "--model",
        required=False,
        default="tomato_model.tflite",
        help="Path to .tflite or .keras model."
    )

    parser.add_argument(
        "--labels",
        required=False,
        default="tomato_model_labels.json",
        help="Path to labels JSON."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_path = Path(
        args.model
    )

    labels_path = Path(
        args.labels
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}"
        )

    if not labels_path.exists():

        raise FileNotFoundError(
            f"\nLabels file not found:\n{labels_path}"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = load_labels(
        labels_path
    )

    print()
    print("=" * 65)
    print("          TOMATO LEAF DISEASE DETECTION")
    print("=" * 65)

    print(
        f"Model  : {model_path}"
    )

    print(
        f"Classes: {len(labels)}"
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image, image_path = get_input_image(
        args.image
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    image_array = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("Running MobileNetV2 model...")
    print()

    if model_path.suffix.lower() == ".tflite":

        predictions = predict_tflite(
            model_path,
            image_array
        )

    elif model_path.suffix.lower() in (
        ".keras",
        ".h5"
    ):

        predictions = predict_keras(
            model_path,
            image_array
        )

    else:

        raise ValueError(
            "Unsupported model format. "
            "Use .tflite, .keras or .h5."
        )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = get_probabilities(
        predictions
    )

    # --------------------------------------------------------
    # Best class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    if predicted_index < len(labels):

        predicted_label = labels[
            predicted_index
        ]

    else:

        predicted_label = (
            f"Class {predicted_index}"
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    display_result(
        image_path,
        predicted_label,
        confidence,
        labels,
        probabilities
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()