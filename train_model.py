"""
train_model.py

Trains a tomato leaf disease classifier using transfer learning on
MobileNetV2 (small + fast enough to later export to TFLite for a
Raspberry Pi).

Expects the folder structure produced by clean_dataset.py:

    clean_dataset/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

Usage:
    pip install tensorflow pillow scikit-learn
    python train_model.py --data clean_dataset --epochs 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# Access Keras through the TensorFlow namespace so static analyzers do not
# require separate tensorflow.keras source stubs.
layers = tf.keras.layers
models = tf.keras.models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def compute_class_weights(train_dir: Path, class_names: list) -> dict:
    """Inverse-frequency class weights so small classes (e.g. mosaic virus)
    aren't drowned out by big ones (e.g. yellow leaf curl virus)."""
    counts = np.array([
        len(list((train_dir / name).glob("*.jpg"))) for name in class_names
    ], dtype=np.float32)

    total = counts.sum()
    n_classes = len(class_names)
    weights = total / (n_classes * counts)

    print("\nClass weights (higher = rarer class, weighted more heavily):")
    for name, count, weight in zip(class_names, counts, weights):
        print(f"  {name:45s} n={int(count):5d}  weight={weight:.2f}")

    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze for stage 1

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ])

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned/split dataset root")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--out", default="tomato_model", help="Output model name (no extension)")
    args = parser.parse_args()

    data_root = Path(args.data)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "val", image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    with open(f"{args.out}_labels.json", "w") as f:
        json.dump(class_names, f, indent=2)

    class_weights = compute_class_weights(data_root / "train", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    model, base = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n--- Stage 1: training classifier head (base frozen) ---")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weights,
    )

    print("\n--- Stage 2: fine-tuning top layers of MobileNetV2 ---")
    base.trainable = True
    # freeze the earlier, more generic layers; only fine-tune the later ones
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
        class_weight=class_weights,
    )

    print("\n--- Evaluating on test set ---")
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}")

    print("\n--- Per-class report on test set ---")
    y_true, y_pred = [], []
    for images, labels_batch in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    model.save(f"{args.out}.keras")
    print(f"\nSaved model to {args.out}.keras and labels to {args.out}_labels.json")

    # Also export a TFLite version for the Raspberry Pi
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(f"{args.out}.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {args.out}.tflite (use this on the Pi)")


if __name__ == "__main__":
    main()
