"""Train a calibrated MobileNetV2 tomato-leaf classifier.

This version is designed for more reliable real-camera behaviour than a
plain softmax classifier.  It uses realistic augmentation, label smoothing,
gentler class balancing, checkpointed fine-tuning, and validation-set
temperature calibration.  It does not make a 10-class-only dataset recognise
non-leaf objects; add real leaf/background photos before retraining for that.

Expected layout:
    clean_dataset/train/<class>/*.jpg
    clean_dataset/val/<class>/*.jpg
    clean_dataset/test/<class>/*.jpg

Example:
    python train_model_v2.py --data clean_dataset --out tomato_model_v2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

IMAGE_SIZE = (224, 224)


def make_dataset(root: Path, class_names: list[str], batch_size: int, shuffle: bool) -> tf.data.Dataset:
    """Build one deterministic directory dataset."""
    return tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        class_names=class_names,
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    ).prefetch(tf.data.AUTOTUNE)


def class_weights(train_root: Path, class_names: list[str]) -> dict[int, float]:
    """Use square-root inverse frequency weights to avoid extreme rare-class bias."""
    counts = np.array([sum(1 for path in (train_root / name).iterdir() if path.is_file()) for name in class_names])
    if np.any(counts == 0):
        empty = [name for name, count in zip(class_names, counts) if count == 0]
        raise ValueError(f"Classes without training images: {empty}")
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    weights /= weights.mean()
    print("\nTraining images and class weights:")
    for name, count, weight in zip(class_names, counts, weights):
        print(f"  {name:48} {count:5d}  weight={weight:.2f}")
    return {index: float(weight) for index, weight in enumerate(weights)}


def build_model(class_count: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    # Batch-normalisation runs in inference mode to preserve ImageNet statistics.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf.keras.layers.Dropout(0.40, name="dropout")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="disease")(x)
    return tf.keras.Model(inputs, outputs, name="tomato_mobilenetv2_calibrated"), base


@tf.keras.utils.register_keras_serializable(package="tomato")
def smoothed_sparse_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Sparse cross entropy with 8% label smoothing to limit overconfidence."""
    label_smoothing = 0.08
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    class_count = tf.shape(y_pred)[-1]
    one_hot = tf.one_hot(y_true, class_count)
    smoothed = one_hot * (1.0 - label_smoothing) + label_smoothing / tf.cast(class_count, tf.float32)
    return tf.keras.losses.categorical_crossentropy(smoothed, y_pred)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss=smoothed_sparse_crossentropy,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
    )


def labels_from_dataset(train_root: Path) -> list[str]:
    labels = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not labels:
        raise ValueError(f"No class folders found in {train_root}")
    return labels


def dataset_labels(dataset: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([batch_labels.numpy() for _, batch_labels in dataset], axis=0)


def calibrated_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to softmax probabilities without changing class order."""
    log_probs = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    log_probs -= log_probs.max(axis=1, keepdims=True)
    scaled = np.exp(log_probs)
    return scaled / scaled.sum(axis=1, keepdims=True)


def find_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Choose a validation temperature that minimises negative log likelihood."""
    candidates = np.linspace(0.5, 3.0, 101)
    losses = []
    rows = np.arange(len(labels))
    for temperature in candidates:
        scaled = calibrated_probabilities(probabilities, float(temperature))
        losses.append(float(-np.mean(np.log(np.clip(scaled[rows, labels], 1e-8, 1.0)))))
    return float(candidates[int(np.argmin(losses))])


def threshold_for_precision(probabilities: np.ndarray, labels: np.ndarray, target_precision: float = 0.90) -> tuple[float, float, float]:
    """Return a threshold with the widest validation coverage meeting target precision."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    best = (0.90, 0.0, 0.0)  # threshold, precision, coverage
    for threshold in np.linspace(0.50, 0.98, 49):
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        precision = float(correct[accepted].mean())
        coverage = float(accepted.mean())
        if precision >= target_precision and coverage > best[2]:
            best = (float(threshold), precision, coverage)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root containing train, val, and test")
    parser.add_argument("--out", type=Path, default=Path("tomato_model_v2"), help="Output prefix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=18)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--unfreeze-layers", type=int, default=60)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    train_root, val_root, test_root = args.data / "train", args.data / "val", args.data / "test"
    for directory in (train_root, val_root, test_root):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing split directory: {directory}")

    labels = labels_from_dataset(train_root)
    train_ds = make_dataset(train_root, labels, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_root, labels, args.batch_size, shuffle=False)
    test_ds = make_dataset(test_root, labels, args.batch_size, shuffle=False)
    weights = class_weights(train_root, labels)

    prefix = args.out.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    label_path = prefix.with_name(f"{prefix.name}_labels.json")
    label_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    model, base = build_model(len(labels))

    head_checkpoint = prefix.with_name(f"{prefix.name}_head_best.keras")
    compile_model(model, 1e-3)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(head_checkpoint, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.35, patience=2, min_lr=1e-6),
        ],
    )
    model = tf.keras.models.load_model(head_checkpoint, compile=False)

    # Fine-tune only the later MobileNetV2 feature layers; leave BatchNorm fixed.
    base = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2")),
        None,
    )
    if base is None:
        raise RuntimeError("Could not find the MobileNetV2 feature extractor after loading the checkpoint.")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    fine_checkpoint = prefix.with_name(f"{prefix.name}_finetuned_best.keras")
    compile_model(model, 1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(fine_checkpoint, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.35, patience=2, min_lr=1e-7),
        ],
    )
    model = tf.keras.models.load_model(fine_checkpoint, compile=False)

    # Calibrate confidence using validation data only; test data remains untouched until reporting.
    val_probabilities = model.predict(val_ds, verbose=1)
    val_labels = dataset_labels(val_ds)
    temperature = find_temperature(val_probabilities, val_labels)
    calibrated_val = calibrated_probabilities(val_probabilities, temperature)
    threshold, validation_precision, coverage = threshold_for_precision(calibrated_val, val_labels)
    calibration_path = prefix.with_name(f"{prefix.name}_calibration.json")
    calibration_path.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "recommended_confidence_threshold": threshold,
                "validation_precision_at_threshold": validation_precision,
                "validation_coverage_at_threshold": coverage,
                "note": "Calibration improves confidence estimates only for images similar to validation data.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    test_probabilities = calibrated_probabilities(model.predict(test_ds, verbose=1), temperature)
    test_labels = dataset_labels(test_ds)
    test_predictions = test_probabilities.argmax(axis=1)
    report = classification_report(test_labels, test_predictions, target_names=labels, output_dict=True, zero_division=0)
    report_path = prefix.with_name(f"{prefix.name}_test_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with prefix.with_name(f"{prefix.name}_test_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for label in labels:
            row = report[label]
            writer.writerow([label, row["precision"], row["recall"], row["f1-score"], row["support"]])

    model_path = prefix.with_suffix(".keras")
    model.save(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path = prefix.with_suffix(".tflite")
    tflite_path.write_bytes(converter.convert())

    print(f"\nSaved model: {model_path}")
    print(f"Saved TFLite model: {tflite_path}")
    print(f"Saved labels: {label_path}")
    print(f"Saved calibration: {calibration_path}")
    print(f"Validation calibration: T={temperature:.2f}; threshold={threshold:.2f}; "
          f"precision={validation_precision:.1%}; coverage={coverage:.1%}")
    print(json.dumps({label: report[label] for label in labels}, indent=2))


if __name__ == "__main__":
    main()
