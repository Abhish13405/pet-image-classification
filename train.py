"""
16-Animal Image Classification - Model Training Pipeline
Trains a transfer learning model (MobileNetV2) with data augmentation
to classify 16 popular animal categories:
Butterfly, Camel, Cat, Chicken, Cow, Dog, Elephant, Goat, Horse, Lion, Monkey, Rabbit, Sheep, Spider, Squirrel, Tiger.
"""

import os
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "pet_classifier.keras")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
CLASSES_JSON_PATH = os.path.join(MODEL_DIR, "classes.json")

# 16 Animal Categories (Expanded with Rabbit, Goat, Lion, Tiger, Monkey, Camel)
TARGET_CLASSES = [
    "butterfly", "camel", "cat", "chicken", "cow", "dog",
    "elephant", "goat", "horse", "lion", "monkey",
    "rabbit", "sheep", "spider", "squirrel", "tiger"
]


def load_dataset(base_dir):
    """
    Scans each animal directory, creates stratified balanced train and validation splits.
    """
    class_names = []
    for c in TARGET_CLASSES:
        d = os.path.join(base_dir, c)
        if os.path.isdir(d):
            imgs = glob.glob(os.path.join(d, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(d, "*.[pP][nN][gG]"))
            if len(imgs) > 0:
                class_names.append(c)

    class_names = sorted(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {str(idx): name.capitalize() for idx, name in enumerate(class_names)}

    train_files, train_labels = [], []
    val_files, val_labels = [], []

    print(f"Detected {len(class_names)} active classes: {class_names}")

    for c in class_names:
        folder = os.path.join(base_dir, c)
        files = glob.glob(os.path.join(folder, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(folder, "*.[pP][nN][gG]"))
        random.shuffle(files)

        split_idx = int(len(files) * 0.8)
        c_train = files[:split_idx]
        c_val = files[split_idx:]

        label_idx = class_to_idx[c]
        train_files.extend(c_train)
        train_labels.extend([label_idx] * len(c_train))

        val_files.extend(c_val)
        val_labels.extend([label_idx] * len(c_val))

        print(f"  • {c:10s}: {len(c_train)} train, {len(c_val)} val (Total: {len(files)})")

    # Shuffle train and val pairs
    train_combined = list(zip(train_files, train_labels))
    random.shuffle(train_combined)
    train_files, train_labels = zip(*train_combined)

    val_combined = list(zip(val_files, val_labels))
    random.shuffle(val_combined)
    val_files, val_labels = zip(*val_combined)

    print(f"\nTotal Training Samples:   {len(train_files)}")
    print(f"Total Validation Samples: {len(val_files)}")

    return list(train_files), list(train_labels), list(val_files), list(val_labels), idx_to_class


def preprocess_image(file_path, label):
    """Reads image file, decodes JPEG, and resizes to target shape."""
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, label


def create_tf_dataset(files, labels, is_training=True):
    """Builds an optimized tf.data.Dataset pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(files), seed=SEED)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes):
    """
    Builds a Transfer Learning model using MobileNetV2 with Data Augmentation
    configured for multi-class classification.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="AnimalClassifier_10Classes")
    return model, base_model


def plot_and_save_history(history_initial, history_fine, save_path):
    """Plots multi-class training and validation accuracy and loss over all epochs."""
    acc = history_initial.history["accuracy"] + (history_fine.history["accuracy"] if history_fine else [])
    val_acc = history_initial.history["val_accuracy"] + (history_fine.history["val_accuracy"] if history_fine else [])

    loss = history_initial.history["loss"] + (history_fine.history["loss"] if history_fine else [])
    val_loss = history_initial.history["val_loss"] + (history_fine.history["val_loss"] if history_fine else [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2)
    if history_fine:
        plt.axvline(x=INITIAL_EPOCHS, color="gray", linestyle="--", label="Start Fine-Tuning")
    plt.title("Multi-Class Classification Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ea580c", linewidth=2)
    if history_fine:
        plt.axvline(x=INITIAL_EPOCHS, color="gray", linestyle="--", label="Start Fine-Tuning")
    plt.title("Sparse Categorical Cross-Entropy Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 65)
    print("16-Animal Image Classification - Multi-Class Training Pipeline")
    print("=" * 65)

    # 1. Prepare data
    train_files, train_labels, val_files, val_labels, idx_to_class = load_dataset(CURRENT_DIR)
    num_classes = len(idx_to_class)
    
    train_ds = create_tf_dataset(train_files, train_labels, is_training=True)
    val_ds = create_tf_dataset(val_files, val_labels, is_training=False)

    # 2. Build model
    print(f"\nBuilding MobileNetV2 Multi-Class ({num_classes} categories) Model...")
    model, base_model = build_model(num_classes)
    model.summary()

    # Phase 1: Feature Extraction
    print("\n[Phase 1] Training top classification head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        verbose=1
    )

    # Phase 2: Fine-Tuning top 30 layers
    print("\n[Phase 2] Fine-tuning upper layers of MobileNetV2...")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        verbose=1
    )

    # 3. Final Evaluation
    print("\n" + "=" * 65)
    print("Final Model Evaluation on Validation Set:")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print("=" * 65)

    # 4. Save Model and Metadata
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    with open(CLASSES_JSON_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=4)
    print(f"Class labels saved to {CLASSES_JSON_PATH}: {idx_to_class}")

    # 5. Plot history
    plot_and_save_history(history_phase1, history_phase2, HISTORY_PLOT_PATH)

    print("\nTraining completed successfully!")
    return model, val_acc


if __name__ == "__main__":
    train()
