"""
Pet Image Classification - Inference Application & Web UI
Supports popular pet and animal categories from Kaggle Animals-10:
Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel.

Provides:
  1. Interactive Streamlit Web UI (`streamlit run main.py` or `python main.py`)
  2. Command-Line Interface (`python main.py --image <path>`)
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "pet_classifier.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
IMAGE_SIZE = (160, 160)

_CACHED_MODEL = None
_CACHED_CLASSES = None


def load_model_and_labels():
    """Loads and caches the trained classification model and class mappings."""
    global _CACHED_MODEL, _CACHED_CLASSES
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Please run 'python train.py' first to train the model."
        )

    if _CACHED_MODEL is None:
        _CACHED_MODEL = tf.keras.models.load_model(MODEL_PATH)

    if _CACHED_CLASSES is None:
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, "r") as f:
                _CACHED_CLASSES = json.load(f)
        else:
            _CACHED_CLASSES = {
                "0": "Butterfly", "1": "Cat", "2": "Chicken", "3": "Cow", "4": "Dog",
                "5": "Elephant", "6": "Horse", "7": "Sheep", "8": "Spider", "9": "Squirrel"
            }

    return _CACHED_MODEL, _CACHED_CLASSES


def predict_image(image_input):
    """
    Runs multi-class inference on an image (PIL Image or file path).
    Returns dict:
      {
         'class_name': str,
         'confidence': float (0-100),
         'top_predictions': list of (class_name, confidence),
         'all_probabilities': dict of {class_name: probability}
      }
    """
    model, classes = load_model_and_labels()

    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input

    # Ensure 3-channel RGB
    img = img.convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # Softmax output across classes
    raw_prediction = model.predict(img_batch, verbose=0)[0]

    # Handle multi-class softmax vs binary sigmoid output
    if len(raw_prediction) > 1:
        top_idx = int(np.argmax(raw_prediction))
        top_class = classes.get(str(top_idx), f"Class {top_idx}")
        top_confidence = float(raw_prediction[top_idx]) * 100.0

        all_probs = {}
        ranked_list = []
        for idx, prob in enumerate(raw_prediction):
            c_name = classes.get(str(idx), f"Class {idx}")
            c_pct = round(float(prob) * 100.0, 2)
            all_probs[c_name] = c_pct
            ranked_list.append((c_name, c_pct))

        ranked_list.sort(key=lambda x: x[1], reverse=True)

        return {
            "class_name": top_class,
            "confidence": round(top_confidence, 2),
            "top_predictions": ranked_list[:5],
            "all_probabilities": all_probs
        }
    else:
        dog_prob = float(raw_prediction[0])
        cat_prob = 1.0 - dog_prob
        top_class = "Dog" if dog_prob >= 0.5 else "Cat"
        top_conf = (dog_prob if dog_prob >= 0.5 else cat_prob) * 100.0
        return {
            "class_name": top_class,
            "confidence": round(top_conf, 2),
            "top_predictions": [("Dog", round(dog_prob * 100, 2)), ("Cat", round(cat_prob * 100, 2))],
            "all_probabilities": {"Dog": round(dog_prob * 100, 2), "Cat": round(cat_prob * 100, 2)}
        }


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_cli(image_path):
    """Command-line interface prediction function."""
    print("=" * 55)
    print("Pet Image Classification - Prediction")
    print("=" * 55)

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    try:
        result = predict_image(image_path)
        print(f"\nResult: [{result['class_name'].upper()}]")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nTop Ranked Class Probabilities:")
        for name, prob in result["top_predictions"]:
            print(f"  • {name:12s}: {prob:6.2f}%")
        print("=" * 55)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


def safe_image(img, caption=None):
    """Safely renders an image across all Streamlit versions (Streamlit Cloud & Local)."""
    import streamlit as st
    for kwargs in [
        {"use_container_width": True},
        {"use_column_width": True},
        {"width": "stretch"},
        {}
    ]:
        try:
            st.image(img, caption=caption, **kwargs)
            return
        except TypeError:
            continue


def run_streamlit_app():
    """Interactive Streamlit Web Dashboard for Pet Image Classification."""
    import streamlit as st

    st.set_page_config(
        page_title="Pet Image Classifier | Deep Learning",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Project Details")
        st.info("**Project**: Pet Image Classification\n\n"
                "**Backbone**: MobileNetV2 (Transfer Learning)\n\n"
                "**Dataset**: Kaggle & Wildlife Dataset (16 Classes)\n\n"
                "**Categories (16)**:\n\n"
                "Butterfly • Camel • Cat • Chicken\n\n"
                "Cow • Dog • Elephant • Goat\n\n"
                "Horse • Lion • Monkey • Rabbit\n\n"
                "Sheep • Spider • Squirrel • Tiger")

        if os.path.exists(MODEL_PATH):
            st.success("Model loaded: `pet_classifier.keras`")
        else:
            st.error("Model not trained yet! Run `python train.py` first.")

        st.divider()

        if os.path.exists(HISTORY_PLOT_PATH):
            st.subheader("Training History")
            safe_image(HISTORY_PLOT_PATH, caption="Accuracy & Loss Curves")

        st.caption("Internship ML Project • Abhishek Kushwaha • B.Tech CSE (CCML)")

    # Main Page Header
    st.markdown('<div class="main-header">Pet Image Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a real photograph of a pet or animal, or pick a sample from the dataset to test deep learning predictions.</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Predict Pet", "Pet Dataset Gallery", "Model Architecture & Benchmark"])

    with tabs[0]:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("1. Provide an Image")
            input_mode = st.radio(
                "Choose input method:",
                ["Upload Image File", "Select Sample from Dataset"],
                horizontal=True
            )

            selected_image = None
            image_name = ""

            # Load active category folders
            _, label_map = load_model_and_labels()
            if label_map:
                cat_options = [c.lower() for c in label_map.values() if os.path.isdir(os.path.join(CURRENT_DIR, c.lower()))]
            else:
                cat_options = [
                    c for c in [
                        "butterfly", "camel", "cat", "chicken", "cow", "dog",
                        "elephant", "goat", "horse", "lion", "monkey",
                        "rabbit", "sheep", "spider", "squirrel", "tiger"
                    ]
                    if os.path.isdir(os.path.join(CURRENT_DIR, c))
                ]

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Choose an animal photo (JPG, JPEG, PNG)...",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    image_name = uploaded_file.name
            else:
                sample_category = st.selectbox(
                    "Select animal category:",
                    cat_options,
                    format_func=lambda x: x.capitalize()
                )
                sample_files = sorted(glob.glob(os.path.join(CURRENT_DIR, sample_category, "*.[jJ][pP][gG]")))

                if sample_files:
                    selected_path = st.selectbox(
                        "Pick a sample image:",
                        sample_files,
                        format_func=lambda p: os.path.basename(p)
                    )
                    if selected_path:
                        selected_image = Image.open(selected_path)
                        image_name = f"{sample_category}/{os.path.basename(selected_path)}"

            if selected_image is not None:
                safe_image(selected_image, caption=f"Selected: {image_name}")

        with col2:
            st.subheader("2. Classification Results")
            if selected_image is not None:
                if not os.path.exists(MODEL_PATH):
                    st.error("Model file not found. Please run `python train.py` first.")
                else:
                    with st.spinner("Analyzing visual features with MobileNetV2..."):
                        res = predict_image(selected_image)

                    badge_color = "#059669" if res["confidence"] > 75 else "#d97706"

                    st.markdown(f"""
                        <div style="background:#f8fafc; border: 2px solid #e2e8f0; border-radius:12px; padding: 20px; margin-bottom: 20px;">
                            <div style="font-size: 14px; text-transform: uppercase; color: #64748b; font-weight: 600;">Top Prediction</div>
                            <div style="font-size: 34px; font-weight: 800; color: #1e293b; margin: 8px 0;">
                                {res['class_name'].upper()}
                            </div>
                            <div style="font-size: 17px; color: {badge_color}; font-weight: 700;">
                                Confidence: {res['confidence']:.2f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Probability Distribution (Top Ranked)")
                    for c_name, c_prob in res["top_predictions"]:
                        col_l, col_r = st.columns([2, 1])
                        with col_l:
                            st.write(f"**{c_name}**")
                        with col_r:
                            st.write(f"**{c_prob:.2f}%**")
                        st.progress(float(min(c_prob / 100.0, 1.0)))

                    if res["confidence"] < 60.0:
                        st.warning("Low Confidence Warning: The model confidence is under 60%. The image might belong to an animal category outside the 16 trained classes, or the picture angle/lighting may be ambiguous.")

            else:
                st.info("Upload an image or choose a dataset sample to view classification results.")

    with tabs[1]:
        st.subheader("Pet Dataset Gallery")
        st.caption(f"Showing sample real photographs from each of the {len(cat_options)} animal classes in the project dataset.")

        for c in cat_options:
            c_samples = sorted(glob.glob(os.path.join(CURRENT_DIR, c, "*.[jJ][pP][gG]")))[:4]
            if c_samples:
                st.markdown(f"##### {c.capitalize()} ({c}/)")
                cols = st.columns(4)
                for idx, p in enumerate(c_samples):
                    with cols[idx]:
                        safe_image(p, caption=os.path.basename(p))
                st.divider()

    with tabs[2]:
        st.subheader("Model Architecture & Technical Engineering")
        st.markdown("""
        - **Dataset Benchmark**: Multi-class animal dataset spanning **16 distinct animal categories** (Butterfly, Camel, Cat, Chicken, Cow, Dog, Elephant, Goat, Horse, Lion, Monkey, Rabbit, Sheep, Spider, Squirrel, Tiger).
        - **Transfer Learning Backbone**: Pre-trained Google MobileNetV2 architecture with ImageNet feature representations.
        - **Data Augmentation**: Runtime `RandomFlip("horizontal")`, `RandomRotation(0.15)`, and `RandomZoom(0.1)` layers to boost generalization.
        - **Two-Phase Multi-Class Training**:
            1. **Feature Extraction**: Base model frozen; custom classification head (`GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(16, softmax)`) trained with Adam.
            2. **Fine-Tuning**: Unfreezes the top 30 layers with micro learning rate (`1e-5`) for precise domain adaptation.
        - **Loss Function**: `SparseCategoricalCrossentropy` for multi-class probability alignment.
        """)


def main():
    parser = argparse.ArgumentParser(description="Pet Image Classification")
    parser.add_argument("--image", type=str, help="Path to an image to classify via CLI")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application")
    args, unknown = parser.parse_known_args()

    # If --image passed, run CLI prediction
    if args.image:
        run_cli(args.image)
        return

    # If run through streamlit command: 'streamlit run main.py'
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_RUN") == "1":
        run_streamlit_app()
    else:
        # Check if user invoked 'python main.py' directly without args
        import subprocess
        print("Launching Streamlit Web Application...")
        print("Opening browser interface at http://localhost:8502")
        env = os.environ.copy()
        env["STREAMLIT_RUN"] = "1"
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--server.port", "8502"], env=env)
        except KeyboardInterrupt:
            print("\nShutting down application.")


if __name__ == "__main__":
    main()
