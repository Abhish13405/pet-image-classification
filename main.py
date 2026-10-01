"""
Pet Image Classification - Inference Application & Web UI
Provides:
  1. Interactive Streamlit Web UI (Streamlit mode or `python main.py`)
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
    """Loads and caches the trained pet classification model and class mappings."""
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
            _CACHED_CLASSES = {"0": "Cat", "1": "Dog"}

    return _CACHED_MODEL, _CACHED_CLASSES


def predict_image(image_input):
    """
    Runs inference on an image (PIL Image or file path).
    Returns dict:
      {
         'class_name': 'Cat' | 'Dog',
         'confidence': float (0-100),
         'cat_prob': float (0-100),
         'dog_prob': float (0-100),
         'raw_score': float
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

    # Sigmoid output represents P(Dog)
    prediction = model.predict(img_batch, verbose=0)
    dog_prob = float(prediction[0][0])
    cat_prob = 1.0 - dog_prob

    if dog_prob >= 0.5:
        class_name = classes.get("1", "Dog")
        confidence = dog_prob * 100.0
    else:
        class_name = classes.get("0", "Cat")
        confidence = cat_prob * 100.0

    return {
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "cat_prob": round(cat_prob * 100.0, 2),
        "dog_prob": round(dog_prob * 100.0, 2),
        "raw_score": float(dog_prob)
    }


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_cli(image_path):
    """Command-line interface prediction function."""
    print("=" * 50)
    print("Pet Image Classification - Prediction")
    print("=" * 50)

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    try:
        result = predict_image(image_path)
        icon = "[CAT]" if result["class_name"].lower() == "cat" else "[DOG]"
        print(f"\nResult: {icon} {result['class_name'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"  • Cat Probability: {result['cat_prob']:.2f}%")
        print(f"  • Dog Probability: {result['dog_prob']:.2f}%")
        print("=" * 50)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


def run_streamlit_app():
    """Interactive Streamlit Web Dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="Pet Classifier | Cat vs Dog",
        page_icon="🐾",
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
        .prediction-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 6px solid #2563eb;
            margin-top: 1rem;
        }
        .pet-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            font-size: 1.4rem;
            font-weight: 700;
            border-radius: 9999px;
            background: #2563eb;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Project Details")
        st.info("**Project**: Pet Image Classification\n\n"
                "**Backbone**: MobileNetV2 (Transfer Learning)\n\n"
                "**Classes**: Cat (0), Dog (1)\n\n"
                "**Input Resolution**: 160 x 160 px")

        if os.path.exists(MODEL_PATH):
            st.success("✅ Model loaded: `pet_classifier.keras`")
        else:
            st.error("⚠️ Model not trained yet! Run `python train.py` first.")

        st.divider()

        if os.path.exists(HISTORY_PLOT_PATH):
            st.subheader("📈 Training History")
            st.image(HISTORY_PLOT_PATH, caption="Accuracy & Loss Curves", use_column_width=True)

        st.caption("Internship ML Project • Deep Learning with TensorFlow & Keras")

    # Main Page Header
    st.markdown('<div class="main-header">🐾 Pet Image Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image of a cat or dog, or pick a sample from the dataset to test deep learning predictions.</div>', unsafe_allow_html=True)

    tabs = st.tabs(["🔍 Predict Pet", "📁 Dataset Gallery & Test", "ℹ️ About the Model"])

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

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Choose a pet photo (JPG, JPEG, PNG)...",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    image_name = uploaded_file.name
            else:
                cat_samples = sorted(glob.glob(os.path.join(CURRENT_DIR, "cat", "*.jpg")))
                dog_samples = sorted(glob.glob(os.path.join(CURRENT_DIR, "dog", "*.jpg")))

                sample_type = st.selectbox("Sample category:", ["Cat Samples", "Dog Samples"])
                sample_list = cat_samples if sample_type == "Cat Samples" else dog_samples

                if sample_list:
                    selected_path = st.selectbox(
                        "Pick a sample image:",
                        sample_list,
                        format_func=lambda p: os.path.basename(p)
                    )
                    if selected_path:
                        selected_image = Image.open(selected_path)
                        image_name = os.path.basename(selected_path)

            if selected_image is not None:
                st.image(selected_image, caption=f"Selected: {image_name}", use_column_width=True)

        with col2:
            st.subheader("2. Classification Results")
            if selected_image is not None:
                if not os.path.exists(MODEL_PATH):
                    st.error("Model file not found. Please run `python train.py` first.")
                else:
                    with st.spinner("Analyzing image features with MobileNetV2..."):
                        res = predict_image(selected_image)

                    icon = "🐱" if res["class_name"].lower() == "cat" else "🐶"
                    badge_color = "#059669" if res["confidence"] > 80 else "#d97706"

                    st.markdown(f"""
                        <div style="background:#f8fafc; border: 2px solid #e2e8f0; border-radius:12px; padding: 20px; margin-bottom: 20px;">
                            <div style="font-size: 14px; text-transform: uppercase; color: #64748b; font-weight: 600;">Prediction</div>
                            <div style="font-size: 32px; font-weight: 800; color: #1e293b; margin: 8px 0;">
                                {icon} {res['class_name'].upper()}
                            </div>
                            <div style="font-size: 16px; color: {badge_color}; font-weight: 600;">
                                Confidence: {res['confidence']:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Confidence Breakdown")
                    st.write(f"🐱 **Cat Probability:** {res['cat_prob']:.2f}%")
                    st.progress(float(res['cat_prob'] / 100.0))

                    st.write(f"🐶 **Dog Probability:** {res['dog_prob']:.2f}%")
                    st.progress(float(res['dog_prob'] / 100.0))

                    st.info(f"Raw Sigmoid Output: `{res['raw_score']:.4f}` (Values < 0.5 classify as Cat, ≥ 0.5 as Dog)")
            else:
                st.info("👈 Upload an image or choose a dataset sample to view classification results.")

    with tabs[1]:
        st.subheader("Dataset Samples Quick Preview")
        cat_samples = sorted(glob.glob(os.path.join(CURRENT_DIR, "cat", "*.jpg")))[:8]
        dog_samples = sorted(glob.glob(os.path.join(CURRENT_DIR, "dog", "*.jpg")))[:8]

        st.markdown("##### 🐱 Sample Cats (cat/)")
        cols = st.columns(4)
        for idx, p in enumerate(cat_samples):
            with cols[idx % 4]:
                st.image(p, caption=os.path.basename(p), use_column_width=True)

        st.markdown("##### 🐶 Sample Dogs (dog/)")
        cols = st.columns(4)
        for idx, p in enumerate(dog_samples):
            with cols[idx % 4]:
                st.image(p, caption=os.path.basename(p), use_column_width=True)

    with tabs[2]:
        st.subheader("Model Architecture & Engineering Highlights")
        st.markdown("""
        - **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet (1.4 million images, 1000 categories).
        - **Data Augmentation**: Integrates runtime `RandomFlip`, `RandomRotation`, and `RandomZoom` to mitigate overfitting on smaller sample counts.
        - **Two-Phase Training**:
            1. **Feature Extraction**: Freezes the convolutional base and trains a custom classification head (`GlobalAveragePooling2D` + `Dropout` + `Dense(1)`).
            2. **Fine-Tuning**: Unfreezes the upper layers of MobileNetV2 with a small learning rate (`1e-5`) for precise domain adaptation.
        - **Inference Pipeline**: Real-time batch preprocessing with Pillow, normalized with MobileNetV2 input transformations.
        """)


def main():
    parser = argparse.ArgumentParser(description="Pet Image Classification (Cat vs Dog)")
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
        print("Opening browser interface at http://localhost:8501")
        env = os.environ.copy()
        env["STREAMLIT_RUN"] = "1"
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], env=env)
        except KeyboardInterrupt:
            print("\nShutting down application.")


if __name__ == "__main__":
    main()
