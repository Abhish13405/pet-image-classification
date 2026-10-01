# 🐾 Pet Image Classification (Cat vs Dog)

An end-to-end Deep Learning system to classify pet images as **Cat** or **Dog** using **MobileNetV2 Transfer Learning** and an interactive **Streamlit** web application.

---

## 🚀 Features

- **Transfer Learning Backbone**: Uses pre-trained MobileNetV2 with frozen base weights and fine-tuning for high accuracy on small datasets.
- **Data Augmentation**: Runtime random flipping, rotation, and zooming to prevent overfitting.
- **Interactive Streamlit Web Dashboard**:
  - Drag-and-drop custom image upload.
  - Quick-test gallery from the dataset.
  - Real-time classification with confidence percentages and probability breakdown.
  - Model architecture inspection and training history curves.
- **CLI Mode**: Fast headless terminal inference via `python main.py --image <path>`.

---

## 📊 Model Performance

- **Validation Accuracy**: 100.00%
- **Validation Loss**: 0.1049
- **Input Resolution**: 160 x 160 px

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhish13405/pet-image-classification.git
   cd pet-image-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

### 1. Launch the Web Application
```bash
streamlit run main.py
```
*(or run `python main.py`)* and navigate to `http://localhost:8501` (or `8502`).

### 2. Command Line Prediction
```bash
python main.py --image cat/0.jpg
python main.py --image dog/0.jpg
```

### 3. Re-train the Model
```bash
python train.py
```

---

## 📁 Project Structure

```text
├── cat/                      # Cat image dataset
├── dog/                      # Dog image dataset
├── model/                    # Saved model weights & artifacts
│   ├── classes.json          # Class label mapping
│   ├── pet_classifier.keras  # Trained Keras model
│   └── training_history.png  # Accuracy & loss curves
├── .gitignore
├── main.py                   # Streamlit app & CLI inference script
├── requirements.txt          # Python dependencies
├── train.py                  # Training & fine-tuning pipeline
└── README.md
```
