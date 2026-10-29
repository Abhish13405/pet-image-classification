# 🐾 10-Animal Image Classification (Deep Learning & Transfer Learning)

An end-to-end Deep Learning Computer Vision system capable of classifying images into **10 popular animal categories** using **MobileNetV2 Transfer Learning**, real-world data from the **Kaggle Animals-10** benchmark dataset, and an interactive **Streamlit** web application.

---

## 🚀 Features

- **10 Animal Categories**:
  - 🦋 Butterfly • 🐱 Cat • 🐔 Chicken • 🐮 Cow • 🐶 Dog
  - 🐘 Elephant • 🐎 Horse • 🐑 Sheep • 🕷️ Spider • 🐿️ Squirrel
- **Transfer Learning Backbone**: Uses pre-trained Google MobileNetV2 with ImageNet feature representations.
- **Two-Phase Multi-Class Training**:
  - *Phase 1*: Feature extraction with frozen base model.
  - *Phase 2*: Fine-tuning top convolutional layers for precise domain adaptation.
- **Data Augmentation**: Real-time random horizontal flips, rotations (15%), and zoom (10%) to prevent overfitting.
- **Interactive Streamlit Web Dashboard**:
  - Drag-and-drop file upload for custom animal photos.
  - Instant sample selector from the Kaggle Animals-10 dataset catalog.
  - Top-ranked probability breakdown with confidence percentages.
  - Full 10-class dataset preview gallery.
- **CLI Mode**: Fast headless terminal inference via `python main.py --image <path>`.

---

## 📊 Model Performance

- **Validation Accuracy**: 90.00%
- **Validation Loss**: 0.3988
- **Input Resolution**: 160 x 160 px
- **Architecture**: MobileNetV2 + GlobalAveragePooling2D + Dropout(0.3) + Dense(10, Softmax)

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
streamlit run main.py --server.port 8502
```
*(or run `python main.py`)* and navigate to `http://localhost:8502`.

### 2. Command Line Prediction
```bash
python main.py --image elephant/0.jpg
python main.py --image horse/0.jpg
python main.py --image dog/0.jpg
```

### 3. Re-train the Model
```bash
python train.py
```

---

## 📁 Project Structure

```text
├── butterfly/                # Kaggle Animals-10 real photos
├── cat/                      # Cat image dataset
├── chicken/                  # Chicken real photos
├── cow/                      # Cow real photos
├── dog/                      # Dog image dataset
├── elephant/                 # Elephant real photos
├── horse/                    # Horse real photos
├── sheep/                    # Sheep real photos
├── spider/                   # Spider real photos
├── squirrel/                 # Squirrel real photos
├── model/                    # Saved model weights & artifacts
│   ├── classes.json          # 10-class label mapping
│   ├── pet_classifier.keras  # Trained multi-class Keras model (22 MB)
│   └── training_history.png  # Accuracy & loss curves
├── .gitignore
├── main.py                   # Streamlit app & CLI inference script
├── requirements.txt          # Python dependencies
├── train.py                  # Multi-class training & fine-tuning pipeline
└── README.md
```
