# 🫁 PulmoScan-AI

### Intelligent Chest X-Ray Disease Classification Using Deep Learning & Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

PulmoScan-AI is a deep learning-based medical imaging system developed to automatically classify Chest X-Ray images into four clinically important categories:

* COVID-19
* Pneumonia
* Tuberculosis (TB)
* Normal

The project leverages **Transfer Learning with DenseNet121**, enabling robust feature extraction from medical images while reducing training time and improving classification performance.

A balanced dataset of **9,800 Chest X-Ray images** was curated from multiple public medical imaging datasets and used to train and evaluate the model.

---

## 🎯 Problem Statement

Accurate diagnosis of pulmonary diseases requires experienced radiologists and can be time-consuming in high-volume healthcare environments.

This project aims to develop an AI-assisted diagnostic system capable of identifying multiple lung diseases directly from Chest X-Ray images, helping support faster and more accessible healthcare screening.

---

## 🚀 Key Features

✅ Multi-Class Disease Classification

✅ Transfer Learning using DenseNet121

✅ Balanced Medical Imaging Dataset

✅ Data Augmentation Pipeline

✅ Early Stopping Regularization

✅ Automated Evaluation Metrics

✅ Confusion Matrix Visualization

✅ Classification Reports

✅ Confidence-Based Predictions

✅ Saved Model for Future Deployment

---

## 🏗️ Project Architecture

```text
Chest X-Ray Image
        │
        ▼
Image Preprocessing
(Resize + Normalize)
        │
        ▼
Data Augmentation
        │
        ▼
DenseNet121 Backbone
(Pretrained on ImageNet)
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer (256)
        │
        ▼
Dropout (0.5)
        │
        ▼
Softmax Output Layer
        │
        ▼
Disease Prediction
```

---

## 📂 Dataset

The dataset was created by combining multiple publicly available Chest X-Ray datasets from Kaggle.

### Disease Classes

| Class             |   Images |
| ----------------- | -------: |
| COVID-19          |     2450 |
| Pneumonia         |     2450 |
| Tuberculosis (TB) |     2450 |
| Normal            |     2450 |
| **Total**         | **9800** |

### Train / Validation Split

| Dataset    | Images |
| ---------- | -----: |
| Training   |   7840 |
| Validation |   1960 |

---

## ⚙️ Data Preprocessing

### Image Resizing

All images are resized to:

```python
224 x 224
```

### Pixel Normalization

```python
rescale = 1.0 / 255
```

### Data Augmentation

```python
rotation_range = 15
zoom_range = 0.2
horizontal_flip = True
```

Data augmentation improves model generalization and reduces overfitting.

---

## 🧠 Model Architecture

### Base Model

```python
DenseNet121(
    weights='imagenet',
    include_top=False
)
```

All pretrained layers are frozen during training to preserve learned visual representations.

### Custom Classification Head

```python
GlobalAveragePooling2D()

Dense(256, activation='relu')

Dropout(0.5)

Dense(4, activation='softmax')
```

### Output Classes

```text
COVID-19
NORMAL
PNEUMONIA
TB
```

---

## 🔬 Training Configuration

| Parameter      | Value                    |
| -------------- | ------------------------ |
| Optimizer      | Adam                     |
| Loss Function  | Categorical Crossentropy |
| Batch Size     | 32                       |
| Epochs         | 20                       |
| Early Stopping | Enabled                  |
| Patience       | 3                        |

### Early Stopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

This prevents overfitting and restores the best-performing model automatically.

---

## 📊 Model Evaluation

The model is evaluated using:

### Accuracy Score

Measures overall classification performance.

### Confusion Matrix

Visualizes correct and incorrect predictions across all disease classes.

### Classification Report

Provides:

* Precision
* Recall
* F1-Score

for each disease category.

### Prediction Confidence

The model outputs both:

* Predicted Disease
* Confidence Percentage

for each X-Ray image.

---

## 📈 Results

### Validation Accuracy

```text
90%
```

The model demonstrates strong performance across all four pulmonary disease categories.

---

## 💻 Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Data Science Libraries

* NumPy
* Pandas
* Scikit-Learn

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* Kaggle Notebook

---

## 📁 Project Structure

```text
PulmoScan-AI/
│
├── gen-ai-final-code.ipynb
├── Biomedical_Imaging_Research_Paper_Final.pdf
├── Biomedical_Imaging_Research_Paper_Final.docx
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/PulmoScan-AI.git
```

Move into project directory:

```bash
cd PulmoScan-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
gen-ai-final-code.ipynb
```

and run all cells.

---

## 🔮 Future Enhancements

* Grad-CAM Explainability
* Multi-Modal CT + X-Ray Fusion
* Severity Grading Prediction
* Fine-Tuning DenseNet121
* Ensemble Learning
* TensorFlow Lite Deployment
* FastAPI/Flask Deployment
* Real-Time Clinical Interface
* Federated Learning for Privacy

---

## 👨‍💻 Authors

**Shah Tanish**

**Prince Bhandari**
---

## 📜 Research Paper

The complete research paper is included in this repository:

```text
Biomedical_Imaging_Research_Paper_Final.pdf
```

---

## ⭐ Acknowledgements

Special thanks to the creators of the public medical imaging datasets and the open-source deep learning community for providing the tools and resources used in this research.
