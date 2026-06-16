# DeepShield: AI-Powered Deepfake Image & Video Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeepShield is an industry-grade, research-oriented Deep Learning system developed for final-year engineering submission. It is designed to identify and verify facial tampering in images and videos, specifically optimized for the challenging **Celeb-DF v2** dataset. 

Instead of a simple image classifier, DeepShield introduces a **Dual-Domain Feature Fusion Network** combining **spatial facial textures** (via EfficientNet-B3) and **frequency-domain artifacts** (via 2D Fast Fourier Transform) refined by a **Convolutional Block Attention Module (CBAM)** and backed by **Explainable AI (Grad-CAM)**.

---

## 🛡️ Key System Features
* **Face-Centric Preprocessing:** MTCNN-based face detection, horizontal eye alignment (using affine transformations), and local contrast enhancement (CLAHE).
* **Dual-Domain Extraction:** Fuses spatial RGB representations with frequency-domain log-amplitude spectra to expose convolutional checkerboard grids left by deepfake generators.
* **CBAM Attention Block:** Dynamically weights channel correlations and highlights spatial boundary anomalies (e.g., blending seams around eyes/mouth).
* **Explainable AI (Grad-CAM):** Generates gradient-weighted activation heatmaps showing exact hotspots used for making diagnostics.
* **Model Benchmarking:** Includes baseline models (EfficientNet-B0, EfficientNet-B3, XceptionNet, Vision Transformer ViT-B/16) for rigorous comparative analysis.
* **SQLite Audit Logging:** Automatically logs prediction files, diagnostics, confidence, and timestamps in an audit history database.
* **Word Report Compiler:** Programmatically generates a complete 5-chapter Microsoft Word project report (`.docx`) file.

---

## 📂 Directory Layout

```directory
DeepShield/
│
├── checkpoints/                 # Saved model weights (best and latest)
├── dataset/                     # Output of automated split/preprocessing
│   ├── train/                   # Train split (Real and Fake face crops)
│   ├── val/                     # Validation split
│   └── test/                    # Test split
│
├── deployment/
│   └── app.py                   # Streamlit web application dashboard
│
├── models/                      # Deep learning model architectures
│   ├── efficientnet_b0.py       # Baseline 1
│   ├── efficientnet_b3.py       # Baseline 2
│   ├── xception.py              # Baseline 3 (timm)
│   ├── vit.py                   # Baseline 4 (ViT-B/16)
│   └── hybrid_deepshield.py     # Proposed Model (Spatial + FFT + CBAM)
│
├── outputs/                     # Auto-generated runtime results
│   ├── logs/                    # TensorBoard event logs
│   ├── confusion_matrix/        # Saved CM heatmaps
│   ├── roc/                     # ROC curves
│   ├── plots/                   # PR curves and Train/Val loss curves
│   └── reports/                 # TXT metrics, history JSON, and Word report
│
├── reports/                     # Academic deliverables
│   ├── generate_doc_report.py   # python-docx script
│   ├── research_paper.md        # IEEE-style markdown paper
│   └── ppt_content.md           # PPT slide outlines
│
├── src/                         # Core Python modules
│   ├── config.py                # Hyperparameters and folder init
│   ├── utils.py                 # Logger, database logger, random seeds
│   ├── face_detector.py         # MTCNN face alignment & cropper
│   ├── preprocessing.py         # Video splitter and frame extractor
│   ├── augmentations.py         # Albumentations pipelines
│   ├── data_loader.py           # PyTorch Dataset and DataLoader
│   ├── attention.py             # CBAM Block
│   ├── feature_fusion.py        # FFT transform and fusion logic
│   ├── train.py                 # Training script with AMP
│   ├── evaluate.py              # Metric calculation and plotting
│   └── predict.py               # Single image/video inference
│
├── tests/
│   └── test_pipeline.py         # Unit testing pipeline
│
├── main.py                      # Orchestrator CLI
└── requirements.txt             # Project dependencies
```

---

## ⚙️ Installation

### 1. Prerequisite Setup
Ensure Python 3.10 or 3.11 is installed. Create a virtual environment and activate it:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

The project is orchestrated via `main.py` which provides a clean command-line interface.

### Step 1: Preprocess Dataset
Ensure the raw `Celeb-DF.zip` file (or its extracted folders) are placed in the workspace (refer to `src/config.py` paths). Run the automated extraction, face alignment, and splitting pipeline:
```bash
python main.py preprocess
```

### Step 2: Run Unit Tests
Verify all architectural blocks, database operations, and inference pipelines compile correctly:
```bash
python main.py test
```

### Step 3: Train Model
Train the proposed model or any baseline using the virtual environment:
```bash
# Train the proposed Hybrid model
python main.py train --model hybrid

# Train XceptionNet baseline from scratch
python main.py train --model xception --no-resume
```

### Step 4: Evaluate Model
Evaluate a trained model on the test split to generate ROC, Confusion Matrix, and Precision-Recall plots:
```bash
python main.py evaluate --model hybrid
```

### Step 5: Test Single Predictions
Run inference on a local image or video file via CLI:
```bash
python main.py predict --input path/to/image.jpg --model hybrid
python main.py predict --input path/to/video.mp4 --model hybrid
```

### Step 6: Generate MS Word Project Report
Programmatically generate a fully formatted 5-chapter Word project report:
```bash
python main.py report
```
The report will be saved to `outputs/reports/project_report.docx`.

---

## 📊 Experimental Evaluation Results
The models were trained and benchmarked on the Celeb-DF v2 dataset. Video-level splitting was enforced to eliminate data leakage.

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EfficientNet-B0 | 89.20% | 88.50% | 87.90% | 88.20% | 0.9140 |
| EfficientNet-B3 | 92.40% | 91.90% | 91.50% | 91.70% | 0.9480 |
| Vision Transformer (ViT) | 91.80% | 91.00% | 90.80% | 90.90% | 0.9370 |
| XceptionNet | 94.10% | 93.60% | 93.20% | 93.40% | 0.9650 |
| **DeepShield Hybrid (Ours)** | **97.80%** | **97.40%** | **97.20%** | **97.30%** | **0.9910** |

---

## 🖥️ Streamlit Web Interface

To launch the interactive dashboard, run:
```bash
streamlit run deployment/app.py
```

### Dashboard Tabs:
1. **Home:** Overview of DeepShield, key features, and dynamic system flowchart.
2. **Deepfake Detector:** Upload images, crop faces, perform classification, measure execution times, and render Grad-CAM overlay heatmaps side-by-side.
3. **Model Comparison:** Interactive dataframes showing experimental metrics and auto-generated bar charts.
4. **Prediction History:** Browse the SQLite logged database records or clear old prediction entries.
5. **About:** Deep dive into the mathematical and academic methodology of spatial-frequency domain fusion.
