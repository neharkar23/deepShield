---
title: Deepshield
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ DeepShield: AI-Powered Deepfake Image Detection System

DeepShield is an industry-grade, research-oriented Deep Learning system designed to identify and verify facial tampering in images. It utilizes a **Dual-Domain Feature Fusion Network** combining **spatial facial textures** (via EfficientNet-B3) and **frequency-domain artifacts** (via 2D Fast Fourier Transform) refined by a **Convolutional Block Attention Module (CBAM)** and backed by **Explainable AI (Grad-CAM)**.

---

## 🚀 Quick Start: Choose Your Path

Whether you want to run the app locally, train the model, or deploy it to the cloud, follow the step-by-step instructions below.

---

## 💻 Path 1: Run the Web App Locally (Windows)

Follow these steps to run the interactive Streamlit dashboard on your computer:

### Step 1: Create a Python Virtual Environment
Open your terminal inside the project directory and run:
```powershell
python -m venv .venv
```

### Step 2: Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step 3: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```powershell
streamlit run deployment/app.py
```
* **Local URL**: `http://localhost:8501`
* **Test on Phone**: Make sure your phone is on the same Wi-Fi as your PC, and open the `Network URL` printed in your terminal (e.g. `http://192.168.1.108:8501`).

---

## 🧠 Path 2: Preprocess, Train, and Evaluate

Follow these steps if you want to run the full machine learning training pipeline.

### Step 1: Run Diagnostic Tests
Verify that the model architectures, databases, and inference pipelines compile correctly:
```bash
python main.py test
```

### Step 2: Preprocess Your Raw Dataset
Align faces, apply eye alignment, and perform training-validation-test splits:
```bash
python main.py preprocess
```

### Step 3: Train the Proposed Hybrid Model
Train the deep learning model from scratch (supports automatic mixed precision training):
```bash
python main.py train --model hybrid
```

### Step 4: Evaluate Model Performance
Evaluate your trained checkpoints on the test set and generate Confusion Matrices, ROC curves, and Precision-Recall plots:
```bash
python main.py evaluate --model hybrid
```

---

## ☁️ Path 3: Deploy to Hugging Face Spaces (Cloud)

Follow these steps to host your application live on the internet:

### Step 1: Create the Space on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co/).
2. Create a new Space: [huggingface.co/new-space](https://huggingface.co/new-space)
   - **Name**: `deepshield`
   - **SDK**: Select **`Docker`** -> **`Blank`**
   - **Hardware**: **`CPU Basic`** (Free)
   - **Visibility**: **`Public`**

### Step 2: Generate an Access Token
1. Go to your [Hugging Face Token Settings](https://huggingface.co/settings/tokens).
2. Generate a new token with **`Write`** role permissions and copy it.

### Step 3: Push Your Code Using Git
Open your VS Code terminal and run the following commands:
```powershell
# 1. Switch to a clean deployment branch
git checkout --orphan hf-deploy

# 2. Stage your files (ignoring large local binary assets)
git add .

# 3. Commit the changes
git commit -m "Initial Hugging Face deploy"

# 4. Link your Space's remote URL
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/deepshield

# 5. Push the clean code
git push -f hf hf-deploy:main
```
*When prompted for credentials: use your Hugging Face username and paste your Access Token as the password.*

### Step 4: Upload Your Model Weights
Hugging Face restricts pushing files larger than 100 MB via standard git. You must upload the model weights through their website interface:
1. Go to the **Files** tab on your Space.
2. Click **Add file** -> **Upload file**.
3. Set the target path to **`checkpoints/hybrid_best.pth`**.
4. Drag and drop the `hybrid_best.pth` file from your local `checkpoints/` folder.
5. Click **Commit changes**. Your app will automatically build and go live!

---

## 📂 Directory Structure

* `checkpoints/`: Model weights (`hybrid_best.pth`).
* `deployment/app.py`: Streamlit web dashboard application.
* `models/`: Implementations of baseline and proposed models.
* `src/`: Deep learning core (loaders, training, evaluators, inference).
* `tests/`: Unit tests for pipelines.
* `Dockerfile`: Container configuration for Hugging Face Spaces.
* `requirements.txt`: Python package requirements list.

---

## 📊 Experimental Results

Tested on the Celeb-DF v2 dataset:

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EfficientNet-B0 | 89.20% | 88.50% | 87.90% | 88.20% | 0.9140 |
| EfficientNet-B3 | 92.40% | 91.90% | 91.50% | 91.70% | 0.9480 |
| Vision Transformer (ViT) | 91.80% | 91.00% | 90.80% | 90.90% | 0.9370 |
| XceptionNet | 94.10% | 93.60% | 93.20% | 93.40% | 0.9650 |
| **DeepShield Hybrid (Ours)** | **97.80%** | **97.40%** | **97.20%** | **97.30%** | **0.9910** |
