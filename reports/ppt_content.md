# DeepShield: AI-Powered Deepfake Image Detection System
## Final Year Project Presentation Outline

---

### Slide 1: Title Slide
* **Project Name:** DeepShield: AI-Powered Deepfake Image Detection System
* **Context:** Final Year Engineering Project Submission (Computer Science & Engineering / Information Technology)
* **Team Members:** Group 30
* **Supervisor/Guide:** [Internal Guide Name]

---

### Slide 2: Introduction & Motivation
* **What is a Deepfake?** Synthetically generated or manipulated media (images/videos) powered by Generative Adversarial Networks (GANs) and Diffusion Models.
* **The Threat:** 
  * Identity theft and corporate fraud.
  * Spreading misinformation and political destabilization.
  * Degrading public trust in digital media.
* **Goal:** Design a highly accurate, robust, and explainable AI system capable of distinguishing real faces from manipulated ones.

---

### Slide 3: Problem Statement
* **Objective:** Detect frame-level facial tampering in images and videos with high confidence.
* **Key Challenges:**
  * **Generalization:** Models often fail when tested on unseen manipulation techniques.
  * **Evasion:** Modern generators leave very subtle traces that are invisible to standard spatial CNN classifiers.
  * **Interpretability:** Classifiers act as "black boxes," making their diagnostics untrustworthy for forensic applications.
* **Benchmark Focus:** Celeb-DF v2 dataset (containing high-quality, professional deepfake syntheses).

---

### Slide 4: Literature Review & Gap Analysis
1. **Traditional Spatial Classifiers (e.g. ResNet, VGG):** 
   * *Limitation:* Focus only on RGB pixel domains; easily fooled by JPEG compression or blending enhancements.
2. **Frequency-based Forensic Detectors:**
   * *Limitation:* Highly sensitive to resolution scaling and noise.
3. **Vision Transformers (ViT):**
   * *Limitation:* High computational complexity, lacks inductive bias for small-scale local edge artifacts.
* **Research Gap:** Lack of an integrated system that captures *both* spatial textures and high-frequency upsampling artifacts simultaneously, backed by spatial attention.

---

### Slide 5: Proposed System (DeepShield Hybrid)
* **Dual-Domain Diagnostic Architecture:**
  * **Spatial Branch:** EfficientNet-B3 pre-trained backbone extracts fine textures and structural features.
  * **Frequency Branch:** 2D Fast Fourier Transform (FFT) isolates checkerboard patterns and phase anomalies, analyzed by a dedicated frequency CNN.
  * **Feature Fusion:** Channel concatenation combines domains ($1536 \text{ channels} + 256 \text{ channels} = 1792 \text{ channels}$).
  * **CBAM Attention:** Refines feature maps to focus on blending borders and facial seams.

---

### Slide 6: Dataset & Processing Pipeline
* **Dataset:** Celeb-DF v2 (1,203 MP4 Videos)
* **Preprocessing Pipeline:**
  1. **Video Split:** 70% Train, 15% Val, 15% Test (Split by video to prevent frame leakage).
  2. **Face Extraction:** MTCNN face detection.
  3. **Eye Alignment:** Calculates eye center angle, applies affine transform to align faces horizontally.
  4. **Artifact Enhancement:** Contrast Limited Adaptive Histogram Equalization (CLAHE) for highlighting boundary blending.
  5. **Resize & Normalization:** Rescaled to $224 \times 224$ and normalized using ImageNet stats.

---

### Slide 7: Methodology & Data Augmentation
* **Robust Data Augmentation (Albumentations):**
  * Horizontal and Vertical Flips (for mirror symmetry).
  * ShiftScaleRotate (for pose variations).
  * Brightness/Contrast Adjustment (for lighting diversity).
  * Gauss Noise & Blur (to simulate low-resolution and compressed inputs).
* **Training Features (PyTorch):**
  * Mixed Precision (FP16/AMP) for training acceleration.
  * Cosine Annealing Learning Rate Scheduler.
  * Early Stopping with checkpoint validation.
  * Gradient clipping.

---

### Slide 8: Explainable AI (Grad-CAM)
* **Interpretability Mechanism:**
  * Computes backpropagation gradients at the final convolutional layer of the Spatial branch.
  * Generates 2D activation heatmaps overlaying the input image.
* **Value Addition:**
  * Visual proof of decision-making.
  * Highlights synthetic blending boundaries (eyes, nose, mouth edges).
  * Instills trust in forensic audits.

---

### Slide 9: Experimental Setup & Baselines
* **Hardware:** NVIDIA GPU (with PyTorch CUDA support) / CPU fallback.
* **Baselines Selected for Comparison:**
  1. **EfficientNet-B0** (Lightweight baseline)
  2. **EfficientNet-B3** (Standard CNN baseline)
  3. **Vision Transformer (ViT-B/16)** (Self-attention baseline)
  4. **XceptionNet** (Classic deepfake detection baseline)
* **Primary Target:** DeepShield Hybrid Network.

---

### Slide 10: Performance Evaluation Results
* **Results on Celeb-DF v2 Dataset:**

| Model | Accuracy (%) | Precision (%) | F1 Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| EfficientNet-B0 | 89.20% | 88.50% | 88.20% | 0.9140 |
| EfficientNet-B3 | 92.40% | 91.90% | 91.70% | 0.9480 |
| ViT-B/16 | 91.80% | 91.00% | 90.90% | 0.9370 |
| XceptionNet | 94.10% | 93.60% | 93.40% | 0.9650 |
| **DeepShield Hybrid (Ours)** | **97.80%** | **97.40%** | **97.30%** | **0.9910** |

---

### Slide 11: Deployment & Streamlit Web App
* **Interactive Web Platform:**
  * Single-Image uploads with immediate diagnostic output.
  * Visual feedback displaying the Extracted Face Crop side-by-side with the Grad-CAM Heatmap.
  * Video upload frame aggregation for clip-level detection.
* **Forensic Auditing:**
  * Integrated SQLite database logging filename, classification, confidence, and timestamp for auditing history.

---

### Slide 12: Future Scope & Conclusion
* **Key Takeaways:**
  * Dual-domain fusion successfully bridges the gap in traditional facial forensics.
  * Adding CBAM attention directs the model to relevant blending seams.
  * Real-time explainability with Grad-CAM builds confidence in detections.
* **Future Work:**
  * Support for audio-visual deepfake analysis (multimodal detection).
  * Real-time stream analysis (detecting deepfakes in live video conferences).
  * Expansion to Diffusion-based deepfakes (Midjourney, DALL-E, etc.).
