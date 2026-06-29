# DeepShield – Python File Explanations

> A concise overview of every `.py` file in the project, grouped by folder.

---

## 📁 Root

### `main.py`
The **central command-line orchestrator** for the entire pipeline.  
It exposes five sub-commands via `argparse`:

| Command | Action |
|---|---|
| `preprocess` | Runs `src/preprocessing.py` |
| `train` | Runs `src/train.py` with a chosen model |
| `evaluate` | Runs `src/evaluate.py` with a chosen model |
| `predict` | Runs `src/predict.py` on an image/video |
| `test` | Discovers and runs all unit tests in `tests/` |

Each sub-command is launched as a child subprocess so that `PYTHONPATH` is correctly injected for all imports.

---

### `diagnose.py`
A **quick sanity-check / debugging script**.  
Loads the saved `hybrid_best.pth` checkpoint, picks 5 real and 5 fake images from the test set, runs inference on each, and prints whether each prediction is `[CORRECT]` or `[WRONG]`.  
Useful for verifying that the trained model works as expected and that labels are not inverted.

---

## 📁 src/

### `config.py`
The **single source of truth for all project settings**.  
Contains the `Config` class with:
- All directory paths (`DATASET_DIR`, `CHECKPOINT_DIR`, `OUTPUT_DIR`, etc.)
- Dataset split ratios (70 / 15 / 15)
- Image size (224 × 224)
- ImageNet normalization stats
- Training hyperparameters (batch size 24, LR 2e-4, 30 epochs, early stopping patience 7, label smoothing 0.1)
- Hardware flags (CUDA/CPU auto-detection, Mixed Precision AMP, Gradient Clipping)
- A `create_dirs()` class-method that creates all required folders on disk.

---

### `utils.py`
**Shared utility functions** used across the whole project:

| Function | Purpose |
|---|---|
| `get_logger()` | Configures a dual-output logger (console + file at `outputs/reports/deepshield.log`) |
| `set_seed()` | Seeds `random`, `numpy`, and `torch` for reproducibility |
| `get_db_connection()` | Opens a SQLite connection to `deepshield.db` |
| `init_db()` | Creates the `predictions` table if it doesn't exist |
| `log_prediction()` | Inserts one prediction row (image name, label, confidence, timestamp) |
| `get_prediction_history()` | Retrieves all saved predictions ordered newest-first |
| `clear_prediction_history()` | Deletes all rows from the predictions table |

---

### `preprocessing.py`
**Dataset preparation pipeline** – run once before training.  
The `ImageDatasetPreparer` class:
1. Scans `dataset/raw/real/` and `dataset/raw/fake/` for `.jpg/.png/.webp` images.
2. Balances both classes to the same count.
3. Randomly shuffles and splits images into **Train / Val / Test** (70 / 15 / 15).
4. Resizes every image to 224 × 224 using LANCZOS resampling and saves them to `dataset/{split}/{real|fake}/`.
5. Prints a final statistics table showing per-split counts.

---

### `augmentations.py`
**Data augmentation pipelines** built with the Albumentations library.  
The `DeepShieldAugmentations` class has two static methods:

- **`get_train_transforms()`** – applies spatial augmentations (flip, rotate, random crop), pixel-level augmentations (brightness/contrast, hue/saturation, CLAHE, Gaussian noise, blur, sharpen), JPEG compression simulation (important for AI-image detection), CoarseDropout regularization, then normalizes and converts to a PyTorch tensor.
- **`get_val_transforms()`** – only resizes, normalizes, and converts to tensor (no augmentation).

---

### `data_loader.py`
**PyTorch dataset and DataLoader factory**.

- **`DeepShieldDataset`** – a custom `Dataset` that reads images from `real/` and `fake/` subfolders, assigns labels (`real=0`, `fake=1`), optionally limits images per class for CPU training, and applies the transform pipeline on `__getitem__`.
- **`get_data_loaders()`** – creates and returns three `DataLoader` objects (train, val, test) using the dataset class, with shuffle, pinned memory, and persistent workers configured according to `Config`.

---

### `attention.py`
**CBAM (Convolutional Block Attention Module)** implementation.  
Three PyTorch modules:

- **`ChannelAttention`** – uses Global Average Pooling + Global Max Pooling fed through a shared 1×1-conv MLP, adds the two outputs, and applies sigmoid to produce channel-wise attention weights.
- **`SpatialAttention`** – computes channel-wise average and max maps, concatenates them, passes through a 7×7 convolution, and applies sigmoid to produce a spatial attention map.
- **`CBAM`** – applies `ChannelAttention` first, then `SpatialAttention` sequentially to the feature map.

---

### `feature_fusion.py`
**Frequency-domain feature extraction and fusion**.

- **`FrequencyFeatureExtractor`** – converts an RGB input to grayscale, applies 2D FFT, takes the log of the amplitude spectrum, then passes it through a 4-layer CNN (Conv → BN → ReLU → MaxPool) to produce a 256-channel feature map at 7×7 resolution.
- **`FeatureFusion`** – concatenates spatial and frequency feature maps along the channel dimension. If their spatial resolutions differ, bilinear interpolation is applied first.

---

### `gradcam.py`
**Grad-CAM explainability module**.  
The `GradCAM` class:
1. Automatically locates the last `Conv2d` layer in the model using `find_last_conv_layer()`.
2. Registers PyTorch forward and backward hooks to capture activations and gradients.
3. **`generate_cam()`** – runs a forward + backward pass, computes gradient-weighted channel averages, applies ReLU, normalizes to [0, 1], and resizes the resulting heatmap to match the input image.
4. **`overlay_heatmap()`** – applies the JET colormap to the CAM and alpha-blends it onto the original BGR image.
5. **`remove_hooks()`** – cleans up the registered hooks after use.

---

### `face_detector.py`
**MTCNN-based face detection and alignment** (optional/legacy module).  
The `FaceDetector` class:
1. Detects a face in a BGR image using `facenet_pytorch`'s MTCNN.
2. Extracts 5 facial landmarks (eye positions).
3. Computes a rotation angle from the eye positions and rotates the image to align eyes horizontally.
4. Crops the aligned face with 10% padding and resizes to the target size (default 224 × 224).
5. Returns `None` if no high-confidence face is found.

> **Note:** This module is not used in the current prediction pipeline (face detection was intentionally removed since the model detects AI-generated images, not just deepfake faces).

---

### `train.py`
**Full model training pipeline**.

- **`get_model()`** – factory function that instantiates the selected model by name (`efficientnet_b0`, `efficientnet_b3`, `xception`, `vit`, `hybrid`).
- **`Trainer` class**:
  - Sets up the model, `AdamW` optimizer, `CosineAnnealingLR` scheduler, `BCEWithLogitsLoss` with label smoothing, mixed-precision `GradScaler`, and TensorBoard writer.
  - On CPU: freezes the EfficientNet backbone to only train the lightweight branches.
  - **`train_epoch()`** – runs one training epoch with AMP autocast, gradient clipping, and progress bar.
  - **`validate()`** – evaluates the model on the validation set with no gradient computation.
  - **`save_checkpoint()` / `load_checkpoint()`** – saves/restores model, optimizer, scheduler, and scaler states. Supports resuming from last saved epoch.
  - **`fit()`** – the main training loop: trains, validates, steps the LR scheduler, saves history JSON, saves checkpoints, and triggers early stopping.

---

### `evaluate.py`
**Post-training evaluation and reporting**.  
The `Evaluator` class loads the best checkpoint and:
1. **`run_evaluation()`** – runs inference on the full test set and collects predictions and probabilities.
2. **`calculate_metrics()`** – computes Accuracy, Precision, Recall, F1, ROC-AUC, and Average Precision.
3. **`plot_confusion_matrix()`** – saves a seaborn heatmap to `outputs/confusion_matrix/`.
4. **`plot_roc_curve()`** – saves an ROC curve plot to `outputs/roc/`.
5. **`plot_precision_recall_curve()`** – saves a PR curve to `outputs/plots/`.
6. **`plot_training_curves()`** – loads the training history JSON and saves loss/accuracy curve plots.
7. **`save_text_report()`** – writes a detailed `.txt` report with all metrics to `outputs/reports/`.

---

### `predict.py`
**Single-image and video inference engine**.  
The `Predictor` class loads the best checkpoint and exposes:

- **`_preprocess_image()`** – accepts a file path, PIL Image, or numpy array; resizes to 224 × 224 and applies val transforms.
- **`predict_image()`** – runs inference, optionally generates a Grad-CAM heatmap overlay, returns a dict with `prediction`, `confidence`, `time_taken`, `input_image`, and `gradcam_overlay`. Logs the result to the SQLite database.
- **`predict_video()`** – samples up to `max_frames` evenly-spaced frames from a video, predicts each frame, averages the probabilities, and returns a final verdict with confidence.

---

## 📁 models/

### `efficientnet_b0.py`
Wraps **EfficientNet-B0** from `torchvision`.  
Replaces the original classifier head with `Dropout(0.2) → Linear(→ 1)` for binary classification. Outputs a single raw logit per image.

---

### `efficientnet_b3.py`
Wraps **EfficientNet-B3** from `torchvision`.  
Same structure as B0 but uses the larger B3 backbone and `Dropout(0.3)`. Outputs a single raw logit.

---

### `xception.py`
Wraps the **Xception** architecture loaded via the `timm` library.  
`timm.create_model("xception", num_classes=1)` automatically replaces the head to output a single logit.

---

### `vit.py`
Wraps **ViT-B/16 (Vision Transformer)** from `torchvision`.  
Replaces `model.heads.head` with a `Linear(→ 1)` layer for binary classification. Outputs a single raw logit.

---

### `hybrid_deepshield.py`
The **primary / default model** of the project — a custom hybrid architecture.  
Five components assembled in sequence:

| Component | Details |
|---|---|
| **Spatial Backbone** | EfficientNet-B3 feature extractor → `[B, 1536, 7, 7]` |
| **Frequency Backbone** | `FrequencyFeatureExtractor` (FFT + 4-layer CNN) → `[B, 256, 7, 7]` |
| **Feature Fusion** | Channel-wise concatenation → `[B, 1792, 7, 7]` |
| **CBAM Attention** | Channel + Spatial attention → `[B, 1792, 7, 7]` |
| **Classifier Head** | Global AvgPool → Flatten → Linear(1792→512) → BN → ReLU → Dropout(0.4) → Linear(512→1) |

This design lets the model detect AI-generation artifacts in both the **spatial domain** (textures, edges) and the **frequency domain** (FFT spectrum anomalies).

---

## 📁 tests/

### `tests/test_pipeline.py`
**Integration unit tests** for the whole pipeline using Python's `unittest` framework.

| Test | What it validates |
|---|---|
| `test_database_logging` | SQLite init, insert, retrieve, and clear operations |
| `test_models_compilation` | All 5 model architectures compile and produce shape `(2, 1)` on a dummy batch |
| `test_face_detector` | `FaceDetector` initializes and handles a dummy image gracefully |
| `test_inference_predictor` | `Predictor` runs inference and returns expected result keys/types |

Run via `python main.py test` or `python -m unittest discover tests/`.

---

## 📁 deployment/

### `deployment/app.py`
The **Streamlit web application** — the user-facing front-end for DeepShield.  
Loads the trained `Predictor`, provides a UI to upload an image or video, displays the prediction result (`Real` / `AI-Generated`) with confidence score, shows the Grad-CAM heatmap overlay for images, and includes a prediction history table backed by the SQLite database.  
Run with: `streamlit run deployment/app.py`
