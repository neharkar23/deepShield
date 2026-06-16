import os
import torch

class Config:
    # -------------------------------------------------------------
    # Project Paths
    # -------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
    VAL_DIR     = os.path.join(DATASET_DIR, "val")
    TEST_DIR    = os.path.join(DATASET_DIR, "test")

    # Raw source image folders (set these before running preprocessing)
    # After CIFAKE extraction these point to:  dataset/raw/real  and  dataset/raw/fake
    RAW_REAL_DIR = os.path.join(DATASET_DIR, "raw", "real")
    RAW_FAKE_DIR = os.path.join(DATASET_DIR, "raw", "fake")

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

    # Output Subdirectories
    PLOTS_DIR      = os.path.join(OUTPUT_DIR, "plots")
    CONF_MATRIX_DIR= os.path.join(OUTPUT_DIR, "confusion_matrix")
    ROC_DIR        = os.path.join(OUTPUT_DIR, "roc")
    REPORTS_DIR    = os.path.join(OUTPUT_DIR, "reports")
    GRADCAM_DIR    = os.path.join(OUTPUT_DIR, "gradcam")

    # Database path
    DB_PATH = os.path.join(REPORTS_DIR, "deepshield.db")

    # -------------------------------------------------------------
    # Dataset Split Ratios
    # -------------------------------------------------------------
    SPLIT_TRAIN = 0.70
    SPLIT_VAL   = 0.15
    SPLIT_TEST  = 0.15

    # -------------------------------------------------------------
    # Preprocessing / Image Settings
    # -------------------------------------------------------------
    IMAGE_SIZE = 224  # EfficientNet-B3 native resolution

    # ImageNet normalization stats
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]

    # ─────────────────────────────────────────────────────────
    # Model Hyperparameters — tuned for RTX 3050 (4GB VRAM)
    # ─────────────────────────────────────────────────────────
    SEED                    = 42
    BATCH_SIZE              = 24          # 24 × 224×224×3 fits safely in 4GB VRAM with AMP
    LEARNING_RATE           = 2e-4        # Fine-tuning LR for pretrained EfficientNet-B3
    WEIGHT_DECAY            = 1e-4
    LABEL_SMOOTHING         = 0.1        # Prevents overconfidence; improves generalisation
    EPOCHS                  = 30          # With early stopping; rarely reaches 30
    EARLY_STOPPING_PATIENCE = 7           # Generous patience for proper convergence
    WARMUP_EPOCHS           = 2           # Cosine annealing warmup

    # DataLoader workers — use 4 on GPU for fast parallel loading
    NUM_WORKERS             = 4

    # 0 = use ALL dataset images (RTX 3050 is fast enough)
    CPU_SUBSET_PER_CLASS    = 0

    # ─────────────────────────────────────────────────────────
    # Hardware
    # ─────────────────────────────────────────────────────────
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION   = True              # FP16 AMP — ~2× speedup on RTX 3050
    GRADIENT_CLIPPING = 1.0
    PIN_MEMORY        = True              # Faster CPU→GPU transfer


    @classmethod
    def create_dirs(cls):
        """Creates the necessary directories for the project structure."""
        dirs = [
            cls.RAW_REAL_DIR,
            cls.RAW_FAKE_DIR,
            cls.TRAIN_DIR, cls.VAL_DIR, cls.TEST_DIR,
            cls.CHECKPOINT_DIR,
            cls.PLOTS_DIR, cls.CONF_MATRIX_DIR,
            cls.ROC_DIR, cls.REPORTS_DIR, cls.GRADCAM_DIR,
            os.path.join(cls.TRAIN_DIR, "real"), os.path.join(cls.TRAIN_DIR, "fake"),
            os.path.join(cls.VAL_DIR,   "real"), os.path.join(cls.VAL_DIR,   "fake"),
            os.path.join(cls.TEST_DIR,  "real"), os.path.join(cls.TEST_DIR,  "fake"),
            os.path.join(cls.BASE_DIR,  "reports"),
            os.path.join(cls.BASE_DIR,  "deployment", "pages"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    Config.create_dirs()
    print("Project directory structure created successfully.")
