import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config

class DeepShieldAugmentations:
    """Provides reusable data augmentation pipelines using Albumentations."""

    @staticmethod
    def get_train_transforms() -> A.Compose:
        """Returns the training augmentation pipeline."""
        return A.Compose([
            # Resize to model input (224x224 for EfficientNet-B3)
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),

            # Spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.4),
            A.RandomResizedCrop(
                size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                scale=(0.8, 1.0), p=0.3
            ),

            # Pixel-level augmentations
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.GaussNoise(p=0.25),
            A.Blur(blur_limit=3, p=0.2),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.2),

            # Critical for AI-image detection: simulate JPEG compression artifacts
            A.ImageCompression(quality_range=(75, 99), p=0.4),

            # Regularization: randomly mask patches
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0, p=0.2
            ),

            # Normalization and Tensor conversion
            A.Normalize(
                mean=Config.NORM_MEAN,
                std=Config.NORM_STD,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

        
    @staticmethod
    def get_val_transforms() -> A.Compose:
        """Returns the validation/test transform pipeline (no augmentations, only normalization)."""
        return A.Compose([
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(
                mean=Config.NORM_MEAN,
                std=Config.NORM_STD,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
