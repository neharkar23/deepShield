import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.utils import logger
from src.augmentations import DeepShieldAugmentations

class DeepShieldDataset(Dataset):
    """Custom PyTorch Dataset for AI-generated image detection."""

    def __init__(self, root_dir: str, transform=None, subset_per_class: int = 0):
        """
        Args:
            root_dir: Path containing real/ and fake/ subdirectories.
            transform: Albumentations transform pipeline.
            subset_per_class: If > 0, randomly sample this many images per class
                              (useful for fast CPU training). 0 = use all images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # real = 0, AI-generated/fake = 1
        self.class_to_idx = {"real": 0, "fake": 1}

        if not os.path.exists(root_dir):
            logger.warning(f"Directory {root_dir} does not exist. Dataset will be empty.")
            return

        for class_name in ["real", "fake"]:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            label = self.class_to_idx[class_name]
            img_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]

            # Apply subset cap for CPU training
            if subset_per_class > 0 and len(img_files) > subset_per_class:
                random.seed(Config.SEED)
                img_files = random.sample(img_files, subset_per_class)

            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, label))

        logger.info(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load with PIL (returns RGB — no BGR conversion needed)
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image: {img_path} — {e}")

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label


def get_data_loaders(
    train_dir: str = Config.TRAIN_DIR,
    val_dir:   str = Config.VAL_DIR,
    test_dir:  str = Config.TEST_DIR,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,   # 4 workers on GPU, 0 on CPU
    subset_per_class: int = Config.CPU_SUBSET_PER_CLASS,
) -> tuple:
    """Creates and returns DataLoaders for train, val, and test splits."""

    train_dataset = DeepShieldDataset(
        root_dir=train_dir,
        transform=DeepShieldAugmentations.get_train_transforms(),
        subset_per_class=subset_per_class,
    )
    val_dataset = DeepShieldDataset(
        root_dir=val_dir,
        transform=DeepShieldAugmentations.get_val_transforms(),
        subset_per_class=0,  # Always use full val set for honest metrics
    )

    _pin = Config.PIN_MEMORY and (Config.DEVICE == "cuda")
    _persist = (num_workers > 0)  # persistent_workers only valid with workers>0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=_persist,
        prefetch_factor=2 if _persist else None,
    )

    test_loader = None
    if test_dir and os.path.exists(test_dir):
        test_dataset = DeepShieldDataset(
            root_dir=test_dir,
            transform=DeepShieldAugmentations.get_val_transforms(),
            subset_per_class=0,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin,
        )

    return train_loader, val_loader, test_loader
