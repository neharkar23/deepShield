import os
import sys
import cv2
import shutil
import random
from tqdm import tqdm
from PIL import Image

# Add project root directory to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import logger, set_seed


class ImageDatasetPreparer:
    """
    Prepares an AI-generated image detection dataset.
    Reads flat folders of real and AI-generated images, resizes them to 224x224,
    and splits them into train/val/test splits under dataset/{split}/{real|fake}/.

    Expected source structure:
        raw_real_dir/   <- folder of real images (.jpg/.png)
        raw_fake_dir/   <- folder of AI-generated images (.jpg/.png)

    Output structure:
        dataset/train/real/, dataset/train/fake/
        dataset/val/real/,   dataset/val/fake/
        dataset/test/real/,  dataset/test/fake/
    """

    def __init__(
        self,
        raw_real_dir: str = Config.RAW_REAL_DIR,
        raw_fake_dir: str = Config.RAW_FAKE_DIR,
        train_ratio: float = Config.SPLIT_TRAIN,
        val_ratio: float = Config.SPLIT_VAL,
    ):
        self.raw_real_dir = raw_real_dir
        self.raw_fake_dir = raw_fake_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        set_seed(Config.SEED)
        Config.create_dirs()

    def _collect_images(self, folder: str) -> list:
        """Returns sorted list of image file paths inside a folder."""
        exts = {'.jpg', '.jpeg', '.png', '.webp'}
        paths = []
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in exts:
                paths.append(os.path.join(folder, fname))
        paths.sort()
        return paths

    def _split(self, items: list):
        """Splits a list into train / val / test subsets."""
        random.shuffle(items)
        n = len(items)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        return items[:train_end], items[train_end:val_end], items[val_end:]

    def _copy_images(self, src_paths: list, out_dir: str, label: str):
        """Resizes and copies images into the output directory."""
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        for src in tqdm(src_paths, desc=f"  {label}"):
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.LANCZOS)
                fname = os.path.basename(src)
                # Ensure unique filename
                dst = os.path.join(out_dir, fname)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(fname)
                    dst = os.path.join(out_dir, f"{base}_{count}{ext}")
                img.save(dst, quality=95)
                count += 1
            except Exception as e:
                logger.warning(f"Skipping {src}: {e}")
        return count

    def run(self):
        """Runs the complete preprocessing pipeline."""
        logger.info("Starting DeepShield AI-image detection dataset preparation...")

        real_imgs = self._collect_images(self.raw_real_dir)
        fake_imgs = self._collect_images(self.raw_fake_dir)

        logger.info(f"Found {len(real_imgs)} real images and {len(fake_imgs)} AI-generated images.")

        # Balance classes
        min_count = min(len(real_imgs), len(fake_imgs))
        real_imgs = real_imgs[:min_count]
        fake_imgs = fake_imgs[:min_count]
        logger.info(f"Balanced to {min_count} images per class.")

        r_train, r_val, r_test = self._split(real_imgs)
        f_train, f_val, f_test = self._split(fake_imgs)

        logger.info("Split summary (images):")
        logger.info(f"  Real  -> Train: {len(r_train)}, Val: {len(r_val)}, Test: {len(r_test)}")
        logger.info(f"  Fake  -> Train: {len(f_train)}, Val: {len(f_val)}, Test: {len(f_test)}")

        splits = {
            "train": {"real": r_train, "fake": f_train},
            "val":   {"real": r_val,   "fake": f_val},
            "test":  {"real": r_test,  "fake": f_test},
        }

        for split_name, categories in splits.items():
            for class_name, img_list in categories.items():
                out_dir = os.path.join(Config.DATASET_DIR, split_name, class_name)
                logger.info(f"Copying {split_name}/{class_name}...")
                count = self._copy_images(img_list, out_dir, f"{split_name}/{class_name}")
                logger.info(f"  -> {count} images saved to {out_dir}")

        self.print_dataset_stats()
        logger.info("Dataset preparation complete.")

    def print_dataset_stats(self):
        """Prints the final dataset statistics."""
        print("\n" + "=" * 55)
        print("     DEEPSHIELD DATASET STATISTICS (AI Detection)")
        print("=" * 55)
        for split in ["train", "val", "test"]:
            real_dir = os.path.join(Config.DATASET_DIR, split, "real")
            fake_dir = os.path.join(Config.DATASET_DIR, split, "fake")
            real_count = len(os.listdir(real_dir)) if os.path.exists(real_dir) else 0
            fake_count = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0
            print(f"{split.upper():6s}: Real={real_count:6d}  AI-Generated={fake_count:6d}  Total={real_count+fake_count:6d}")
        print("=" * 55)


if __name__ == "__main__":
    preparer = ImageDatasetPreparer()
    preparer.run()
