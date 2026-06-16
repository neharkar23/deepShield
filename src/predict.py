import os
import sys
import time
import numpy as np
import torch
from PIL import Image

# Add project root directory to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import logger, log_prediction, init_db
from src.augmentations import DeepShieldAugmentations
from src.train import get_model
from src.gradcam import GradCAM


class Predictor:
    """
    Handles AI-generated image detection inference on images and videos,
    including Grad-CAM explainability overlays.

    NOTE: Face detection (MTCNN) has been intentionally removed.
    This model detects AI-generated images (Stable Diffusion, StyleGAN, etc.)
    vs real photographs — face crops are NOT needed for this task.
    """

    def __init__(self, model_name: str = "hybrid"):
        self.model_name = model_name
        self.device = Config.DEVICE

        # Load best weights
        self.model = get_model(self.model_name, pretrained=False)
        self.ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{self.model_name}_best.pth")

        # Initialize database
        init_db()

        if os.path.exists(self.ckpt_path):
            logger.info(f"Predictor loading best weights from {self.ckpt_path}")
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(
                f"No trained weights found at {self.ckpt_path}. "
                "Using randomly initialized model weights — predictions will be unreliable."
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = DeepShieldAugmentations.get_val_transforms()

    def _preprocess_image(self, img_input):
        """
        Accepts a file path (str) or a PIL Image / numpy RGB array.
        Returns (img_name, img_rgb_np, img_tensor).
        """
        if isinstance(img_input, str):
            img_name = os.path.basename(img_input)
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img_name = f"uploaded_{int(time.time())}.jpg"
            pil_img = img_input.convert("RGB")
        else:
            # Numpy array (H, W, 3) in RGB
            img_name = f"uploaded_{int(time.time())}.jpg"
            pil_img = Image.fromarray(img_input.astype("uint8")).convert("RGB")

        # Resize to model input size
        pil_img = pil_img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.LANCZOS)
        img_rgb_np = np.array(pil_img)

        # Apply val transforms (normalize + to tensor)
        augmented = self.transforms(image=img_rgb_np)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)

        return img_name, img_rgb_np, img_tensor

    def predict_image(self, img_input, generate_cam: bool = True) -> dict:
        """
        Predicts whether an image is Real or AI-Generated.

        Args:
            img_input: File path (str), PIL Image, or numpy RGB array.
            generate_cam: If True, generates a Grad-CAM overlay highlighting
                          the regions that triggered the AI-generated prediction.
        Returns:
            dict with keys: prediction, confidence, time_taken, input_image, gradcam_overlay
        """
        start_time = time.time()

        img_name, img_rgb_np, img_tensor = self._preprocess_image(img_input)

        gradcam_overlay = None

        if generate_cam:
            cam_extractor = GradCAM(self.model)
            img_tensor.requires_grad = True
            cam_map = cam_extractor.generate_cam(img_tensor)

            with torch.no_grad():
                logits = self.model(img_tensor)

            # Overlay on the resized RGB image (convert to BGR for cv2 overlay)
            import cv2
            img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
            gradcam_overlay = cam_extractor.overlay_heatmap(img_bgr, cam_map, alpha=0.5)
            cam_extractor.remove_hooks()
        else:
            with torch.no_grad():
                logits = self.model(img_tensor)

        # Sigmoid → probability of being AI-generated (label=1 = fake/AI)
        prob = torch.sigmoid(logits).item()

        is_fake = prob >= 0.5
        prediction = "AI-Generated" if is_fake else "Real"
        confidence = prob if is_fake else (1.0 - prob)

        elapsed_time = time.time() - start_time

        # Log to database
        log_prediction(img_name, prediction, confidence)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "time_taken": elapsed_time,
            "input_image": img_rgb_np,        # resized RGB numpy for display
            "gradcam_overlay": gradcam_overlay,  # BGR numpy overlay (or None)
        }

    def predict_video(self, video_path: str, max_frames: int = 20) -> dict:
        """
        Samples frames from a video and returns an averaged AI-generated prediction.
        """
        import cv2
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video file {video_path} is empty or corrupted.")

        step = max(1, total_frames // max_frames)
        frame_indices = [i * step for i in range(max_frames) if i * step < total_frames]

        frame_probabilities = []
        logger.info(f"Analyzing {len(frame_indices)} frames from: {video_path}")

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            try:
                # Convert BGR frame to RGB numpy for predict_image
                import cv2 as _cv2
                frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                res = self.predict_image(frame_rgb, generate_cam=False)
                prob = res["confidence"] if res["prediction"] == "AI-Generated" else (1.0 - res["confidence"])
                frame_probabilities.append(prob)
            except Exception:
                continue

        cap.release()

        if len(frame_probabilities) == 0:
            raise RuntimeError("Failed to analyze any frames from the video.")

        avg_prob = float(np.mean(frame_probabilities))
        is_fake = avg_prob >= 0.5
        prediction = "AI-Generated" if is_fake else "Real"
        confidence = avg_prob if is_fake else (1.0 - avg_prob)

        elapsed_time = time.time() - start_time
        video_name = os.path.basename(video_path)

        log_prediction(video_name, prediction, confidence)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "time_taken": elapsed_time,
            "analyzed_frames": len(frame_probabilities),
            "average_probability": avg_prob,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict AI-Generated vs Real on images/videos.")
    parser.add_argument("--model", type=str, default="hybrid", help="Model name to use")
    parser.add_argument("--input", type=str, required=True, help="Path to image or video file")
    args = parser.parse_args()

    predictor = Predictor(model_name=args.model)

    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".mp4", ".avi", ".mov"]:
        res = predictor.predict_video(args.input)
        print("\n" + "=" * 50)
        print("     AI-GENERATED VIDEO DETECTION RESULTS")
        print("=" * 50)
        print(f"Video File:  {os.path.basename(args.input)}")
        print(f"Prediction:  {res['prediction']}")
        print(f"Confidence:  {res['confidence']*100:.2f}%")
        print(f"Time Taken:  {res['time_taken']:.2f} seconds")
        print(f"Frames:      {res['analyzed_frames']}")
        print("=" * 50 + "\n")
    else:
        res = predictor.predict_image(args.input)
        print("\n" + "=" * 50)
        print("     AI-GENERATED IMAGE DETECTION RESULTS")
        print("=" * 50)
        print(f"Image File:  {os.path.basename(args.input)}")
        print(f"Prediction:  {res['prediction']}")
        print(f"Confidence:  {res['confidence']*100:.2f}%")
        print(f"Time Taken:  {res['time_taken']:.4f} seconds")
        print("=" * 50 + "\n")
