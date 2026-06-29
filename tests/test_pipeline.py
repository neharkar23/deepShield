import os
import sys
import unittest
import numpy as np
import torch

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import init_db, log_prediction, get_prediction_history, clear_prediction_history
from src.face_detector import FaceDetector
from src.predict import Predictor

class TestDeepShieldPipeline(unittest.TestCase):
    """Integrative unit tests to validate the DeepShield detection pipeline."""
    
    @classmethod
    def setUpClass(cls):
        # Enforce CPU mode for testing to avoid CUDA memory conflicts
        Config.DEVICE = "cpu"
        init_db()

    def test_database_logging(self):
        """Validates SQLite database operations: logging, retrieving, and clearing."""
        clear_prediction_history()
        history_before = get_prediction_history()
        self.assertEqual(len(history_before), 0, "Database should be empty after clear")
        
        # Log a prediction
        log_prediction("test_face.jpg", "Fake", 0.985)
        
        history_after = get_prediction_history()
        self.assertEqual(len(history_after), 1, "Database should contain exactly one prediction")
        self.assertEqual(history_after[0]["image_name"], "test_face.jpg")
        self.assertEqual(history_after[0]["prediction"], "Fake")
        self.assertAlmostEqual(history_after[0]["confidence"], 0.985)
        
        # Clear again
        clear_prediction_history()
        self.assertEqual(len(get_prediction_history()), 0)

    def test_models_compilation(self):
        """Verifies that all model classes compile and return the correct tensor shapes."""
        from models.efficientnet_b0 import EfficientNetB0
        from models.efficientnet_b3 import EfficientNetB3
        from models.xception import XceptionNet
        from models.vit import VisionTransformer
        from models.hybrid_deepshield import HybridDeepShield
        
        dummy_batch = torch.randn(2, 3, 224, 224)
        
        models_to_test = {
            "EfficientNet-B0": EfficientNetB0(pretrained=False),
            "EfficientNet-B3": EfficientNetB3(pretrained=False),
            "XceptionNet": XceptionNet(pretrained=False),
            "ViT": VisionTransformer(pretrained=False),
            "HybridDeepShield": HybridDeepShield(pretrained=False)
        }
        
        for name, model in models_to_test.items():
            model.eval()
            with torch.no_grad():
                outputs = model(dummy_batch)
            self.assertEqual(outputs.shape, (2, 1), f"Model {name} output shape mismatch. Expected (2, 1), got {outputs.shape}")

    def test_face_detector(self):
        """Tests FaceDetector initialization and processing on a dummy image."""
        detector = FaceDetector(device="cpu")
        
        # Create dummy BGR image (solid blue square)
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        dummy_img[:, :, 0] = 255 # Blue channel
        
        # Run cropping (should fall back or handle gracefully since there is no actual human face)
        crop = detector.align_and_crop_face(dummy_img, target_size=224)
        # Note: crop might be None because there's no face, which is expected behavior
        if crop is not None:
            self.assertEqual(crop.shape, (224, 224, 3))

    def test_inference_predictor(self):
        """Verifies that Predictor can perform inference on raw images."""
        predictor = Predictor(model_name="hybrid")
        
        # Solid grey dummy image
        dummy_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Run predictor (with generate_cam=False to skip backpropagation hooks on dummy inputs)
        result = predictor.predict_image(dummy_img, generate_cam=False)
        
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("time_taken", result)
        self.assertIn("face_crop", result)
        
        self.assertIn(result["prediction"], ["Real", "Fake"])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(result["face_crop"].shape, (224, 224, 3))

if __name__ == "__main__":
    unittest.main()
