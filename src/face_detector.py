import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from src.utils import logger

class FaceDetector:
    """Uses MTCNN for face detection and performs eye alignment and cropping."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Initialize MTCNN for single face detection
        self.detector = MTCNN(
            keep_all=False, 
            post_process=False, 
            device=self.device, 
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
        logger.info(f"FaceDetector initialized on device: {device}")

    def align_and_crop_face(self, img: np.ndarray, target_size: int = 224) -> np.ndarray:
        """
        Detects a face in the image, aligns the eyes horizontally, crops the face,
        and resizes it to target_size. Returns None if no face is detected.
        """
        if img is None:
            return None
            
        h, w, c = img.shape
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect face and landmarks
            boxes, probs, landmarks = self.detector.detect(img_rgb, landmarks=True)
            
            if boxes is None or len(boxes) == 0 or probs[0] < 0.85:
                # No high-confidence face detected
                return None
                
            box = boxes[0]
            pts = landmarks[0]  # Shape: (5, 2)
            
            # Eye coordinates: 0 is left eye, 1 is right eye
            left_eye = pts[0]
            right_eye = pts[1]
            
            # Calculate rotation angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotation center: midpoint of eyes
            eye_center = (
                float((left_eye[0] + right_eye[0]) / 2.0),
                float((left_eye[1] + right_eye[1]) / 2.0)
            )
            
            # Rotate image to align eyes horizontally
            rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
            rotated_img = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)
            
            # Detect face on the rotated image for a perfect crop
            rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
            r_boxes, r_probs = self.detector.detect(rotated_img_rgb, landmarks=False)
            
            if r_boxes is not None and len(r_boxes) > 0 and r_probs[0] > 0.80:
                crop_box = r_boxes[0]
            else:
                # Fallback to the original bounding box coordinates
                crop_box = box
                
            # Crop and clamp coordinates
            x1, y1, x2, y2 = map(int, crop_box)
            
            # Add padding around the face for context
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)
            
            # Extract crop
            face_crop = rotated_img[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
                
            # Resize to target size
            face_crop_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
            return face_crop_resized
            
        except Exception as e:
            logger.warning(f"Error during face detection/alignment: {e}")
            return None
