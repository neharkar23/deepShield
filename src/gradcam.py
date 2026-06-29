import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import logger

def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Recursively searches and returns the last Conv2d layer in the model structure."""
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            logger.info(f"Found target layer for Grad-CAM: name={name}, module={module}")
            return module
    raise ValueError("No Conv2d layer found in the model.")

class GradCAM:
    """Generates Class Activation Maps (Grad-CAM) to explain model predictions."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        
        # If target layer is not specified, dynamically find the last conv layer
        if target_layer is None:
            self.target_layer = find_last_conv_layer(self.model)
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        # Using register_full_backward_hook which is standard in modern PyTorch
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates a 2D Class Activation Map (scaled 0 to 1) for the given input tensor.
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        # Backward pass on raw logit
        logits.backward()
        
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations were not captured. Check hooks setup.")
            # Fallback to dummy CAM
            return np.zeros(input_tensor.shape[2:])
            
        # Get gradients and activations
        gradients = self.gradients  # Shape: [B, C, H, W]
        activations = self.activations  # Shape: [B, C, H, W]
        
        # Global average pool gradients to get channel weights: Shape: [B, C, 1, 1]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of channels
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        
        # Apply ReLU to keep only positive contribution features
        cam = F.relu(cam)
        
        # Remove batch and channel dims to get 2D map: Shape: [H, W]
        cam = cam.squeeze().cpu().numpy()
        
        # Handle zero-gradients case
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)
            
        # Resize to input tensor size
        h, w = input_tensor.shape[2:]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return cam_resized

    def overlay_heatmap(self, img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays the CAM heatmap onto the original BGR image.
        Returns the blended image in BGR format.
        """
        # Convert CAM to 0-255 range and apply Jet colormap
        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blended overlay
        overlaid_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
        return overlaid_img

    def remove_hooks(self):
        """Cleans up the hooks registered in PyTorch."""
        self.forward_hook.remove()
        self.backward_hook.remove()
        logger.info("Hooks removed successfully.")
        
    def __del__(self):
        # Safety cleanup
        try:
            self.remove_hooks()
        except Exception:
            pass
