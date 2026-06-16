import torch
import torch.nn as nn
import timm

class XceptionNet(nn.Module):
    """Xception network classifier for Deepfake detection using timm."""
    
    def __init__(self, pretrained: bool = True):
        super(XceptionNet, self).__init__()
        
        # Load Xception model from timm
        # timm handles replacing the linear classification head automatically when num_classes is set.
        self.model = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=1  # Output single logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
