import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    """EfficientNet-B0 baseline classifier for Deepfake detection."""
    
    def __init__(self, pretrained: bool = True):
        super(EfficientNetB0, self).__init__()
        
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.efficientnet_b0(weights=weights)
        
        # Replace classifier head: EfficientNet classifier is nn.Sequential(Dropout, Linear)
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1)  # Outputs logits for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
