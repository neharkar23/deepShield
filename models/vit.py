import torch
import torch.nn as nn
import torchvision.models as models

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT-B/16) classifier for Deepfake detection."""
    
    def __init__(self, pretrained: bool = True):
        super(VisionTransformer, self).__init__()
        
        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.vit_b_16(weights=weights)
        
        # Replace heads classifier head: ViT heads are in self.model.heads
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, 1)  # Outputs logits for binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
