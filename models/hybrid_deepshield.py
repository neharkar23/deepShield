import torch
import torch.nn as nn
import torchvision.models as models
from src.attention import CBAM
from src.feature_fusion import FrequencyFeatureExtractor, FeatureFusion

class HybridDeepShield(nn.Module):
    """
    DeepShield Hybrid Network (Primary Model).
    Fuses spatial features (EfficientNet-B3) with frequency domain features (FFT + CNN)
    and applies a Convolutional Block Attention Module (CBAM) before classification.
    """
    
    def __init__(self, pretrained: bool = True):
        super(HybridDeepShield, self).__init__()
        
        # 1. Spatial Backbone (EfficientNet-B3)
        if pretrained:
            weights = models.EfficientNet_B3_Weights.DEFAULT
        else:
            weights = None
        
        efficientnet_b3 = models.efficientnet_b3(weights=weights)
        self.spatial_backbone = efficientnet_b3.features  # Outputs feature map of shape: [B, 1536, 7, 7]
        
        # 2. Frequency Backbone (2D FFT + 4-layer CNN)
        self.frequency_backbone = FrequencyFeatureExtractor(
            in_channels=1, 
            out_channels=256
        )  # Outputs feature map of shape: [B, 256, 7, 7]
        
        # 3. Feature Fusion Layer
        self.fusion = FeatureFusion(
            spatial_channels=1536, 
            frequency_channels=256
        )  # Concatenates along channels: [B, 1792, 7, 7]
        
        # 4. Attention Block (CBAM)
        self.attention = CBAM(
            in_channels=1792, 
            reduction_ratio=16
        )
        
        # 5. Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)  # Outputs raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [B, 3, 224, 224]
        Returns:
            Binary classification logit of shape [B, 1]
        """
        # Extract spatial features: [B, 1536, 7, 7]
        spatial_feats = self.spatial_backbone(x)
        
        # Extract frequency features: [B, 256, 7, 7]
        freq_feats = self.frequency_backbone(x)
        
        # Fuse spatial and frequency features: [B, 1792, 7, 7]
        fused_feats = self.fusion(spatial_feats, freq_feats)
        
        # Apply CBAM attention: [B, 1792, 7, 7]
        attended_feats = self.attention(fused_feats)
        
        # Pool and classify
        pooled = self.global_pool(attended_feats)  # [B, 1792, 1, 1]
        logits = self.classifier(pooled)           # [B, 1]
        
        return logits
