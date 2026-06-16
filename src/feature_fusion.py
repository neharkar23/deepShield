import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyFeatureExtractor(nn.Module):
    """
    Extracts frequency-domain features from an RGB image using 2D FFT 
    and a small Convolutional Neural Network.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super(FrequencyFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)  # 28 -> 7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB tensor to grayscale dynamically: Shape: [B, 3, 224, 224] -> [B, 224, 224]
        # Using standard ITU-R BT.601 coefficients
        x_gray = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        
        # Apply 2D FFT
        fft = torch.fft.fft2(x_gray)
        fft_shift = torch.fft.fftshift(fft)
        
        # Calculate amplitude and log transform for scaling
        amplitude = torch.abs(fft_shift)
        log_spectrum = torch.log(amplitude + 1e-8)
        
        # Add channel dimension: Shape: [B, 1, 224, 224]
        log_spectrum = log_spectrum.unsqueeze(1)
        
        # Pass log spectrum through Frequency CNN
        freq_features = self.conv1(log_spectrum)
        freq_features = self.conv2(freq_features)
        freq_features = self.conv3(freq_features)
        freq_features = self.conv4(freq_features)
        
        return freq_features


class FeatureFusion(nn.Module):
    """
    Fuses spatial and frequency features by concatenating them along the channel dimension.
    """
    
    def __init__(self, spatial_channels: int = 1536, frequency_channels: int = 256):
        super(FeatureFusion, self).__init__()
        self.spatial_channels = spatial_channels
        self.frequency_channels = frequency_channels
        self.fused_channels = spatial_channels + frequency_channels

    def forward(self, spatial_feats: torch.Tensor, freq_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_feats: Tensor of shape [B, spatial_channels, H, W]
            freq_feats: Tensor of shape [B, frequency_channels, H, W]
        """
        # Ensure spatial resolutions match before concatenation
        if spatial_feats.shape[2:] != freq_feats.shape[2:]:
            freq_feats = F.interpolate(freq_feats, size=spatial_feats.shape[2:], mode='bilinear', align_corners=False)
            
        fused_feats = torch.cat([spatial_feats, freq_feats], dim=1)
        return fused_feats
