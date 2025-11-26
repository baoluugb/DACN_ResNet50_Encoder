import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet34, resnet50, ResNet
from torchvision.models.feature_extraction import create_feature_extractor


class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding for spatial features

    Adds learnable positional information to 2D feature maps from CNN encoder.
    Uses sinusoidal encoding similar to original Transformer but extended to 2D.
    """

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 256):
        super().__init__()

        # Create 2D positional encoding
        pe = torch.zeros(max_h, max_w, d_model)

        # Encoding for height and width dimensions
        pos_h = torch.arange(
            0, max_h, dtype=torch.float).unsqueeze(1)  # (H, 1)
        pos_w = torch.arange(
            0, max_w, dtype=torch.float).unsqueeze(1)  # (W, 1)

        # Create different frequency components for each dimension
        # Split d_model into 2 halves: first half for height, second half for width
        div_term = torch.exp(torch.arange(
            0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))

        # Height encoding: use first half of channels
        pe[:, :, 0:d_model //
            2:2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, 1:d_model //
            2:2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_w, 1)

        # Width encoding: use second half of channels
        pe[:, :, d_model //
            2::2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(max_h, 1, 1)
        pe[:, :, d_model//2 +
            1::2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(max_h, 1, 1)

        self.register_buffer('pe', pe)  # (max_h, max_w, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, d_model) feature map
        Returns:
            x + positional encoding: (B, H, W, d_model)
        """
        B, H, W, C = x.shape
        return x + self.pe[:H, :W, :C].unsqueeze(0)


class DenseBlock(nn.Module):
    """Dense block with multiple conv layers - DEPRECATED"""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(self._make_layer(
                in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """Transition layer between dense blocks (downsampling + channel reduction) - DEPRECATED"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """ResNet-based encoder with 2D positional encoding

    Architecture:
    - ResNet backbone (resnet34 or resnet50) for feature extraction
    - Removes fully connected layer and average pooling from ResNet
    - Extracts features from the last convolutional layer (layer4)
    - 2D positional encoding applied to spatial features
    - Feature projection to d_model dimension
    """

    def __init__(self, d_model: int = 512, backbone: str = 'resnet34',
                 pos_max_h: int = 64, pos_max_w: int = 256, pretrained: bool = False):
        """
        Args:
            d_model: Output feature dimension
            backbone: ResNet backbone to use ('resnet34' or 'resnet50')
            pos_max_h: Maximum height for positional encoding
            pos_max_w: Maximum width for positional encoding
            pretrained: Whether to use pretrained ImageNet weights
        """
        super().__init__()

        self.d_model = d_model

        # Load ResNet backbone
        if backbone == 'resnet34':
            resnet = resnet34(pretrained=pretrained)
            self.resnet_channels = 512  # ResNet34 outputs 512 channels from layer4
        elif backbone == 'resnet50':
            # Use dilated convolutions in layer3 and layer4 to increase feature map resolution
            # This changes output from 1/32 (16x16) to 1/8 (64x64) of input size
            # Crucial for recognizing small mathematical symbols
            resnet = resnet50(pretrained=pretrained,
                              replace_stride_with_dilation=[False, True, True])
            self.resnet_channels = 2048  # ResNet50 outputs 2048 channels from layer4
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Extract feature extractor for layer4 (last convolutional layer)
        # This removes the avgpool and fc layers from ResNet
        self.feature_extractor = create_feature_extractor(
            resnet,
            return_nodes={'layer4': 'features'}
        )

        # 2D Positional encoding
        self.pos_encoding_2d = PositionalEncoding2D(
            d_model, max_h=pos_max_h, max_w=pos_max_w)

        # Project from ResNet output channels to d_model
        self.feature_proj = nn.Linear(self.resnet_channels, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) input images (can be 1 or 3 channels)
        Returns:
            features: (B, H_p * W_p, d_model) flattened spatial features
            (H_p, W_p): spatial dimensions of feature map
        """
        B, C, H_img, W_img = x.shape

        # Convert grayscale to RGB if needed (ResNet expects 3 channels)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features from ResNet layer4
        # Input: (B, 3, H, W) -> Output: (B, 512/2048, H/32, W/32)
        feature_dict = self.feature_extractor(x)
        x = feature_dict['features']  # (B, C_feat, H_p, W_p)

        B, C_feat, H_p, W_p = x.shape

        # Reshape: (B, C, H, W) -> (B, H, W, C) for positional encoding
        x = x.permute(0, 2, 3, 1)  # (B, H_p, W_p, C_feat)

        # Project to d_model
        x = self.feature_proj(x)  # (B, H_p, W_p, d_model)

        # Add 2D positional encoding
        x = self.pos_encoding_2d(x)  # (B, H_p, W_p, d_model)

        # Flatten spatial dimensions
        x = x.reshape(B, H_p * W_p, self.d_model)  # (B, H_p*W_p, d_model)

        return x, (H_p, W_p)
