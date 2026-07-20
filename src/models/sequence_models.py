"""
Sequence Models for Sign Language Recognition.
Includes CNN+LSTM, CNN+GRU, 2D CNN → LSTM.

Input format for all models: (batch, seq_len, C, H, W)
Output: (batch, num_classes) logits
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from typing import Tuple


def _make_resnet_backbone(variant: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Build a ResNet backbone with the FC head removed.

    Returns:
        (backbone, feature_dim)
    """
    if variant == "resnet18":
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = 512
    elif variant == "resnet34":
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        feature_dim = 512
    else:
        raise ValueError(f"Unsupported backbone '{variant}'. Choose 'resnet18' or 'resnet34'.")

    # Remove the final FC layer; output is (batch, feature_dim, 1, 1)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone, feature_dim


class CNNLSTM(nn.Module):
    """CNN backbone + LSTM for video sequence classification.

    Args:
        num_classes: Number of output classes.
        cnn_backbone: Backbone architecture ('resnet18' or 'resnet34').
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability.
        pretrained: Use ImageNet pretrained backbone weights.

    Input:  (batch, seq_len, C, H, W)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = "resnet18",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super(CNNLSTM, self).__init__()
        self.cnn, cnn_features = _make_resnet_backbone(cnn_backbone, pretrained)

        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, C, H, W)
        B, T, C, H, W = x.shape
        # Batch all frames together for a single CNN forward pass
        frames = x.view(B * T, C, H, W)
        features = self.cnn(frames)          # (B*T, feat, 1, 1)
        features = features.view(B, T, -1)   # (B, T, feat)

        lstm_out, _ = self.lstm(features)    # (B, T, hidden)
        return self.fc(lstm_out[:, -1, :])   # classify on last timestep


class CNNGRU(nn.Module):
    """CNN backbone + GRU for video sequence classification.

    Args:
        num_classes: Number of output classes.
        cnn_backbone: Backbone architecture ('resnet18').
        hidden_size: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability.
        pretrained: Use ImageNet pretrained backbone weights.

    Input:  (batch, seq_len, C, H, W)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = "resnet18",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super(CNNGRU, self).__init__()
        self.cnn, cnn_features = _make_resnet_backbone(cnn_backbone, pretrained)

        self.gru = nn.GRU(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        features = self.cnn(frames).view(B, T, -1)

        gru_out, _ = self.gru(features)
        return self.fc(gru_out[:, -1, :])


class CNN2DLSTM(nn.Module):
    """2D CNN → LSTM for video sequence classification.

    Functionally identical to CNNLSTM; kept as a named alias for the
    tutorial group assignments.

    Input:  (batch, seq_len, C, H, W)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = "resnet18",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super(CNN2DLSTM, self).__init__()
        self.cnn, cnn_features = _make_resnet_backbone(cnn_backbone, pretrained)

        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        features = self.cnn(frames).view(B, T, -1)

        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])
