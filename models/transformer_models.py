"""
Transformer-based Models for Sign Language Recognition.
Includes CNN+Transformer, TimeSformer, VideoSwinTransformer (simplified).

Input format for all models: (batch, seq_len/num_frames, C, H, W)
Output: (batch, num_classes) logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# CNN + Transformer Encoder
# ---------------------------------------------------------------------------

class CNNTransformer(nn.Module):
    """ResNet-18 frame encoder + Transformer encoder for video classification.

    The ResNet extracts per-frame features; the Transformer attends over the
    temporal sequence.

    Args:
        num_classes:     Number of output classes.
        cnn_backbone:    Only 'resnet18' supported.
        d_model:         Transformer embedding dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of Transformer encoder layers.
        dim_feedforward: FFN intermediate dimension.
        dropout:         Dropout probability.
        pretrained:      Load ImageNet pretrained ResNet-18.

    Input:  (batch, seq_len, C, H, W)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = "resnet18",
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        if cnn_backbone != "resnet18":
            raise ValueError("Only 'resnet18' is supported as a backbone.")

        resnet = tv_models.resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # drop FC
        cnn_features = 512

        self.projection = nn.Linear(cnn_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # keep (batch, seq, dim) convention
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Batch all frames → single CNN pass
        frames = x.view(B * T, C, H, W)
        features = self.cnn(frames).view(B, T, -1)    # (B, T, 512)

        # Project → positional encoding → transformer
        features = self.projection(features)           # (B, T, d_model)
        features = self.pos_encoder(features)
        encoded = self.transformer_encoder(features)   # (B, T, d_model)

        # Use CLS-like: mean-pool over time
        pooled = encoded.mean(dim=1)                   # (B, d_model)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# TimeSformer (simplified)
# ---------------------------------------------------------------------------

class TimeSformer(nn.Module):
    """Simplified TimeSformer for video classification.

    Based on: "Is Space-Time Attention All You Need for Video Understanding?"
    (Bertasius et al., 2021). This is a simplified version that uses standard
    multi-head attention rather than the divided space-time attention of the
    original paper.

    Args:
        num_classes:     Number of output classes.
        img_size:        Spatial resolution of each frame.
        patch_size:      Size of each 2D patch (must divide img_size evenly).
        num_frames:      Expected number of input frames.
        d_model:         Embedding dimension (must be divisible by nhead).
        nhead:           Number of attention heads.
        num_layers:      Number of Transformer encoder layers.
        dim_feedforward: FFN intermediate dimension.
        dropout:         Dropout probability.

    Input:  (batch, num_frames, C=3, H=img_size, W=img_size)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_frames = num_frames
        self.d_model = d_model

        num_patches = (img_size // patch_size) ** 2
        # Per-frame patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        # Learnable positional and temporal embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Patch embed all frames at once
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)                          # (B*T, d, h_p, w_p)
        x = x.flatten(2).transpose(1, 2)                 # (B*T, num_patches, d)

        # Prepend CLS token
        cls = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B*T, num_patches+1, d)

        # Spatial positional embedding
        x = x + self.pos_embed                           # broadcast over B*T

        # Reshape for temporal embedding: (B, T, num_patches+1, d)
        num_tokens = x.size(1)
        x = x.view(B, T, num_tokens, self.d_model)
        x = x + self.temporal_embed[:, :T, :, :]        # add temporal embedding

        # Flatten tokens across time: (B, T*(num_patches+1), d)
        x = x.view(B, T * num_tokens, self.d_model)

        x = self.transformer(x)                          # (B, T*tokens, d)

        # Use CLS token of the first frame as the representation
        x = self.norm(x[:, 0, :])
        return self.head(x)


# ---------------------------------------------------------------------------
# VideoSwinTransformer (simplified)
# ---------------------------------------------------------------------------

class VideoSwinTransformer(nn.Module):
    """Simplified Video Swin Transformer for video classification.

    Note: The full Video Swin Transformer uses shifted window attention across
    space and time. This implementation uses standard full attention for
    simplicity. It captures the correct input/output interface and
    embedding strategy but not the exact shifted-window mechanism.

    Args:
        num_classes: Number of output classes.
        img_size:    Spatial resolution (H = W).
        patch_size:  Spatial patch stride.
        num_frames:  Expected number of input frames.
        embed_dim:   Base embedding dimension.
        num_heads:   Attention heads per layer.
        num_layers:  Number of Transformer encoder layers.
        dropout:     Dropout probability.

    Input:  (batch, num_frames, C=3, H, W)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 4,
        num_frames: int = 8,
        embed_dim: int = 96,
        num_heads: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Spatio-temporal patch embedding (1 frame at a time spatially)
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # (B, T, C, H, W) → (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.patch_embed(x)                 # (B, embed, T, h_p, w_p)

        # Flatten spatial+temporal tokens: (B, embed, T*h_p*w_p)
        x = x.flatten(2).transpose(1, 2)        # (B, tokens, embed)

        x = self.transformer(x)                 # (B, tokens, embed)
        x = x.mean(dim=1)                       # global average pool
        x = self.norm(x)
        return self.head(x)
