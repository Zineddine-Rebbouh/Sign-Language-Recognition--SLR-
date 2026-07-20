"""
Transformer Models for Sign Language Recognition.
Includes CNN+Transformer, TimeSformer, Video Swin Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class CNNTransformer(nn.Module):
    """CNN backbone + Transformer encoder for video classification."""
    
    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = 'resnet18',
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super(CNNTransformer, self).__init__()
        
        # CNN backbone
        import torchvision.models as models
        if cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            cnn.fc = nn.Identity()
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            cnn_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # Projection to d_model
        self.projection = nn.Linear(cnn_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Extract CNN features for each frame
        features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            feat = self.cnn(frame)
            feat = feat.view(batch_size, -1)
            features.append(feat)
        
        # Stack: (batch, seq_len, features)
        features = torch.stack(features, dim=1)
        
        # Project to d_model
        features = self.projection(features)  # (batch, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch, d_model)
        features = features.transpose(0, 1)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer encoder
        encoded = self.transformer_encoder(features)
        
        # Use last output: (batch, d_model)
        output = encoded[-1, :, :]
        
        # Classify
        output = self.fc(output)
        
        return output


class TimeSformer(nn.Module):
    """TimeSformer: Is Space-Time Attention All You Need for Video Understanding?"""
    
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        attention_type: str = 'divided_space_time'
    ):
        super(TimeSformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))  # +1 for cls token
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Classifier
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # x shape: (batch, num_frames, channels, height, width)
        B, T, C, H, W = x.shape
        
        # Patch embedding: (batch*T, d_model, h, w)
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)  # (batch*T, d_model, h_patches, w_patches)
        
        # Flatten patches: (batch*T, num_patches, d_model)
        num_patches = x.size(2) * x.size(3)
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Reshape for temporal dimension: (batch, T, num_patches+1, d_model)
        x = x.view(B, T, num_patches + 1, self.d_model)
        
        # Add temporal embedding
        x = x + self.temporal_embed.unsqueeze(2)
        
        # Flatten: (batch, T*(num_patches+1), d_model)
        x = x.view(B, T * (num_patches + 1), self.d_model)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x.transpose(0, 1)).transpose(0, 1)
        
        # Use cls token
        x = self.norm(x[:, 0, :])  # (batch, d_model)
        
        # Classify
        output = self.head(x)
        
        return output


class VideoSwinTransformer(nn.Module):
    """Simplified Video Swin Transformer."""
    
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 4,
        num_frames: int = 8,
        embed_dim: int = 96,
        depths: list = [2, 2, 6, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 7,
        dropout: float = 0.1
    ):
        super(VideoSwinTransformer, self).__init__()
        
        # Simplified implementation
        # Full Video Swin is complex, this is a basic version
        
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Simplified transformer layers (using standard transformer)
        # In practice, Video Swin uses shifted window attention
        encoder_layers = nn.TransformerEncoderLayer(
            embed_dim, num_heads[0], embed_dim * 4, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, depths[0])
        
        # Classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch, num_frames, channels, height, width)
        B, T, C, H, W = x.shape
        
        # Patch embedding: (batch, embed_dim, T, h_patches, w_patches)
        x = x.transpose(1, 2)  # (batch, channels, T, H, W)
        x = self.patch_embed(x)
        
        # Flatten: (batch, embed_dim, T*h_patches*w_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch, T*h_patches*w_patches, embed_dim)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Classify
        x = self.norm(x)
        output = self.head(x)
        
        return output

