"""
Sequence Models for Sign Language Recognition.
Includes CNN+LSTM, CNN+GRU, 2D CNN→LSTM.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class CNNLSTM(nn.Module):
    """CNN + LSTM for sequence-based classification."""
    
    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = 'resnet18',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super(CNNLSTM, self).__init__()
        
        # CNN backbone
        if cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            cnn.fc = nn.Identity()
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])  # Remove final fc
            cnn_features = 512
        elif cnn_backbone == 'resnet34':
            cnn = models.resnet34(pretrained=pretrained)
            cnn.fc = nn.Identity()
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            cnn_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame)
            features = features.view(batch_size, -1)  # Flatten
            cnn_features.append(features)
        
        # Stack CNN features: (batch, seq_len, features)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use last output
        output = self.fc(lstm_out[:, -1, :])
        
        return output


class CNNGRU(nn.Module):
    """CNN + GRU for sequence-based classification."""
    
    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = 'resnet18',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super(CNNGRU, self).__init__()
        
        # CNN backbone
        if cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            cnn.fc = nn.Identity()
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            cnn_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # GRU
        self.gru = nn.GRU(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process frames through CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame)
            features = features.view(batch_size, -1)
            cnn_features.append(features)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # GRU
        gru_out, h_n = self.gru(cnn_features)
        
        # Use last output
        output = self.fc(gru_out[:, -1, :])
        
        return output


class CNN2DLSTM(nn.Module):
    """2D CNN → LSTM for sequence-based classification."""
    
    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = 'resnet18',
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super(CNN2DLSTM, self).__init__()
        
        # 2D CNN backbone
        if cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            cnn.fc = nn.Identity()
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            cnn_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Extract features from each frame
        features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            feat = self.cnn(frame)
            feat = feat.view(batch_size, -1)
            features.append(feat)
        
        # Stack: (batch, seq_len, features)
        features = torch.stack(features, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Classify using last output
        output = self.fc(lstm_out[:, -1, :])
        
        return output

