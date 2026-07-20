"""
Spatiotemporal Models for Sign Language Recognition.
Includes 3D CNN, C3D, I3D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv3DBlock(nn.Module):
    """3D Convolutional block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Simple3DCNN(nn.Module):
    """Simple 3D CNN for video classification."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = Conv3DBlock(input_channels, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = Conv3DBlock(64, 128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = Conv3DBlock(128, 256, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4 = Conv3DBlock(256, 512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class C3D(nn.Module):
    """C3D (Convolutional 3D) model for video classification."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(C3D, self).__init__()
        
        # C3D architecture
        self.conv1 = Conv3DBlock(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = Conv3DBlock(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = Conv3DBlock(128, 256, kernel_size=3, padding=1)
        self.conv3b = Conv3DBlock(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = Conv3DBlock(256, 512, kernel_size=3, padding=1)
        self.conv4b = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.conv5b = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class I3D(nn.Module):
    """I3D (Inflated 3D ConvNet) - Simplified version."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super(I3D, self).__init__()
        
        # Simplified I3D architecture
        # In practice, I3D uses Inception modules inflated to 3D
        # This is a simplified version
        
        # Stem
        self.conv1 = Conv3DBlock(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Inception-like blocks (simplified)
        self.conv2 = Conv3DBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3a = Conv3DBlock(192, 384, kernel_size=1)
        self.conv3b = Conv3DBlock(192, 96, kernel_size=1)
        self.conv3c = Conv3DBlock(96, 208, kernel_size=3, padding=1)
        
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        
        self.conv4a = Conv3DBlock(384 + 208, 512, kernel_size=1)
        self.conv4b = Conv3DBlock(384 + 208, 112, kernel_size=1)
        self.conv4c = Conv3DBlock(112, 224, kernel_size=3, padding=1)
        
        self.pool4 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 224, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        branch1 = self.conv3a(x)
        branch2 = self.conv3c(self.conv3b(x))
        x = torch.cat([branch1, branch2], dim=1)
        x = self.pool3(x)
        
        branch1 = self.conv4a(x)
        branch2 = self.conv4c(self.conv4b(x))
        x = torch.cat([branch1, branch2], dim=1)
        x = self.pool4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

