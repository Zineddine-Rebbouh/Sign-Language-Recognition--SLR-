"""
2D CNN Models for Sign Language Recognition.
Includes VGG, ResNet, MobileNet, AlexNet, InceptionV3.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class VGGModel(nn.Module):
    """VGG model variants for frame-based classification."""
    
    def __init__(self, num_classes: int, variant: str = 'vgg11', pretrained: bool = True):
        super(VGGModel, self).__init__()
        
        variants = {
            'vgg11': models.vgg11,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19
        }
        
        if variant not in variants:
            raise ValueError(f"Variant must be one of {list(variants.keys())}")
        
        vgg = variants[variant](pretrained=pretrained)
        
        # Replace classifier
        num_features = vgg.classifier[0].in_features
        vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.model = vgg
    
    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    """ResNet model variants for frame-based classification."""
    
    def __init__(self, num_classes: int, variant: str = 'resnet18', pretrained: bool = True):
        super(ResNetModel, self).__init__()
        
        variants = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101
        }
        
        if variant not in variants:
            raise ValueError(f"Variant must be one of {list(variants.keys())}")
        
        resnet = variants[variant](pretrained=pretrained)
        
        # Replace classifier
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        
        self.model = resnet
    
    def forward(self, x):
        return self.model(x)


class MobileNetModel(nn.Module):
    """MobileNet model for frame-based classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(MobileNetModel, self).__init__()
        
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier
        num_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        self.model = mobilenet
    
    def forward(self, x):
        return self.model(x)


class AlexNetModel(nn.Module):
    """AlexNet model for frame-based classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(AlexNetModel, self).__init__()
        
        alexnet = models.alexnet(pretrained=pretrained)
        
        # Replace classifier
        num_features = alexnet.classifier[1].in_features
        alexnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        
        self.model = alexnet
    
    def forward(self, x):
        return self.model(x)


class InceptionV3Model(nn.Module):
    """InceptionV3 model for frame-based classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(InceptionV3Model, self).__init__()
        
        inception = models.inception_v3(pretrained=pretrained, aux_logits=False)
        
        # Replace classifier
        num_features = inception.fc.in_features
        inception.fc = nn.Linear(num_features, num_classes)
        
        self.model = inception
    
    def forward(self, x):
        return self.model(x)


def get_2d_cnn_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Factory function to get 2D CNN models.
    
    Args:
        model_name: Name of the model (vgg11, vgg16, vgg19, resnet18, resnet34, 
                   resnet50, resnet101, mobilenet, alexnet, inceptionv3)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower.startswith('vgg'):
        variant = model_name_lower
        return VGGModel(num_classes, variant=variant, pretrained=pretrained)
    
    elif model_name_lower.startswith('resnet'):
        variant = model_name_lower
        return ResNetModel(num_classes, variant=variant, pretrained=pretrained)
    
    elif model_name_lower == 'mobilenet':
        return MobileNetModel(num_classes, pretrained=pretrained)
    
    elif model_name_lower == 'alexnet':
        return AlexNetModel(num_classes, pretrained=pretrained)
    
    elif model_name_lower == 'inceptionv3':
        return InceptionV3Model(num_classes, pretrained=pretrained)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

