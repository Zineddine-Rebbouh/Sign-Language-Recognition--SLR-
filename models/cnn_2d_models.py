"""
2D CNN Models for Sign Language Recognition.
Includes VGG, ResNet, MobileNet, AlexNet, InceptionV3.

All models accept a (batch, C, H, W) tensor and return (batch, num_classes) logits.
For video inputs, apply the model frame-by-frame or use sequence_models.py.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    VGG11_Weights, VGG16_Weights, VGG19_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    MobileNet_V2_Weights, AlexNet_Weights, Inception_V3_Weights,
)
from typing import Optional


class VGGModel(nn.Module):
    """VGG model variants for frame-based classification.

    Args:
        num_classes: Number of output classes.
        variant: One of 'vgg11', 'vgg16', 'vgg19'.
        pretrained: Load ImageNet pretrained weights.
    """

    _VARIANTS = {
        "vgg11": (models.vgg11, VGG11_Weights.DEFAULT),
        "vgg16": (models.vgg16, VGG16_Weights.DEFAULT),
        "vgg19": (models.vgg19, VGG19_Weights.DEFAULT),
    }

    def __init__(self, num_classes: int, variant: str = "vgg11", pretrained: bool = True):
        super(VGGModel, self).__init__()

        if variant not in self._VARIANTS:
            raise ValueError(f"Variant must be one of {list(self._VARIANTS.keys())}")

        fn, weights = self._VARIANTS[variant]
        vgg = fn(weights=weights if pretrained else None)

        # Replace classifier head
        num_features = vgg.classifier[0].in_features
        vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.model = vgg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNetModel(nn.Module):
    """ResNet model variants for frame-based classification.

    Args:
        num_classes: Number of output classes.
        variant: One of 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        pretrained: Load ImageNet pretrained weights.
    """

    _VARIANTS = {
        "resnet18":  (models.resnet18,  ResNet18_Weights.DEFAULT),
        "resnet34":  (models.resnet34,  ResNet34_Weights.DEFAULT),
        "resnet50":  (models.resnet50,  ResNet50_Weights.DEFAULT),
        "resnet101": (models.resnet101, ResNet101_Weights.DEFAULT),
    }

    def __init__(self, num_classes: int, variant: str = "resnet18", pretrained: bool = True):
        super(ResNetModel, self).__init__()

        if variant not in self._VARIANTS:
            raise ValueError(f"Variant must be one of {list(self._VARIANTS.keys())}")

        fn, weights = self._VARIANTS[variant]
        resnet = fn(weights=weights if pretrained else None)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.model = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MobileNetModel(nn.Module):
    """MobileNetV2 for frame-based classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(MobileNetModel, self).__init__()
        mobilenet = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        num_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes),
        )
        self.model = mobilenet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AlexNetModel(nn.Module):
    """AlexNet for frame-based classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(AlexNetModel, self).__init__()
        alexnet = models.alexnet(
            weights=AlexNet_Weights.DEFAULT if pretrained else None
        )
        num_features = alexnet.classifier[1].in_features
        alexnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.model = alexnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class InceptionV3Model(nn.Module):
    """InceptionV3 for frame-based classification (299×299 input required).

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(InceptionV3Model, self).__init__()
        inception = models.inception_v3(
            weights=Inception_V3_Weights.DEFAULT if pretrained else None,
            aux_logits=False,
        )
        inception.fc = nn.Linear(inception.fc.in_features, num_classes)
        self.model = inception

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_2d_cnn_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Factory function to instantiate a 2D CNN model by name.

    Args:
        model_name: One of 'vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34',
                    'resnet50', 'resnet101', 'mobilenet', 'alexnet', 'inceptionv3'.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()

    if name.startswith("vgg"):
        return VGGModel(num_classes, variant=name, pretrained=pretrained)
    elif name.startswith("resnet"):
        return ResNetModel(num_classes, variant=name, pretrained=pretrained)
    elif name == "mobilenet":
        return MobileNetModel(num_classes, pretrained=pretrained)
    elif name == "alexnet":
        return AlexNetModel(num_classes, pretrained=pretrained)
    elif name == "inceptionv3":
        return InceptionV3Model(num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: vgg11, vgg16, vgg19, "
            "resnet18, resnet34, resnet50, resnet101, mobilenet, alexnet, inceptionv3."
        )
