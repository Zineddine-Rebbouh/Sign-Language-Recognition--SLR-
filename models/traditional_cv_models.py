"""
Traditional Computer Vision Models for Sign Language Recognition.
Includes HOG+CNN, HOG+LBPH, Inception+HOG+LBPH+KNN.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional, Tuple, List
from pathlib import Path


def extract_lbph_features(
    frames: List[np.ndarray],
    radius: int = 1,
    n_points: int = 8
) -> np.ndarray:
    """
    Extract LBPH (Local Binary Pattern Histogram) features.
    
    Args:
        frames: List of preprocessed frames
        radius: Radius for LBP
        n_points: Number of points for LBP
    
    Returns:
        Array of LBPH features
    """
    features = []
    
    for frame in frames:
        # Convert to uint8 if normalized
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
        
        # Calculate LBP
        lbp = np.zeros_like(frame)
        h, w = frame.shape
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = frame[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if frame[x, y] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalize
        features.append(hist)
    
    return np.array(features)


class HOGCNN(nn.Module):
    """HOG features + CNN classifier."""
    
    def __init__(
        self,
        num_classes: int,
        hog_feature_dim: int,
        cnn_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5
    ):
        super(HOGCNN, self).__init__()
        
        # CNN layers for HOG features
        layers = []
        in_dim = hog_feature_dim
        
        for hidden_dim in cnn_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Linear(in_dim, num_classes)
    
    def forward(self, hog_features):
        # hog_features: (batch, hog_feature_dim)
        features = self.feature_extractor(hog_features)
        output = self.classifier(features)
        return output


class HOGLBPHClassifier:
    """HOG + LBPH features with classifier."""
    
    def __init__(self, classifier_type: str = 'svm', **classifier_kwargs):
        """
        Initialize HOG+LBPH classifier.
        
        Args:
            classifier_type: 'svm' or 'knn'
            **classifier_kwargs: Arguments for classifier
        """
        self.classifier_type = classifier_type
        
        if classifier_type == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(probability=True, **classifier_kwargs)
        elif classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(**classifier_kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def fit(self, hog_features: np.ndarray, lbph_features: np.ndarray, labels: np.ndarray):
        """Train classifier on combined HOG+LBPH features."""
        # Concatenate features
        combined_features = np.concatenate([hog_features, lbph_features], axis=1)
        self.classifier.fit(combined_features, labels)
    
    def predict(self, hog_features: np.ndarray, lbph_features: np.ndarray):
        """Predict using combined features."""
        combined_features = np.concatenate([hog_features, lbph_features], axis=1)
        return self.classifier.predict(combined_features)
    
    def predict_proba(self, hog_features: np.ndarray, lbph_features: np.ndarray):
        """Predict probabilities."""
        combined_features = np.concatenate([hog_features, lbph_features], axis=1)
        return self.classifier.predict_proba(combined_features)


class InceptionHOGLBPHKNN:
    """Inception features + HOG + LBPH + KNN classifier."""
    
    def __init__(
        self,
        num_classes: int,
        k: int = 5,
        weights: str = 'distance'
    ):
        """
        Initialize Inception+HOG+LBPH+KNN model.
        
        Args:
            num_classes: Number of classes
            k: Number of neighbors for KNN
            weights: KNN weights ('uniform' or 'distance')
        """
        self.num_classes = num_classes
        self.knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        
        # Inception model for feature extraction
        try:
            import torchvision.models as models
            inception = models.inception_v3(pretrained=True, aux_logits=False)
            inception.fc = nn.Identity()
            self.inception = inception.eval()
        except:
            self.inception = None
            print("Warning: Inception model not available, will use HOG+LBPH only")
    
    def extract_inception_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract Inception features from frames."""
        if self.inception is None:
            return np.array([])
        
        import torch
        from torchvision import transforms
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        features = []
        with torch.no_grad():
            for frame in frames:
                # Convert grayscale to RGB
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # Transform and extract features
                frame_tensor = transform(frame).unsqueeze(0)
                feat = self.inception(frame_tensor)
                features.append(feat.squeeze().numpy())
        
        return np.array(features)
    
    def fit(
        self,
        frames: List[np.ndarray],
        hog_features: np.ndarray,
        lbph_features: np.ndarray,
        labels: np.ndarray
    ):
        """Train KNN on combined features."""
        # Extract Inception features
        inception_features = self.extract_inception_features(frames)
        
        # Combine all features
        if inception_features.size > 0:
            combined_features = np.concatenate([
                inception_features,
                hog_features,
                lbph_features
            ], axis=1)
        else:
            combined_features = np.concatenate([hog_features, lbph_features], axis=1)
        
        self.knn.fit(combined_features, labels)
    
    def predict(
        self,
        frames: List[np.ndarray],
        hog_features: np.ndarray,
        lbph_features: np.ndarray
    ):
        """Predict using combined features."""
        inception_features = self.extract_inception_features(frames)
        
        if inception_features.size > 0:
            combined_features = np.concatenate([
                inception_features,
                hog_features,
                lbph_features
            ], axis=1)
        else:
            combined_features = np.concatenate([hog_features, lbph_features], axis=1)
        
        return self.knn.predict(combined_features)
    
    def predict_proba(
        self,
        frames: List[np.ndarray],
        hog_features: np.ndarray,
        lbph_features: np.ndarray
    ):
        """Predict probabilities."""
        inception_features = self.extract_inception_features(frames)
        
        if inception_features.size > 0:
            combined_features = np.concatenate([
                inception_features,
                hog_features,
                lbph_features
            ], axis=1)
        else:
            combined_features = np.concatenate([hog_features, lbph_features], axis=1)
        
        return self.knn.predict_proba(combined_features)

