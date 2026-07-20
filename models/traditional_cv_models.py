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
    n_points: int = 8,
) -> np.ndarray:
    """Extract LBP histogram features using scikit-image (fast C implementation).

    Falls back to a pure-Python implementation if scikit-image is not available,
    but the pure-Python path is very slow (O(H×W) per frame) and should only be
    used for tiny test cases.

    Args:
        frames: List of preprocessed grayscale frames (H, W), dtype float32 [0,1]
                or uint8 [0,255].
        radius: Radius for the circular LBP neighbourhood.
        n_points: Number of circularly-symmetric neighbour set points.

    Returns:
        Feature array of shape (n_frames, 256).
    """
    try:
        from skimage.feature import local_binary_pattern
        _use_skimage = True
    except ImportError:
        _use_skimage = False

    features = []

    for frame in frames:
        # Normalise to uint8
        if frame.dtype != np.uint8:
            frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            frame_u8 = frame

        if _use_skimage:
            lbp = local_binary_pattern(frame_u8, n_points, radius, method="uniform")
            # uniform LBP has n_points+2 possible values; bin into 256 for consistency
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, n_points + 2))
        else:
            # Pure-Python fallback (slow — O(H×W) per frame)
            h, w = frame_u8.shape
            lbp = np.zeros((h, w), dtype=np.uint8)
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = int(frame_u8[i, j])
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        xi = int(round(i + radius * np.cos(angle)))
                        yi = int(round(j + radius * np.sin(angle)))
                        xi = np.clip(xi, 0, h - 1)
                        yi = np.clip(yi, 0, w - 1)
                        if int(frame_u8[xi, yi]) >= center:
                            code |= 1 << k
                    lbp[i, j] = code
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7  # L1 normalise
        features.append(hist)

    return np.array(features)


class HOGCNN(nn.Module):
    """MLP classifier on top of HOG features (misleadingly named 'HOG+CNN').

    This is a fully-connected network, not a convolutional one.
    The name follows the tutorial assignment naming convention.

    Args:
        num_classes: Number of output classes.
        hog_feature_dim: Dimensionality of the input HOG feature vector.
        cnn_hidden_dims: Hidden layer sizes.
        dropout: Dropout probability.

    Input:  (batch, hog_feature_dim)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        hog_feature_dim: int,
        cnn_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
    ):
        super(HOGCNN, self).__init__()

        layers: List[nn.Module] = []
        in_dim = hog_feature_dim
        for hidden_dim in cnn_hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, hog_features: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(hog_features)
        return self.classifier(features)


class HOGLBPHClassifier:
    """HOG + LBPH features with an sklearn classifier (SVM or KNN).

    Args:
        classifier_type: 'svm' or 'knn'.
        **classifier_kwargs: Keyword arguments forwarded to the sklearn classifier.
    """

    def __init__(self, classifier_type: str = "svm", **classifier_kwargs):
        self.classifier_type = classifier_type

        if classifier_type == "svm":
            from sklearn.svm import SVC
            self.classifier = SVC(probability=True, **classifier_kwargs)
        elif classifier_type == "knn":
            self.classifier = KNeighborsClassifier(**classifier_kwargs)
        else:
            raise ValueError(f"Unknown classifier type '{classifier_type}'. Choose 'svm' or 'knn'.")

    def _combine(self, hog: np.ndarray, lbph: np.ndarray) -> np.ndarray:
        return np.concatenate([hog, lbph], axis=1)

    def fit(self, hog_features: np.ndarray, lbph_features: np.ndarray, labels: np.ndarray):
        """Train classifier on concatenated HOG+LBPH features."""
        self.classifier.fit(self._combine(hog_features, lbph_features), labels)

    def predict(self, hog_features: np.ndarray, lbph_features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self._combine(hog_features, lbph_features))

    def predict_proba(self, hog_features: np.ndarray, lbph_features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self._combine(hog_features, lbph_features))


class InceptionHOGLBPHKNN:
    """Inception deep features + HOG + LBPH combined with a KNN classifier.

    If torchvision is not available the Inception branch is silently skipped
    and only HOG+LBPH features are used.

    Args:
        num_classes: Number of sign language classes.
        k: Number of neighbours for KNN.
        weights: KNN weight function ('uniform' or 'distance').
    """

    def __init__(self, num_classes: int, k: int = 5, weights: str = "distance"):
        self.num_classes = num_classes
        self.knn = KNeighborsClassifier(n_neighbors=k, weights=weights)

        self.inception = None
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
            inception.fc = nn.Identity()
            self.inception = inception.eval()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Inception model not available ({exc}). Using HOG+LBPH only.")

    def extract_inception_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract 2048-dim Inception feature vectors from a list of frames."""
        if self.inception is None:
            return np.empty((0,))

        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        features = []
        with torch.no_grad():
            for frame in frames:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                tensor = transform(frame).unsqueeze(0)
                feat = self.inception(tensor)
                features.append(feat.squeeze().numpy())

        return np.array(features)

    def _combine(
        self,
        frames: List[np.ndarray],
        hog: np.ndarray,
        lbph: np.ndarray,
    ) -> np.ndarray:
        inc = self.extract_inception_features(frames)
        if inc.size > 0:
            return np.concatenate([inc, hog, lbph], axis=1)
        return np.concatenate([hog, lbph], axis=1)

    def fit(
        self,
        frames: List[np.ndarray],
        hog_features: np.ndarray,
        lbph_features: np.ndarray,
        labels: np.ndarray,
    ):
        """Train KNN on combined Inception+HOG+LBPH features."""
        self.knn.fit(self._combine(frames, hog_features, lbph_features), labels)

    def predict(
        self,
        frames: List[np.ndarray],
        hog_features: np.ndarray,
        lbph_features: np.ndarray,
    ) -> np.ndarray:
        return self.knn.predict(self._combine(frames, hog_features, lbph_features))

    def predict_proba(
        self,
        frames: List[np.ndarray],
        hog_features: np.ndarray,
        lbph_features: np.ndarray,
    ) -> np.ndarray:
        return self.knn.predict_proba(self._combine(frames, hog_features, lbph_features))
