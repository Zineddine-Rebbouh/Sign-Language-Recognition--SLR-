"""
Keypoint-based Models for Sign Language Recognition.
Uses MediaPipe hand keypoints (21 joints × 3 coordinates).
Includes GCN, ST-GCN, KeypointLSTM.

All models expect keypoint tensors, NOT raw video frames.
Use preprocessing.py → extract_mediapipe_keypoints() to produce inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Graph Convolutional Layer
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """Single graph convolutional layer: H' = A · H · W.

    Args:
        in_features:  Input feature dimension per node.
        out_features: Output feature dimension per node.
        bias:         Whether to include a learnable bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes)  normalised adjacency matrix

        Returns:
            (batch, num_nodes, out_features)
        """
        support = torch.matmul(x, self.weight)      # (B, N, out)
        out = torch.matmul(adj, support)             # (B, N, out)  — adj broadcasts
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# GCN — spatial only, averages over time
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """Graph Convolutional Network for skeleton-based sign recognition.

    Processes each frame independently through GCN layers,
    then averages spatial features over time before classification.

    Args:
        num_classes:     Number of sign classes.
        num_joints:      Nodes in the graph (21 for MediaPipe single hand).
        in_channels:     Features per node (3 for x, y, z).
        hidden_channels: List of GCN output dims per layer.
        dropout:         Dropout probability.

    Input:  (batch, num_frames, num_joints, in_channels)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 21,
        in_channels: int = 3,
        hidden_channels: List[int] = [64, 128, 256],
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_joints = num_joints

        # Learnable adjacency (initialised to fully connected)
        adj_init = torch.ones(num_joints, num_joints) / num_joints
        self.register_buffer("adj", adj_init)

        # Build GCN stack
        gc_layers: List[nn.Module] = []
        in_dim = in_channels
        for out_dim in hidden_channels:
            gc_layers.append(GraphConvolution(in_dim, out_dim))
            gc_layers.append(nn.ReLU())
            gc_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.gc_layers = nn.ModuleList(gc_layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def _apply_gcn(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C) → (B, N, C')"""
        for layer in self.gc_layers:
            if isinstance(layer, GraphConvolution):
                x = layer(x, self.adj)
            else:
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        # Process all frames in a single batch operation
        x = x.view(B * T, N, C)
        x = self._apply_gcn(x)           # (B*T, N, C')
        x = x.mean(dim=1)                # (B*T, C')  — pool over joints
        x = x.view(B, T, -1).mean(dim=1) # (B, C')   — pool over time
        return self.fc(x)


# ---------------------------------------------------------------------------
# ST-GCN — spatial + temporal convolution (fixed)
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):
    """One Spatial-Temporal GCN block: graph conv → temporal conv → BN → ReLU.

    Args:
        in_channels:          Input channels per node per frame.
        out_channels:         Output channels.
        num_joints:           Number of skeleton nodes.
        temporal_kernel_size: Kernel size for the 1-D temporal conv.
        dropout:              Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int,
        temporal_kernel_size: int = 9,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        padding = temporal_kernel_size // 2
        # Input to temporal conv: (B*N, out_channels, T)
        self.tcn = nn.Conv1d(out_channels, out_channels, temporal_kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N, C_in)
            adj: (N, N)

        Returns:
            (B, T, N, C_out)
        """
        B, T, N, C = x.shape

        # --- Spatial graph conv ---
        x_flat = x.view(B * T, N, C)
        x_flat = self.gcn(x_flat, adj)           # (B*T, N, C_out)
        x_s = x_flat.view(B, T, N, -1)          # (B, T, N, C_out)

        # --- Temporal conv over T for each (B, N) pair ---
        C_out = x_s.shape[-1]
        # (B, T, N, C_out) → (B*N, C_out, T)
        x_t = x_s.permute(0, 2, 3, 1).contiguous().view(B * N, C_out, T)
        x_t = self.tcn(x_t)                      # (B*N, C_out, T)
        x_t = self.bn(x_t)

        # --- Residual ---
        # reshape original x for residual: (B*N, C_in, T)
        x_res = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)
        x_res = self.residual(x_res)              # (B*N, C_out, T)
        x_t = x_t + x_res

        x_t = self.relu(x_t)
        x_t = self.drop(x_t)

        # (B*N, C_out, T) → (B, T, N, C_out)
        x_out = x_t.view(B, N, C_out, T).permute(0, 3, 1, 2).contiguous()
        return x_out


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for sign recognition.

    Args:
        num_classes:          Number of sign classes.
        num_joints:           Number of skeleton nodes (21 for MediaPipe hand).
        in_channels:          Features per node (3 for x, y, z).
        hidden_channels:      Channel sizes for each ST-GCN block.
        dropout:              Dropout probability.
        temporal_kernel_size: Temporal convolution kernel size.

    Input:  (batch, num_frames, num_joints, in_channels)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 21,
        in_channels: int = 3,
        hidden_channels: List[int] = [64, 64, 128, 128, 256, 256],
        dropout: float = 0.5,
        temporal_kernel_size: int = 9,
    ):
        super().__init__()
        self.num_joints = num_joints

        adj_init = torch.ones(num_joints, num_joints) / num_joints
        self.register_buffer("adj", adj_init)

        blocks: List[STGCNBlock] = []
        in_ch = in_channels
        for out_ch in hidden_channels:
            blocks.append(
                STGCNBlock(in_ch, out_ch, num_joints, temporal_kernel_size, dropout)
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        for block in self.blocks:
            x = block(x, self.adj)

        # Global average pooling over T and N
        x = x.mean(dim=(1, 2))   # (B, C_out)
        return self.fc(x)


# ---------------------------------------------------------------------------
# KeypointLSTM — flatten joints, feed sequence to LSTM
# ---------------------------------------------------------------------------

class KeypointLSTM(nn.Module):
    """LSTM classifier on flattened keypoint sequences.

    Args:
        num_classes: Number of sign classes.
        num_joints:  Number of skeleton nodes per frame.
        in_channels: Features per node.
        hidden_size: LSTM hidden dimension.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout probability.

    Input:  (batch, num_frames, num_joints, in_channels)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 21,
        in_channels: int = 3,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        input_size = num_joints * in_channels

        self.lstm = nn.LSTM(
            input_size=input_size,
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
        B, T, N, C = x.shape
        x = x.view(B, T, N * C)        # flatten joints: (B, T, N*C)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---------------------------------------------------------------------------
# MediaPipe extraction helper
# ---------------------------------------------------------------------------

def extract_mediapipe_keypoints(
    video_path,
    num_frames: Optional[int] = None,
    confidence: float = 0.5,
) -> np.ndarray:
    """Extract hand keypoints frame-by-frame using MediaPipe Hands.

    Args:
        video_path: Path to the video file (str or Path).
        num_frames: If set, sample exactly this many evenly-spaced frames.
        confidence: MediaPipe min_detection_confidence threshold.

    Returns:
        keypoints: float32 array of shape (actual_frames, 21, 3).
                   Frames where no hand is detected are filled with zeros.

    Raises:
        ImportError: If mediapipe is not installed.
        FileNotFoundError: If the video file does not exist.
    """
    import cv2
    from pathlib import Path

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "MediaPipe is required for keypoint extraction. "
            "Install it with: pip install mediapipe"
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build frame index list
    if num_frames is not None and num_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence,
    )

    keypoints_list = []
    frame_idx = 0
    ptr = 0

    try:
        while cap.isOpened() and ptr < len(indices):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == indices[ptr]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    lms = result.multi_hand_landmarks[0].landmark
                    kp = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                else:
                    kp = np.zeros((21, 3), dtype=np.float32)

                keypoints_list.append(kp)
                ptr += 1

            frame_idx += 1
    finally:
        cap.release()
        hands.close()

    if len(keypoints_list) == 0:
        return np.zeros((1, 21, 3), dtype=np.float32)

    return np.stack(keypoints_list, axis=0)   # (T, 21, 3)
