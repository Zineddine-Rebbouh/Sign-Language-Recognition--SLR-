"""
Model loading and inference service for the SLR API.

This module is the ONLY place where the model is loaded and predictions are
made. The FastAPI app calls into this service — it never touches PyTorch
or preprocessing directly.

Key design decisions:
  - Uses the EXACT same ImprovedVideoProcessor from src/preprocessing.py
    (no duplicated preprocessing logic).
  - Model is loaded once at construction time, not per-request.
  - All input validation happens here so the API layer stays thin.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from api.schemas import PredictionResponse, TopKPrediction

logger = logging.getLogger(__name__)

# Allowed video extensions — must match what the training pipeline supports
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Minimum frames a video must produce to be usable
MIN_FRAMES = 5

# Maximum file size in bytes (100 MB — generous for short sign videos)
MAX_FILE_SIZE = 100 * 1024 * 1024


class ModelService:
    """Encapsulates model loading, preprocessing, and inference.

    Created once at app startup and shared across all requests.
    Thread-safe for read-only inference (no state mutation after init).
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """Load the model and preprocessing pipeline from a checkpoint.

        Args:
            checkpoint_path: Path to the best_model.pth checkpoint file.
            device: Force a specific device ("cpu" or "cuda"). If None,
                    auto-detects CUDA availability.
        """
        import sys
        # Ensure project root is on sys.path so src imports work
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.model_3dcnn import Improved3DCNN
        from src.preprocessing import ImprovedVideoProcessor

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Extract class mapping
        self.idx_to_class = checkpoint.get("class_mapping")
        if not self.idx_to_class:
            raise ValueError("Checkpoint has no 'class_mapping' key.")
        self.idx_to_class = {str(k): v for k, v in self.idx_to_class.items()}
        self.num_classes = len(self.idx_to_class)

        # Build model and load weights
        self.model = Improved3DCNN(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing — SAME ImprovedVideoProcessor used in training.
        # This is the single source of truth: target_size, num_frames,
        # hand detection, CLAHE — all match training exactly.
        saved_args = checkpoint.get("args", {})
        use_hand = not saved_args.get("no_hand_detection", False)

        try:
            self.processor = ImprovedVideoProcessor(
                target_size=(112, 112),
                num_frames=30,
                use_hand_detection=use_hand,
            )
        except ImportError:
            logger.warning("MediaPipe unavailable — falling back to full-frame mode.")
            self.processor = ImprovedVideoProcessor(
                target_size=(112, 112),
                num_frames=30,
                use_hand_detection=False,
            )
            use_hand = False

        self.use_hand_detection = use_hand
        logger.info(
            f"Model ready — {self.num_classes} classes, "
            f"{self.model.get_parameter_count() / 1e6:.2f}M params, "
            f"device={self.device}, hand_detection={use_hand}"
        )

    def validate_video_file(self, filename: str, file_size: int) -> Optional[str]:
        """Validate an uploaded video file before processing.

        Returns None if valid, or an error message string if invalid.
        """
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return (
                f"Unsupported file type '{ext}'. "
                f"Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Check file size
        if file_size > MAX_FILE_SIZE:
            return (
                f"File too large ({file_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            )

        if file_size == 0:
            return "Empty file uploaded."

        return None

    def predict(self, video_path: Path) -> PredictionResponse:
        """Run inference on a video file.

        This is the core prediction method. It:
          1. Validates the video is readable
          2. Runs the EXACT same preprocessing as training
          3. Runs the model forward pass
          4. Returns structured results

        Args:
            video_path: Path to a temporary video file on disk.

        Returns:
            PredictionResponse with predicted class, confidence, top-k, etc.

        Raises:
            ValueError: If the video can't be read or has too few frames.
        """
        start_time = time.time()

        # ── Step 1: Validate video readability ───────────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not read video file. The file may be corrupt or in an unsupported codec.")

        # Count total frames (quick check before full processing)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames < MIN_FRAMES:
            raise ValueError(
                f"Video too short: {total_frames} frames detected "
                f"(minimum {MIN_FRAMES} required for reliable prediction)."
            )

        # ── Step 2: Preprocess with the SHARED pipeline ─────
        # ImprovedVideoProcessor.process_video() handles:
        #   - Frame extraction
        #   - MediaPipe hand detection + ROI crop
        #   - CLAHE contrast enhancement
        #   - Resize to 112×112
        #   - Temporal padding/sampling to exactly 30 frames
        #   - Normalisation to [0, 1]
        frames = self.processor.process_video(str(video_path))

        # Check for hand detection (peek at a middle frame)
        hand_detected = False
        if self.use_hand_detection and self.processor.detector:
            try:
                probe_cap = cv2.VideoCapture(str(video_path))
                # Sample a frame from the middle of the video
                mid_frame_idx = total_frames // 2
                probe_cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
                ret, probe_frame = probe_cap.read()
                probe_cap.release()
                if ret:
                    rgb = cv2.cvtColor(probe_frame, cv2.COLOR_BGR2RGB)
                    bbox = self.processor.detector.find_hand_bbox(rgb)
                    hand_detected = bbox is not None
            except Exception:
                pass  # Non-critical — just report False

        # ── Step 3: Convert to tensor and run inference ──────
        # frames shape: (30, 112, 112, 3) — same as SLRDataset.__getitem__
        tensor = torch.from_numpy(frames).float()
        tensor = tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        tensor = tensor.unsqueeze(0)           # (1, C, T, H, W)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)           # (1, num_classes)
            probs = torch.softmax(logits, dim=1)  # (1, num_classes)

        probs_np = probs.cpu().numpy()[0]  # (num_classes,)

        # ── Step 4: Build response ───────────────────────────
        top_k = min(5, self.num_classes)
        top_k_indices = np.argsort(probs_np)[::-1][:top_k]

        top_k_predictions = [
            TopKPrediction(
                class_name=self.idx_to_class[str(idx)],
                class_index=int(idx),
                confidence=round(float(probs_np[idx]), 4),
            )
            for idx in top_k_indices
        ]

        top_1_idx = int(top_k_indices[0])
        elapsed_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            predicted_class=self.idx_to_class[str(top_1_idx)],
            class_index=top_1_idx,
            confidence=round(float(probs_np[top_1_idx]), 4),
            top_k_predictions=top_k_predictions,
            hand_detected=hand_detected,
            num_frames_extracted=total_frames,
            processing_time_ms=round(elapsed_ms, 1),
        )
