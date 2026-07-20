"""
Real-time Sign Language Recognition via Webcam.

Run from the project root:
    python scripts/realtime_inference.py \\
        --checkpoint ./checkpoints/best_model.pth \\
        --camera 0 \\
        --threshold 0.6

Controls
--------
  q   — quit
  r   — reset frame buffer (useful if model gets stuck)
  s   — save current frame as PNG to results/inference_snapshots/

Error handling
--------------
  - Camera not available → graceful error message, no crash
  - No hand detected in a frame → full frame fallback (same as training)
  - MediaPipe initialisation failure → falls back to no-hand-detection mode
  - Model inference error → logged, prediction held from last good frame
  - Buffer underflow (< 30 frames) → shows "Warming up…" instead of running inference
"""

import argparse
import collections
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Bootstrap sys.path so `src` imports work from any directory ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_3dcnn import Improved3DCNN
from src.preprocessing import ImprovedVideoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── UI constants ──────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
COLOR_PRED = (255, 255, 255)   # white text for prediction
COLOR_BBOX = (0, 255, 0)       # green hand bounding box
COLOR_BAR  = (0, 220, 255)     # yellow-cyan for buffer progress bar
COLOR_WARN = (0, 100, 255)     # orange for warnings


def _draw_hud(frame: np.ndarray, prediction: str, confidence: float,
              buffer_fill: float, hand_detected: bool, fps: float) -> None:
    """Draw heads-up display overlays on the frame in-place."""
    h, w = frame.shape[:2]

    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Prediction + confidence
    cv2.putText(frame, f"Sign: {prediction}", (12, 40), FONT, 1.0, COLOR_PRED, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {confidence:.0%}", (12, 75), FONT, 0.7, COLOR_PRED, 1, cv2.LINE_AA)

    # Buffer fill bar (bottom strip)
    bar_w = int(w * buffer_fill)
    cv2.rectangle(frame, (0, h - 10), (bar_w, h), COLOR_BAR, -1)

    # Hand detection indicator dot (top-right)
    dot_color = (0, 255, 0) if hand_detected else (0, 0, 200)
    cv2.circle(frame, (w - 20, 20), 10, dot_color, -1)
    cv2.putText(frame, "hand" if hand_detected else "no hand",
                (w - 75, 50), FONT, 0.4, dot_color, 1, cv2.LINE_AA)

    # FPS (bottom-right)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 75, h - 18), FONT, 0.5, (200, 200, 200), 1)

    # Controls hint (bottom-left)
    cv2.putText(frame, "q=quit  r=reset  s=snapshot", (8, h - 18),
                FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


def main(args: argparse.Namespace) -> None:

    # ── Load checkpoint ───────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train a model first with:  python scripts/train.py --help"
        )
        sys.exit(1)

    logger.info(f"Loading model from {checkpoint_path} …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        logger.error(f"Failed to load checkpoint: {exc}")
        sys.exit(1)

    idx_to_class = checkpoint.get("class_mapping")
    if not idx_to_class:
        logger.error("Checkpoint has no 'class_mapping'. Re-train with scripts/train.py.")
        sys.exit(1)

    idx_to_class  = {str(k): v for k, v in idx_to_class.items()}
    num_classes   = len(idx_to_class)

    model = Improved3DCNN(num_classes=num_classes)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        logger.error(f"Model architecture mismatch: {exc}")
        sys.exit(1)

    model = model.to(device)
    model.eval()
    logger.info(f"Model ready — {num_classes} classes on {device}")

    # ── Preprocessing (identical to training) ─────────────────
    # IMPORTANT: ImprovedVideoProcessor is the single source of truth for
    # all preprocessing — the same object instance config is used here and
    # in training. If you ever change target_size or num_frames, change it
    # in ONE place (e.g., config) and both training and inference stay in sync.
    use_hand = not args.no_hand_detection
    try:
        processor = ImprovedVideoProcessor(
            target_size=(112, 112),
            num_frames=30,
            use_hand_detection=use_hand,
            confidence=0.3,
        )
        logger.info(f"Hand detection: {'enabled' if use_hand else 'disabled'}")
    except ImportError as exc:
        logger.warning(f"MediaPipe unavailable ({exc}) — falling back to full-frame mode.")
        processor = ImprovedVideoProcessor(
            target_size=(112, 112), num_frames=30, use_hand_detection=False
        )
        use_hand = False

    # ── Open camera ───────────────────────────────────────────
    logger.info(f"Opening camera {args.camera} …")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(
            f"Cannot open camera index {args.camera}.\n"
            "Try a different --camera index (e.g., 0, 1, 2).\n"
            "On Linux: check permissions with  ls -la /dev/video*"
        )
        sys.exit(1)

    # ── Snapshot directory ─────────────────────────────────────
    snapshot_dir = PROJECT_ROOT / "results" / "inference_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ── State ─────────────────────────────────────────────────
    NUM_FRAMES = 30
    frame_buffer    = collections.deque(maxlen=NUM_FRAMES)
    current_pred    = "Warming up…"
    current_conf    = 0.0
    hand_detected   = False
    last_infer_time = time.time()
    fps_timer       = time.time()
    fps             = 0.0
    frame_count     = 0

    logger.info("Feed started — press 'q' to quit, 'r' to reset buffer, 's' to snapshot.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed — camera may have disconnected.")
                break

            frame = cv2.flip(frame, 1)  # mirror so it feels natural to the user

            # ── Preprocess this frame ─────────────────────────
            # _process_frame() runs MediaPipe and CLAHE internally.
            # We cache the bbox separately for the HUD overlay to avoid
            # calling MediaPipe a second time on the same frame.
            hand_detected = False
            try:
                processed = processor._process_frame(frame)
                frame_buffer.append(processed)

                # Peek at hand detection result for the HUD dot
                if processor.detector:
                    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bbox = processor.detector.find_hand_bbox(rgb)
                    if bbox:
                        hand_detected = True
                        x0, y0, x1, y1 = bbox
                        cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_BBOX, 2)
                        cv2.putText(frame, "hand", (x0, y0 - 8),
                                    FONT, 0.5, COLOR_BBOX, 1, cv2.LINE_AA)
            except Exception as exc:
                logger.debug(f"Preprocessing skipped for frame: {exc}")
                # Don't crash — just keep the buffer as-is and show the raw frame

            # ── Inference ─────────────────────────────────────
            now = time.time()
            buffer_ready = len(frame_buffer) == NUM_FRAMES
            interval_ok  = (now - last_infer_time) >= args.inference_interval

            if buffer_ready and interval_ok:
                try:
                    frames_np = np.array(frame_buffer)                    # (30, 112, 112, 3)
                    tensor    = torch.from_numpy(frames_np).float()
                    tensor    = tensor.permute(3, 0, 1, 2).unsqueeze(0)   # (1, 3, 30, 112, 112)
                    tensor    = tensor.to(device)

                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = torch.softmax(logits, dim=1)
                        conf, pred_idx = torch.max(probs, dim=1)
                        conf      = conf.item()
                        pred_idx  = pred_idx.item()

                    if conf >= args.threshold:
                        current_pred = idx_to_class[str(pred_idx)]
                        current_conf = conf
                    else:
                        current_pred = f"Low confidence ({conf:.0%})"
                        current_conf = conf

                except Exception as exc:
                    logger.warning(f"Inference error (keeping last prediction): {exc}")

                last_infer_time = now

            elif not buffer_ready:
                current_pred = f"Warming up ({len(frame_buffer)}/{NUM_FRAMES} frames)…"
                current_conf = 0.0

            # ── HUD overlay ───────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            _draw_hud(
                frame,
                prediction  = current_pred,
                confidence  = current_conf,
                buffer_fill = len(frame_buffer) / NUM_FRAMES,
                hand_detected = hand_detected,
                fps         = fps,
            )

            cv2.imshow("SLR — Sign Language Recognition", frame)

            # ── Key handling ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested.")
                break
            elif key == ord("r"):
                frame_buffer.clear()
                current_pred = "Buffer reset."
                current_conf = 0.0
                logger.info("Frame buffer cleared.")
            elif key == ord("s"):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = snapshot_dir / f"snapshot_{ts}.png"
                cv2.imwrite(str(path), frame)
                logger.info(f"Snapshot saved → {path}")

    finally:
        # Always release resources, even on exception
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Bye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Sign Language Recognition via webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",
                        required=True,
                        help="Path to best_model.pth checkpoint.")
    parser.add_argument("--camera",
                        type=int, default=0,
                        help="Camera device index (0 = default webcam).")
    parser.add_argument("--threshold",
                        type=float, default=0.5,
                        help="Minimum softmax confidence to display a prediction.")
    parser.add_argument("--inference_interval",
                        type=float, default=0.5,
                        help="Minimum seconds between inference calls.")
    parser.add_argument("--no_hand_detection",
                        action="store_true",
                        help="Disable MediaPipe hand detection (use full frame).")

    args = parser.parse_args()
    main(args)
