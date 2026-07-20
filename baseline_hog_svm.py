"""
Baseline Sign Language Recognition: HOG + SVM.

Group 1 (Phase 1) — Mandatory baseline.

Usage
-----
python baseline_hog_svm.py --dataset_root /path/to/dataset --output_dir ./results/baseline

The dataset is expected to follow the folder structure::

    dataset_root/
        class_a/
            video1.mp4
        class_b/
            video1.mp4
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from evaluate import evaluate_model, save_results
from utils_video import (
    extract_frames,
    get_video_statistics,
    preprocess_frame,
    split_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------

def extract_hog_features(
    frames: List[np.ndarray],
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Extract HOG descriptors from a list of frames.

    Args:
        frames:          List of preprocessed frames (float32 [0,1] or uint8).
        orientations:    Number of gradient orientation bins.
        pixels_per_cell: Cell size in pixels.
        cells_per_block: Number of cells per normalisation block.

    Returns:
        HOG feature matrix of shape (n_frames, n_features).
    """
    h, w = frames[0].shape[:2]
    hog = cv2.HOGDescriptor(
        _winSize=(w, h),
        _blockSize=(
            cells_per_block[0] * pixels_per_cell[0],
            cells_per_block[1] * pixels_per_cell[1],
        ),
        _blockStride=(pixels_per_cell[0], pixels_per_cell[1]),
        _cellSize=(pixels_per_cell[0], pixels_per_cell[1]),
        _nbins=orientations,
    )

    features = []
    for frame in tqdm(frames, desc="HOG extraction", leave=False):
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        features.append(hog.compute(frame).flatten())

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Frame loading with video-level split
# ---------------------------------------------------------------------------

def _load_frames_from_items(
    items: List[Tuple[Path, int]],
    max_frames_per_video: int,
    frame_skip: int,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess frames from a list of (video_path, label) items.

    Args:
        items:                List of (video_path, class_idx) tuples.
        max_frames_per_video: Maximum frames per video.
        frame_skip:           Sample every Nth frame.
        target_size:          (width, height) for frame resizing.

    Returns:
        (frames_array, labels_array)
    """
    all_frames: List[np.ndarray] = []
    all_labels: List[int] = []

    for video_path, label in tqdm(items, desc="Loading videos"):
        raw_frames = extract_frames(
            video_path,
            max_frames=max_frames_per_video,
            frame_skip=frame_skip,
            target_size=target_size,
        )
        if not raw_frames:
            logger.warning(f"Skipping (no frames): {video_path}")
            continue

        processed = [
            preprocess_frame(f, target_size=target_size, normalize=True)
            for f in raw_frames
        ]
        all_frames.extend(processed)
        all_labels.extend([label] * len(processed))

    return np.array(all_frames), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_baseline_hog_svm(
    dataset_root: Path,
    output_dir: Path,
    kernel: str = "linear",
    C: float = 1.0,
    gamma: str = "scale",
    max_videos_per_class: Optional[int] = None,
    max_frames_per_video: int = 10,
    frame_skip: int = 5,
    target_size: Tuple[int, int] = (64, 64),
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    hog_orientations: int = 9,
    hog_pixels_per_cell: Tuple[int, int] = (8, 8),
    hog_cells_per_block: Tuple[int, int] = (2, 2),
) -> dict:
    """Train and evaluate a HOG + SVM baseline model.

    The dataset is split at the **video** level before frame extraction
    so that no video contributes frames to more than one partition
    (avoiding data leakage).

    Args:
        dataset_root:         Root directory of the dataset.
        output_dir:           Directory to save model, scaler, and results.
        kernel:               SVM kernel: 'linear' or 'rbf'.
        C:                    SVM regularisation parameter.
        gamma:                SVM gamma (RBF only): 'scale' or 'auto'.
        max_videos_per_class: Cap on videos per class (None = all).
        max_frames_per_video: Maximum frames extracted per video.
        frame_skip:           Sample every Nth frame.
        target_size:          (width, height) for frame resizing.
        val_size:             Fraction of videos for validation.
        test_size:            Fraction of videos for test.
        random_state:         RNG seed for reproducibility.
        hog_orientations:     HOG orientation bins.
        hog_pixels_per_cell:  HOG cell size in pixels.
        hog_cells_per_block:  HOG block size in cells.

    Returns:
        Dictionary with evaluation metrics and saved model paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BASELINE: HOG + SVM")
    logger.info("=" * 60)
    logger.info(f"  dataset:  {dataset_root}")
    logger.info(f"  kernel:   {kernel}  C={C}  gamma={gamma}")
    logger.info(f"  target:   {target_size}  frames/video: {max_frames_per_video}")

    # --- Dataset statistics ---
    stats = get_video_statistics(dataset_root)
    logger.info(f"\nDataset: {stats['num_classes']} classes, {stats['total_videos']} videos")

    # --- Video-level split (prevents frame leakage) ---
    logger.info("\nSplitting dataset at video level...")
    train_items, val_items, test_items, idx_to_class = split_dataset(
        dataset_root,
        max_videos_per_class=max_videos_per_class,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    # --- Load frames per split ---
    logger.info("\nLoading train frames...")
    X_train_raw, y_train = _load_frames_from_items(
        train_items, max_frames_per_video, frame_skip, target_size
    )
    logger.info(f"  Train: {len(X_train_raw)} frames")

    logger.info("Loading test frames...")
    X_test_raw, y_test = _load_frames_from_items(
        test_items, max_frames_per_video, frame_skip, target_size
    )
    logger.info(f"  Test: {len(X_test_raw)} frames")

    if len(X_train_raw) == 0:
        logger.error("No training frames loaded. Check dataset path.")
        return {}

    # --- HOG feature extraction ---
    logger.info("\nExtracting HOG features...")
    X_train_hog = extract_hog_features(
        list(X_train_raw), hog_orientations, hog_pixels_per_cell, hog_cells_per_block
    )
    X_test_hog = extract_hog_features(
        list(X_test_raw), hog_orientations, hog_pixels_per_cell, hog_cells_per_block
    )
    logger.info(f"  HOG feature dim: {X_train_hog.shape[1]}")

    # --- Feature normalisation ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.transform(X_test_hog)

    # --- Train SVM ---
    logger.info(f"\nTraining SVM ({kernel} kernel)...")
    svm_params: dict = {"C": C, "random_state": random_state, "probability": True}
    if kernel == "rbf":
        svm_params["gamma"] = gamma
    svm = SVC(kernel=kernel, **svm_params)
    svm.fit(X_train_scaled, y_train)
    logger.info("  Training complete.")

    # --- Evaluate ---
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    results = evaluate_model(
        model=svm,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name=f"HOG_SVM_{kernel}",
        class_names=class_names,
        save_dir=output_dir,
    )

    # --- Save model + scaler ---
    model_path = output_dir / f"hog_svm_{kernel}.pkl"
    scaler_path = output_dir / f"hog_svm_{kernel}_scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"\nModel saved:  {model_path}")
    logger.info(f"Scaler saved: {scaler_path}")

    # --- Save results ---
    results["model_path"] = str(model_path)
    results["scaler_path"] = str(scaler_path)
    results["split_info"] = {
        "train_videos": len(train_items),
        "val_videos": len(val_items),
        "test_videos": len(test_items),
        "random_state": random_state,
        "note": "Split performed at video level to prevent frame-leakage.",
    }
    results["hog_params"] = {
        "orientations": hog_orientations,
        "pixels_per_cell": hog_pixels_per_cell,
        "cells_per_block": hog_cells_per_block,
    }
    results["svm_params"] = {"kernel": kernel, "C": C, "gamma": gamma}

    results_path = output_dir / f"hog_svm_{kernel}_results.json"
    save_results(results, results_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HOG + SVM baseline for Sign Language Recognition."
    )
    parser.add_argument(
        "--dataset_root", type=Path, required=True,
        help="Root directory of the dataset (class-per-folder structure).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./results/baseline"),
        help="Directory to save model, scaler, and evaluation results.",
    )
    parser.add_argument(
        "--kernel", choices=["linear", "rbf"], default="linear",
        help="SVM kernel (default: linear).",
    )
    parser.add_argument("--C", type=float, default=1.0, help="SVM C (default: 1.0).")
    parser.add_argument("--gamma", default="scale", help="SVM gamma for RBF (default: scale).")
    parser.add_argument(
        "--max_videos_per_class", type=int, default=None,
        help="Cap on videos per class for quick experiments.",
    )
    parser.add_argument("--max_frames_per_video", type=int, default=10)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--target_size", type=int, nargs=2, default=[64, 64],
                        metavar=("W", "H"))
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--both_kernels", action="store_true",
        help="Train with both linear and RBF kernels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    kernels = ["linear", "rbf"] if args.both_kernels else [args.kernel]

    for k in kernels:
        logger.info(f"\n{'='*60}\nKernel: {k.upper()}\n{'='*60}")
        train_baseline_hog_svm(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir / k,
            kernel=k,
            C=args.C,
            gamma=args.gamma,
            max_videos_per_class=args.max_videos_per_class,
            max_frames_per_video=args.max_frames_per_video,
            frame_skip=args.frame_skip,
            target_size=tuple(args.target_size),
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.seed,
        )
