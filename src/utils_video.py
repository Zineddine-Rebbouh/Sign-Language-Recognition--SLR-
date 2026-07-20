"""
Video processing utilities for Sign Language Recognition.

Provides:
  - extract_frames()       Extract raw frames from a video file.
  - preprocess_frame()     Resize + normalise a single frame.
  - load_video_dataset()   Load an entire folder-structured dataset.
  - split_dataset()        Video-level train/val/test split (no leakage).
  - get_video_statistics() Count videos and classes in a dataset root.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Extract frames from a video file.

    Args:
        video_path:  Path to the video file.
        max_frames:  Stop after this many extracted frames (None = all).
        frame_skip:  Sample every Nth frame (1 = all frames).
        target_size: (width, height) to resize each frame. None = original.

    Returns:
        List of grayscale uint8 frames as numpy arrays.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    frames: List[np.ndarray] = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if target_size:
                    gray = cv2.resize(gray, target_size)
                frames.append(gray)
                if max_frames and len(frames) >= max_frames:
                    break

            frame_count += 1
    except Exception as exc:
        logger.error(f"Error reading {video_path}: {exc}")
    finally:
        cap.release()

    return frames


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    normalize: bool = True,
) -> np.ndarray:
    """Resize and optionally normalise a single grayscale frame.

    Args:
        frame:       Input grayscale frame (H, W) uint8.
        target_size: (width, height) for resizing.
        normalize:   If True, scale pixel values to [0, 1] as float32.

    Returns:
        Preprocessed frame.
    """
    resized = cv2.resize(frame, target_size)
    if normalize:
        return resized.astype(np.float32) / 255.0
    return resized


# ---------------------------------------------------------------------------
# Dataset loading — frame-level (for HOG/SVM)
# ---------------------------------------------------------------------------

def load_video_dataset(
    dataset_root: Path,
    class_names: Optional[List[str]] = None,
    max_videos_per_class: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    frame_skip: int = 1,
    target_size: Tuple[int, int] = (64, 64),
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load frames from a folder-structured dataset.

    Expected layout::

        dataset_root/
            class_a/
                video1.mp4
            class_b/
                video1.mp4
                video2.mp4

    **Important**: for classification models that train on frame-level
    features (e.g. HOG+SVM) you must call `split_dataset()` on the video
    paths *before* extracting frames so that no video appears in both
    train and test splits.  Splitting after frame extraction causes data
    leakage because frames from the same video share appearance.

    Args:
        dataset_root:         Root directory.
        class_names:          If given, only load these classes.
        max_videos_per_class: Cap on videos per class (None = all).
        max_frames_per_video: Cap on frames per video (None = all).
        frame_skip:           Extract every Nth frame.
        target_size:          (width, height) for frame resizing.

    Returns:
        (frames_list, labels_list, video_paths_list) where every frame is
        tagged with its class index and the path of the video it came from.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return [], [], []

    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if class_names:
        class_dirs = [d for d in class_dirs if d.name in class_names]

    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    logger.info(f"Found {len(class_dirs)} classes")

    all_frames: List[np.ndarray] = []
    all_labels: List[int] = []
    all_paths: List[str] = []

    for class_dir in tqdm(class_dirs, desc="Loading classes"):
        class_idx = class_to_idx[class_dir.name]

        video_files: List[Path] = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(class_dir.glob(ext))
        video_files.sort()

        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]

        for video_path in tqdm(video_files, desc=class_dir.name, leave=False):
            raw_frames = extract_frames(
                video_path,
                max_frames=max_frames_per_video,
                frame_skip=frame_skip,
                target_size=target_size,
            )
            if not raw_frames:
                logger.warning(f"No frames from {video_path}")
                continue

            processed = [
                preprocess_frame(f, target_size=target_size, normalize=True)
                for f in raw_frames
            ]
            all_frames.extend(processed)
            all_labels.extend([class_idx] * len(processed))
            all_paths.extend([str(video_path)] * len(processed))

    logger.info(f"Total frames loaded: {len(all_frames)}")
    return all_frames, all_labels, all_paths


# ---------------------------------------------------------------------------
# Video-level split (no leakage)
# ---------------------------------------------------------------------------

def split_dataset(
    dataset_root: Path,
    class_names: Optional[List[str]] = None,
    max_videos_per_class: Optional[int] = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
    Dict[int, str],
]:
    """Split a dataset into train / val / test at the **video** level.

    Splitting at video level prevents frames from the same clip appearing
    in both train and test (data leakage).  Each split is stratified by
    class so class balance is preserved.

    Args:
        dataset_root:         Root directory.
        class_names:          If given, only use these classes.
        max_videos_per_class: Cap on videos per class (None = all).
        val_size:             Fraction of data for validation.
        test_size:            Fraction of data for test.
        random_state:         RNG seed for reproducibility.

    Returns:
        (train_items, val_items, test_items, idx_to_class)
        Each *_items is a list of (video_path, class_idx) tuples.
        idx_to_class maps integer class index → class name string.
    """
    dataset_root = Path(dataset_root)
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if class_names:
        class_dirs = [d for d in class_dirs if d.name in class_names]

    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    all_paths: List[Path] = []
    all_labels: List[int] = []

    for class_dir in class_dirs:
        video_files: List[Path] = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(class_dir.glob(ext))
        video_files.sort()

        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]

        all_paths.extend(video_files)
        all_labels.extend([class_to_idx[class_dir.name]] * len(video_files))

    all_paths_arr = np.array(all_paths)
    all_labels_arr = np.array(all_labels)

    # First split: train vs. temp (val+test)
    temp_fraction = val_size + test_size
    paths_train, paths_temp, labels_train, labels_temp = train_test_split(
        all_paths_arr, all_labels_arr,
        test_size=temp_fraction,
        random_state=random_state,
        stratify=all_labels_arr,
    )

    # Second split: val vs. test from temp
    relative_test = test_size / temp_fraction
    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_temp, labels_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=labels_temp,
    )

    train_items = list(zip(paths_train, labels_train))
    val_items = list(zip(paths_val, labels_val))
    test_items = list(zip(paths_test, labels_test))

    logger.info(
        f"Split: {len(train_items)} train / {len(val_items)} val / "
        f"{len(test_items)} test videos"
    )

    return train_items, val_items, test_items, idx_to_class


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def get_video_statistics(dataset_root: Path) -> dict:
    """Count videos and classes in a dataset root.

    Returns:
        Dictionary with keys: num_classes, classes, videos_per_class,
        total_videos.
    """
    dataset_root = Path(dataset_root)
    stats: dict = {
        "num_classes": 0,
        "classes": [],
        "videos_per_class": {},
        "total_videos": 0,
    }

    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    stats["num_classes"] = len(class_dirs)
    stats["classes"] = sorted(d.name for d in class_dirs)

    for class_dir in class_dirs:
        videos: List[Path] = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            videos.extend(class_dir.glob(ext))
        stats["videos_per_class"][class_dir.name] = len(videos)
        stats["total_videos"] += len(videos)

    return stats
