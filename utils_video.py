"""
Video processing utilities for Sign Language Recognition.
Handles frame extraction, preprocessing, and video loading.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    target_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None = all)
        frame_skip: Extract every Nth frame (1 = all frames)
        target_size: Target (width, height) for resizing (None = original)
    
    Returns:
        List of frames as numpy arrays (grayscale)
    """
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
    
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize if needed
                if target_size:
                    gray = cv2.resize(gray, target_size)
                
                frames.append(gray)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
    
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
    
    finally:
        cap.release()
    
    return frames


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single frame.
    
    Args:
        frame: Input frame (grayscale)
        target_size: Target (width, height)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed frame
    """
    # Resize
    resized = cv2.resize(frame, target_size)
    
    # Normalize
    if normalize:
        resized = resized.astype(np.float32) / 255.0
    
    return resized


def load_video_dataset(
    dataset_root: Path,
    class_names: Optional[List[str]] = None,
    max_videos_per_class: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    frame_skip: int = 1,
    target_size: Tuple[int, int] = (64, 64)
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load videos from WLASL dataset structure.
    
    Expected structure:
        dataset_root/
            class1/
                video1.mp4
                video2.mp4
            class2/
                video1.mp4
                ...
    
    Args:
        dataset_root: Root directory of dataset
        class_names: List of class names to load (None = all)
        max_videos_per_class: Maximum videos per class (None = all)
        max_frames_per_video: Maximum frames per video (None = all)
        frame_skip: Extract every Nth frame
        target_size: Target frame size
    
    Returns:
        Tuple of (frames_list, labels_list, video_paths_list)
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return [], [], []
    
    all_frames = []
    all_labels = []
    all_paths = []
    
    # Get class directories
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    if class_names:
        class_dirs = [d for d in class_dirs if d.name in class_names]
    
    class_dirs = sorted(class_dirs)
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    
    logger.info(f"Found {len(class_dirs)} classes")
    
    # Process each class
    for class_dir in tqdm(class_dirs, desc="Loading classes"):
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        # Get video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(list(class_dir.glob(ext)))
        
        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]
        
        logger.info(f"Processing {len(video_files)} videos for class '{class_name}'")
        
        # Extract frames from each video
        for video_path in tqdm(video_files, desc=f"Class {class_name}", leave=False):
            frames = extract_frames(
                video_path,
                max_frames=max_frames_per_video,
                frame_skip=frame_skip,
                target_size=target_size
            )
            
            if len(frames) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                continue
            
            # Preprocess frames
            processed_frames = [
                preprocess_frame(f, target_size=target_size, normalize=True)
                for f in frames
            ]
            
            all_frames.extend(processed_frames)
            all_labels.extend([class_idx] * len(processed_frames))
            all_paths.extend([str(video_path)] * len(processed_frames))
    
    logger.info(f"Total frames loaded: {len(all_frames)}")
    logger.info(f"Number of classes: {len(class_dirs)}")
    
    return all_frames, all_labels, all_paths


def get_video_statistics(dataset_root: Path) -> dict:
    """
    Get statistics about the video dataset.
    
    Returns:
        Dictionary with statistics
    """
    dataset_root = Path(dataset_root)
    stats = {
        'num_classes': 0,
        'classes': [],
        'videos_per_class': {},
        'total_videos': 0
    }
    
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    stats['num_classes'] = len(class_dirs)
    stats['classes'] = [d.name for d in class_dirs]
    
    for class_dir in class_dirs:
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(list(class_dir.glob(ext)))
        stats['videos_per_class'][class_dir.name] = len(video_files)
        stats['total_videos'] += len(video_files)
    
    return stats

