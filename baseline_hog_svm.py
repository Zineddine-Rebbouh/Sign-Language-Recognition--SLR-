"""
Baseline Sign Language Recognition using HOG + SVM.
Phase 1: Mandatory baseline model.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import pickle
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2

from utils_video import load_video_dataset, preprocess_frame
from evaluate import evaluate_model, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_hog_features(
    frames: List[np.ndarray],
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from frames.
    
    Args:
        frames: List of preprocessed frames
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
    
    Returns:
        Array of HOG features (n_samples, n_features)
    """
    hog = cv2.HOGDescriptor(
        _winSize=(frames[0].shape[1], frames[0].shape[0]),
        _blockSize=(cells_per_block[0] * pixels_per_cell[0],
                    cells_per_block[1] * pixels_per_cell[1]),
        _blockStride=(pixels_per_cell[0], pixels_per_cell[1]),
        _cellSize=(pixels_per_cell[0], pixels_per_cell[1]),
        _nbins=orientations
    )
    
    features = []
    for frame in tqdm(frames, desc="Extracting HOG features"):
        # Convert to uint8 if normalized
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
        
        # Compute HOG descriptor
        hog_features = hog.compute(frame)
        features.append(hog_features.flatten())
    
    return np.array(features)


def train_baseline_hog_svm(
    dataset_root: Path,
    output_dir: Path,
    kernel: str = 'linear',
    C: float = 1.0,
    gamma: Optional[str] = 'scale',
    max_videos_per_class: Optional[int] = None,
    max_frames_per_video: Optional[int] = 10,
    frame_skip: int = 5,
    target_size: Tuple[int, int] = (64, 64),
    test_size: float = 0.2,
    random_state: int = 42,
    hog_orientations: int = 9,
    hog_pixels_per_cell: Tuple[int, int] = (8, 8),
    hog_cells_per_block: Tuple[int, int] = (2, 2)
) -> dict:
    """
    Train baseline HOG + SVM model.
    
    Args:
        dataset_root: Root directory of WLASL dataset
        output_dir: Directory to save model and results
        kernel: SVM kernel ('linear' or 'rbf')
        C: SVM regularization parameter
        gamma: SVM gamma parameter (for RBF kernel)
        max_videos_per_class: Maximum videos per class
        max_frames_per_video: Maximum frames per video
        frame_skip: Extract every Nth frame
        target_size: Target frame size
        test_size: Test set fraction
        random_state: Random seed
        hog_orientations: HOG orientations
        hog_pixels_per_cell: HOG pixels per cell
        hog_cells_per_block: HOG cells per block
    
    Returns:
        Dictionary with results and model info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("BASELINE MODEL: HOG + SVM")
    logger.info("=" * 60)
    logger.info(f"Kernel: {kernel}")
    logger.info(f"C: {C}")
    logger.info(f"Gamma: {gamma}")
    
    # Load dataset
    logger.info("\nLoading video dataset...")
    frames, labels, video_paths = load_video_dataset(
        dataset_root=dataset_root,
        max_videos_per_class=max_videos_per_class,
        max_frames_per_video=max_frames_per_video,
        frame_skip=frame_skip,
        target_size=target_size
    )
    
    if len(frames) == 0:
        logger.error("No frames loaded! Check dataset path.")
        return {}
    
    frames_array = np.array(frames)
    labels_array = np.array(labels)
    
    logger.info(f"Loaded {len(frames_array)} frames")
    logger.info(f"Number of classes: {len(np.unique(labels_array))}")
    
    # Extract HOG features
    logger.info("\nExtracting HOG features...")
    hog_features = extract_hog_features(
        frames_array,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block
    )
    
    logger.info(f"HOG feature shape: {hog_features.shape}")
    
    # Split train/test
    logger.info("\nSplitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        hog_features, labels_array,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Normalize features
    logger.info("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    logger.info(f"\nTraining SVM with {kernel} kernel...")
    svm_params = {'C': C, 'random_state': random_state}
    if kernel == 'rbf':
        svm_params['gamma'] = gamma
    
    svm = SVC(kernel=kernel, **svm_params, probability=True)
    svm.fit(X_train_scaled, y_train)
    
    logger.info("Training completed!")
    
    # Evaluate
    logger.info("\nEvaluating model...")
    results = evaluate_model(
        model=svm,
        X_test=X_test_scaled,
        y_test=y_test,
        scaler=scaler,
        model_name=f"HOG_SVM_{kernel}",
        class_names=None  # Will be inferred
    )
    
    # Save model and scaler
    model_path = output_dir / f"hog_svm_{kernel}.pkl"
    scaler_path = output_dir / f"hog_svm_{kernel}_scaler.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Scaler saved to: {scaler_path}")
    
    # Save results
    results_path = output_dir / f"hog_svm_{kernel}_results.json"
    save_results(results, results_path)
    
    results['model_path'] = str(model_path)
    results['scaler_path'] = str(scaler_path)
    results['hog_params'] = {
        'orientations': hog_orientations,
        'pixels_per_cell': hog_pixels_per_cell,
        'cells_per_block': hog_cells_per_block
    }
    results['svm_params'] = {
        'kernel': kernel,
        'C': C,
        'gamma': gamma
    }
    
    return results


if __name__ == "__main__":
    # Example usage for Kaggle
    import sys
    
    # Kaggle paths
    DATASET_ROOT = Path("/kaggle/input/wlasl-processed")
    OUTPUT_DIR = Path("/kaggle/working/baseline_hog_svm")
    
    # For testing, you can override paths
    if len(sys.argv) > 1:
        DATASET_ROOT = Path(sys.argv[1])
    if len(sys.argv) > 2:
        OUTPUT_DIR = Path(sys.argv[2])
    
    # Train with linear kernel
    logger.info("Training with LINEAR kernel...")
    results_linear = train_baseline_hog_svm(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR / "linear",
        kernel='linear',
        C=1.0,
        max_videos_per_class=50,  # Limit for faster training
        max_frames_per_video=10,
        frame_skip=5
    )
    
    # Train with RBF kernel
    logger.info("\n\nTraining with RBF kernel...")
    results_rbf = train_baseline_hog_svm(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR / "rbf",
        kernel='rbf',
        C=1.0,
        gamma='scale',
        max_videos_per_class=50,
        max_frames_per_video=10,
        frame_skip=5
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)

