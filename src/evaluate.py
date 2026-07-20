"""
Comprehensive evaluation module for Sign Language Recognition.
Implements all metrics from Evaluation_Criteria_with_Calculations_LipReading_SignLanguage.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, top_k_accuracy_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities (n_samples, n_classes)
        k: Top k predictions to consider
    
    Returns:
        Top-k accuracy score
    """
    if y_proba.shape[1] < k:
        k = y_proba.shape[1]
    
    try:
        return top_k_accuracy_score(y_true, y_proba, k=k)
    except Exception as e:
        logger.warning(f"Error calculating top-{k} accuracy: {e}")
        # Manual calculation
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        return correct.mean()


def calculate_sequence_accuracy(
    y_true_sequences: List[List[int]],
    y_pred_sequences: List[List[int]]
) -> float:
    """
    Calculate sequence accuracy (exact match).
    
    Args:
        y_true_sequences: List of true label sequences
        y_pred_sequences: List of predicted label sequences
    
    Returns:
        Sequence accuracy (fraction of exactly matching sequences)
    """
    if len(y_true_sequences) != len(y_pred_sequences):
        logger.error("Sequence lengths don't match!")
        return 0.0
    
    correct = 0
    for true_seq, pred_seq in zip(y_true_sequences, y_pred_sequences):
        if len(true_seq) == len(pred_seq) and all(t == p for t, p in zip(true_seq, pred_seq)):
            correct += 1
    
    return correct / len(y_true_sequences) if len(y_true_sequences) > 0 else 0.0


def calculate_frame_wise_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate frame-wise accuracy (same as standard accuracy for frame-level predictions).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Frame-wise accuracy
    """
    return accuracy_score(y_true, y_pred)


def calculate_temporal_detection_accuracy(
    y_true_sequences: List[List[int]],
    y_pred_sequences: List[List[int]],
    tolerance: int = 2
) -> float:
    """
    Calculate temporal detection accuracy with tolerance.
    
    Args:
        y_true_sequences: List of true label sequences
        y_pred_sequences: List of predicted label sequences
        tolerance: Frame tolerance for detection
    
    Returns:
        Temporal detection accuracy
    """
    if len(y_true_sequences) != len(y_pred_sequences):
        return 0.0
    
    correct_detections = 0
    total_detections = 0
    
    for true_seq, pred_seq in zip(y_true_sequences, y_pred_sequences):
        # Find unique labels in true sequence
        true_labels = set(true_seq)
        
        for label in true_labels:
            total_detections += 1
            # Find positions of label in true sequence
            true_positions = [i for i, t in enumerate(true_seq) if t == label]
            
            # Check if predicted sequence has this label within tolerance
            found = False
            for pos in true_positions:
                start = max(0, pos - tolerance)
                end = min(len(pred_seq), pos + tolerance + 1)
                if label in pred_seq[start:end]:
                    found = True
                    break
            
            if found:
                correct_detections += 1
    
    return correct_detections / total_detections if total_detections > 0 else 0.0


def calculate_map(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None
) -> float:
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        class_names: Optional class names
    
    Returns:
        mAP score
    """
    from sklearn.metrics import average_precision_score
    
    n_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Convert to one-hot encoding
    y_true_onehot = np.zeros_like(y_proba)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    # Calculate AP for each class
    aps = []
    for i in range(n_classes):
        try:
            ap = average_precision_score(y_true_onehot[:, i], y_proba[:, i])
            aps.append(ap)
        except:
            pass
    
    return np.mean(aps) if len(aps) > 0 else 0.0


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Path to save the figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        n_classes = len(np.unique(y_true))
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    
    # Plot normalized confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[object] = None,
    model_name: str = "model",
    class_names: Optional[List[str]] = None,
    top_k_values: List[int] = [1, 3, 5],
    save_dir: Optional[Path] = None,
    y_test_sequences: Optional[List[List[int]]] = None,
    y_pred_sequences: Optional[List[List[int]]] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model (must have predict and predict_proba methods)
        X_test: Test features
        y_test: Test labels
        scaler: Optional feature scaler
        model_name: Name of the model
        class_names: Optional class names
        top_k_values: List of k values for top-k accuracy
        save_dir: Directory to save results
        y_test_sequences: Optional true label sequences (for sequence metrics)
        y_pred_sequences: Optional predicted label sequences (for sequence metrics)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"{'='*60}")
    
    # Prepare test data
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Probabilities (if available)
    try:
        y_proba = model.predict_proba(X_test_scaled)
    except:
        logger.warning("Model does not support predict_proba, using one-hot encoding")
        n_classes = len(np.unique(y_test))
        y_proba = np.zeros((len(y_pred), n_classes))
        y_proba[np.arange(len(y_pred)), y_pred] = 1.0
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Top-k accuracy
    top_k_scores = {}
    for k in top_k_values:
        if k <= y_proba.shape[1]:
            top_k_acc = calculate_top_k_accuracy(y_test, y_proba, k=k)
            top_k_scores[f"top_{k}_accuracy"] = top_k_acc
            logger.info(f"Top-{k} Accuracy: {top_k_acc:.4f}")
    
    # mAP
    map_score = calculate_map(y_test, y_proba, class_names)
    
    # Sequence metrics (if provided)
    sequence_accuracy = None
    temporal_detection_accuracy = None
    if y_test_sequences is not None and y_pred_sequences is not None:
        sequence_accuracy = calculate_sequence_accuracy(y_test_sequences, y_pred_sequences)
        temporal_detection_accuracy = calculate_temporal_detection_accuracy(
            y_test_sequences, y_pred_sequences
        )
        logger.info(f"Sequence Accuracy: {sequence_accuracy:.4f}")
        logger.info(f"Temporal Detection Accuracy: {temporal_detection_accuracy:.4f}")
    
    # Frame-wise accuracy (same as accuracy for frame-level)
    frame_wise_accuracy = calculate_frame_wise_accuracy(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Compile results
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'frame_wise_accuracy': float(frame_wise_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'map': float(map_score),
        'top_k_accuracy': top_k_scores,
        'precision_per_class': precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else precision_per_class,
        'recall_per_class': recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else recall_per_class,
        'f1_per_class': f1_per_class.tolist() if isinstance(f1_per_class, np.ndarray) else f1_per_class,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_test),
        'n_classes': len(np.unique(y_test))
    }
    
    if sequence_accuracy is not None:
        results['sequence_accuracy'] = float(sequence_accuracy)
    if temporal_detection_accuracy is not None:
        results['temporal_detection_accuracy'] = float(temporal_detection_accuracy)
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"mAP: {map_score:.4f}")
    logger.info(f"Frame-wise Accuracy: {frame_wise_accuracy:.4f}")
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        results_path = save_dir / f"{model_name}_results.json"
        save_results(results, results_path)
        
        # Save confusion matrix plot
        cm_path = save_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
        
        # Save classification report
        if class_names:
            report = classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True
            )
            report_path = save_dir / f"{model_name}_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
    
    return results


def save_results(results: Dict, file_path: Path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        file_path: Path to save JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {file_path}")


def compare_models(results_list: List[Dict], save_path: Optional[Path] = None) -> Dict:
    """
    Compare multiple models and generate summary.
    
    Args:
        results_list: List of result dictionaries
        save_path: Optional path to save comparison CSV
    
    Returns:
        Comparison dictionary
    """
    comparison = {
        'model_names': [],
        'accuracies': [],
        'f1_scores': [],
        'top_5_accuracies': [],
        'map_scores': []
    }
    
    for results in results_list:
        comparison['model_names'].append(results.get('model_name', 'Unknown'))
        comparison['accuracies'].append(results.get('accuracy', 0.0))
        comparison['f1_scores'].append(results.get('f1_score', 0.0))
        comparison['top_5_accuracies'].append(
            results.get('top_k_accuracy', {}).get('top_5_accuracy', 0.0)
        )
        comparison['map_scores'].append(results.get('map', 0.0))
    
    # Create DataFrame for easy viewing
    try:
        import pandas as pd
        df = pd.DataFrame(comparison)
        df = df.sort_values('accuracy', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"\n{df.to_string(index=False)}")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"\nComparison saved to: {save_path}")
    except ImportError:
        logger.warning("Pandas not available, skipping DataFrame creation")
    
    return comparison

