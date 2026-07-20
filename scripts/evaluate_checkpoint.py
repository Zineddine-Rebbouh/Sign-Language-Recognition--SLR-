"""
Evaluate a trained 3D-CNN checkpoint on the held-out test set.

Run from the project root:
    python scripts/evaluate_checkpoint.py \\
        --checkpoint ./checkpoints/best_model.pth \\
        --dataset_root ./data/raw/Full\\ Data \\
        --output_dir ./results/metrics

Outputs saved to --output_dir:
    Improved3DCNN_results.json               — accuracy, F1, mAP, per-class
    Improved3DCNN_confusion_matrix.png       — normalised heatmap
    Improved3DCNN_classification_report.json — sklearn classification_report

Why a separate eval script?
    Training and evaluation are intentionally decoupled so you can re-run
    evaluation with a different threshold / class-set without re-training.
    It also proves the checkpoint is self-contained and portable.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Bootstrap sys.path so `src` imports work from any directory ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SLRDataset
from src.evaluate import evaluate_model
from src.model_3dcnn import Improved3DCNN
from src.preprocessing import ImprovedVideoProcessor
from src.utils_video import split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def gather_predictions(model, dataloader, device):
    """Collect y_true, y_pred, y_proba over the entire dataloader.

    Returning raw arrays lets us call any metric function downstream without
    being tied to sklearn's estimator interface.
    """
    model.eval()
    y_true_list, y_pred_list, y_proba_list = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            logits = model(inputs)                          # (B, C)
            probs  = torch.softmax(logits, dim=1)           # (B, C)
            preds  = torch.argmax(probs, dim=1)             # (B,)

            y_true_list.append(labels.numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_proba_list.append(probs.cpu().numpy())

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_proba_list),
    )


def main(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    idx_to_class = checkpoint.get("class_mapping")
    if not idx_to_class:
        logger.error(
            "Checkpoint has no 'class_mapping' key. "
            "Re-train with the current scripts/train.py."
        )
        sys.exit(1)

    # idx_to_class keys may be int OR str depending on how they were saved
    # Normalise to str keys so look-ups are consistent.
    idx_to_class = {str(k): v for k, v in idx_to_class.items()}
    num_classes   = len(idx_to_class)
    class_names   = [idx_to_class[str(i)] for i in range(num_classes)]

    model = Improved3DCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logger.info(
        f"Model loaded — {model.get_parameter_count() / 1e6:.2f} M params, "
        f"{num_classes} classes, epoch {checkpoint.get('epoch', '?')}"
    )

    # ── Reproduce the exact same test split ───────────────────
    # We use the same random_seed that was stored in the checkpoint args
    # (if available) so the test split is identical to what was held out
    # during training. This is what makes the reported number reproducible.
    saved_args = checkpoint.get("args", {})
    seed = saved_args.get("seed", args.seed)
    if seed != args.seed:
        logger.warning(
            f"Checkpoint was trained with seed={seed}; "
            f"you passed --seed {args.seed}. Using checkpoint seed={seed}."
        )

    logger.info(f"Splitting dataset (seed={seed}) to recover test set …")
    _, _, test_items, _ = split_dataset(
        dataset_root,
        max_videos_per_class=args.max_videos_per_class,
        val_size=0.15,
        test_size=0.15,
        random_state=seed,
    )
    logger.info(f"Test set: {len(test_items)} videos")

    # ── Same preprocessor as training (shared, not duplicated) ─
    no_hand = saved_args.get("no_hand_detection", False)
    processor = ImprovedVideoProcessor(
        target_size=(112, 112),
        num_frames=30,
        use_hand_detection=not no_hand,
    )

    test_dataset = SLRDataset(test_items, processor=processor, augment=False)
    test_loader  = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ── Gather predictions ────────────────────────────────────
    y_true, y_pred, y_proba = gather_predictions(model, test_loader, device)

    # ── Full evaluation — saves JSON + confusion matrix PNG ───
    # evaluate_model() (in src/evaluate.py) writes:
    #   - {model_name}_results.json              (all metrics)
    #   - {model_name}_confusion_matrix.png      (normalised heatmap)
    #   - {model_name}_classification_report.json (per-class P/R/F1)
    #
    # We pass a tiny adapter object so evaluate_model's sklearn-style
    # interface (predict / predict_proba) can consume our pre-computed arrays.
    # This avoids running inference twice.
    class _PrecomputedAdapter:
        """Wraps pre-computed numpy arrays as a sklearn-compatible estimator."""
        def predict(self, X):
            return y_pred
        def predict_proba(self, X):
            return y_proba

    results = evaluate_model(
        model=_PrecomputedAdapter(),
        X_test=np.zeros(len(y_true)),   # unused — predictions already computed
        y_test=y_true,
        model_name="Improved3DCNN",
        class_names=class_names,
        save_dir=output_dir,
    )

    # ── Append reproducibility metadata to the JSON ───────────
    results["reproducibility"] = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "seed": seed,
        "split_level": "video",
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "test_videos": len(test_items),
    }
    from src.evaluate import save_results
    save_results(results, output_dir / "Improved3DCNN_results.json")

    logger.info("\n" + "=" * 60)
    logger.info(f"  Accuracy : {results['accuracy'] * 100:.2f}%")
    logger.info(f"  F1 (wtd) : {results['f1_score']:.4f}")
    logger.info(f"  mAP      : {results['map']:.4f}")
    logger.info("=" * 60)
    logger.info(f"Artefacts saved → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 3D-CNN checkpoint on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",    required=True,                  help="Path to best_model.pth")
    parser.add_argument("--dataset_root",  required=True,                  help="Dataset root directory")
    parser.add_argument("--output_dir",    default="./results/metrics",    help="Where to save artifacts")
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--seed",          type=int,   default=42,         help="Fallback seed (checkpoint seed takes priority)")
    parser.add_argument("--max_videos_per_class", type=int, default=None)

    args = parser.parse_args()
    main(args)
