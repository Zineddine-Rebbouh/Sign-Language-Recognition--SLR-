"""
Train the 3D-CNN model for Sign Language Recognition.

Run from the project root:
    python scripts/train.py --dataset_root "./data/raw/Full Data" --output_dir ./checkpoints

Why from project root? So that `from src.xxx import` resolves correctly without
installing the package. All paths in --args are relative to where you run from.

MLflow integration
------------------
Every training run is automatically tracked in MLflow:
  - All CLI hyperparameters are logged as params
  - Per-epoch train/val loss and accuracy are logged as metrics
  - Final per-class precision/recall/F1 are logged (when --run_test is set)
  - Artifacts: confusion matrix, training curves, class mapping, split info
  - The model is versioned via mlflow.pytorch.log_model()

To browse past runs:
    mlflow ui --backend-store-uri ./mlruns
    # then open http://localhost:5000
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from datetime import datetime

# ── Bootstrap: add project root to sys.path so `src` is importable
#    regardless of which directory the user runs this script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow
import mlflow.pytorch

from src.dataset import SLRDataset
from src.model_3dcnn import Improved3DCNN
from src.preprocessing import ImprovedVideoProcessor
from src.utils_video import split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all RNG sources so results are reproducible across runs.

    Interview note: even with a fixed seed, results may differ slightly between
    CPU and GPU, or between different GPU hardware, due to non-deterministic
    CUDA kernels. `torch.backends.cudnn.deterministic = True` forces
    determinism at a ~10% speed cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ─────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """One full pass over the training set.

    Why label_smoothing? It prevents the model from becoming over-confident
    on training labels, which improves generalisation — especially important
    when the dataset is small.

    Why gradient clipping? 3D CNNs with many layers can suffer from exploding
    gradients. Clipping to max_norm=1.0 keeps training stable.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """One full pass over the validation set (no grad, no augmentation)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Val  ", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def plot_training_curves(history: dict, save_path: Path) -> None:
    """Save a training/validation loss+accuracy plot as a PNG artifact.

    Why save this? It gives you evidence of training dynamics (convergence,
    overfitting, etc.) which you can include in your thesis without re-running.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display needed
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
        ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
        ax1.set_title("Loss per Epoch", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
        ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
        ax2.set_title("Accuracy per Epoch (%)", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("3D-CNN Training Curves", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training curves saved → {save_path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping training curve plot.")


def gather_predictions(model, dataloader, device):
    """Collect y_true, y_pred, y_proba over a dataloader.

    Shared helper so both training-time test evaluation and
    evaluate_checkpoint.py use the same logic.
    """
    model.eval()
    y_true_list, y_pred_list, y_proba_list = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Test eval", leave=False):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_list.append(labels.numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_proba_list.append(probs.cpu().numpy())

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_proba_list),
    )


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save a confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={"label": "Normalized Count"},
        )
        plt.title("Confusion Matrix (Normalized)", fontsize=16, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved → {save_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping confusion matrix.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        sys.exit(1)

    # ── MLflow setup ─────────────────────────────────────────
    # File-based store by default — no server process needed.
    # Switch to a tracking server by changing --mlflow_tracking_uri.
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    logger.info(
        f"MLflow experiment: '{args.experiment_name}' "
        f"(tracking URI: {args.mlflow_tracking_uri})"
    )

    # ── Split at video level (no frame leakage) ──────────────
    # Why video-level? If you split after frame extraction, frames from the
    # same video appear in both train AND test — the model sees test data
    # during training, so accuracy is inflated. Video-level split is the
    # only correct approach.
    logger.info(f"Splitting dataset at video level: {dataset_root}")
    train_items, val_items, test_items, idx_to_class = split_dataset(
        dataset_root,
        max_videos_per_class=args.max_videos_per_class,
        val_size=0.15,
        test_size=0.15,
        random_state=args.seed,
    )

    # Persist the class mapping so eval/inference scripts can load it
    # independently of the checkpoint (useful for debugging).
    class_map_path = output_dir / "class_mapping.json"
    with open(class_map_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    # Also persist the split sizes for the record
    split_info = {
        "train_videos": len(train_items),
        "val_videos":   len(val_items),
        "test_videos":  len(test_items),
        "total_classes": len(idx_to_class),
        "random_seed":  args.seed,
        "val_fraction":  0.15,
        "test_fraction": 0.15,
        "split_level":   "video",
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(
        f"Split → {len(train_items)} train / {len(val_items)} val / "
        f"{len(test_items)} test videos across {len(idx_to_class)} classes"
    )

    # ── Shared preprocessing (train = val = inference) ───────
    # Why shared? If training applies CLAHE and inference doesn't, the model
    # sees a different input distribution at test time → accuracy drops.
    # Both use the exact same ImprovedVideoProcessor instance parameters.
    processor = ImprovedVideoProcessor(
        target_size=(112, 112),
        num_frames=30,
        use_hand_detection=not args.no_hand_detection,
    )

    train_dataset = SLRDataset(train_items, processor=processor, augment=True)
    val_dataset   = SLRDataset(val_items,   processor=processor, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = Improved3DCNN(num_classes=len(idx_to_class), dropout_rate=args.dropout)
    model = model.to(device)
    logger.info(f"Parameters: {model.get_parameter_count() / 1e6:.2f} M")

    # ── Loss: label smoothing ─────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimiser: Adam + cosine LR decay ────────────────────
    # Why cosine annealing? It gradually reduces the learning rate following a
    # cosine curve rather than step drops — smoother convergence, less
    # sensitive to the exact decay schedule.
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision speeds up training ~2× on Ampere GPUs with no accuracy loss
    scaler = (
        torch.cuda.amp.GradScaler()
        if args.mixed_precision and device.type == "cuda"
        else None
    )

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Start MLflow run ─────────────────────────────────────
    with mlflow.start_run(run_name=args.run_name) as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log ALL hyperparameters so runs are fully reproducible
        # and comparable side-by-side in the MLflow UI.
        mlflow.log_params({
            # Data
            "dataset_root": str(dataset_root),
            "train_videos": len(train_items),
            "val_videos": len(val_items),
            "test_videos": len(test_items),
            "num_classes": len(idx_to_class),
            "max_videos_per_class": args.max_videos_per_class or "all",
            # Preprocessing
            "target_size": "112x112",
            "num_frames": 30,
            "use_hand_detection": not args.no_hand_detection,
            # Architecture
            "model": "Improved3DCNN",
            "dropout_rate": args.dropout,
            "parameters_M": round(model.get_parameter_count() / 1e6, 2),
            # Training
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": 0.1,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "patience": args.patience,
            "mixed_precision": args.mixed_precision,
            "gradient_clip_max_norm": 1.0,
            # Reproducibility
            "seed": args.seed,
            "device": str(device),
        })

        # Tag the run with useful metadata for filtering in the UI
        mlflow.set_tags({
            "model_family": "3D-CNN",
            "dataset": "ASL-20",
            "stage": "training",
        })

        logger.info("Starting training…")
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
            logger.info(f"  Val    loss={val_loss:.4f}  acc={val_acc:.2f}%")

            # Log per-epoch metrics — MLflow plots these as time series
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                ckpt_path = output_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "class_mapping": idx_to_class,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"  ✓ New best → {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    break

        # ── Log final summary metrics ────────────────────────
        final_epoch = len(history["train_loss"])
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "epochs_completed": final_epoch,
        })

        # ── Persist history + training curves ────────────────
        history_path = results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"History saved → {history_path}")

        curves_path = results_dir / "training_curves.png"
        plot_training_curves(history, curves_path)

        # ── Log artifacts to MLflow ──────────────────────────
        # These show up in the MLflow UI under the "Artifacts" tab
        # so you can inspect them without hunting for local files.
        mlflow.log_artifact(str(class_map_path))
        mlflow.log_artifact(str(output_dir / "split_info.json"))
        mlflow.log_artifact(str(history_path))
        if curves_path.exists():
            mlflow.log_artifact(str(curves_path))

        # ── Optional: test-set evaluation within the same run ─
        if args.run_test:
            logger.info("\n── Running test-set evaluation ──")
            test_dataset = SLRDataset(test_items, processor=processor, augment=False)
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )

            # Reload best checkpoint for evaluation
            best_ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])

            y_true, y_pred, y_proba = gather_predictions(model, test_loader, device)

            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
            )

            test_acc = accuracy_score(y_true, y_pred)
            test_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            test_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            test_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            mlflow.log_metrics({
                "test_accuracy": test_acc * 100,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            })

            logger.info(f"  Test accuracy:  {test_acc * 100:.2f}%")
            logger.info(f"  Test F1 (wtd):  {test_f1:.4f}")

            # Per-class metrics
            class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
            per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

            for i, name in enumerate(class_names):
                if i < len(per_class_f1):
                    mlflow.log_metrics({
                        f"class_{name}_precision": per_class_precision[i],
                        f"class_{name}_recall": per_class_recall[i],
                        f"class_{name}_f1": per_class_f1[i],
                    })

            # Confusion matrix
            cm_path = results_dir / "confusion_matrix.png"
            plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))

            # Save per-class report as JSON artifact
            per_class_report = {
                class_names[i]: {
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "f1": float(per_class_f1[i]),
                }
                for i in range(len(class_names))
                if i < len(per_class_f1)
            }
            report_path = results_dir / "per_class_report.json"
            with open(report_path, "w") as f:
                json.dump(per_class_report, f, indent=2)
            mlflow.log_artifact(str(report_path))

        # ── Log the model to MLflow ──────────────────────────
        # This versions the model properly. The checkpoint is stored
        # inside the MLflow run, so you can load it later with:
        #   mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        logger.info("Logging model to MLflow…")
        ckpt_path = output_dir / "best_model.pth"
        if ckpt_path.exists():
            # Reload best model weights before logging
            best_ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="SLR-3DCNN",
        )
        # Also log the raw checkpoint file as an artifact
        if ckpt_path.exists():
            mlflow.log_artifact(str(ckpt_path))

        logger.info(
            f"\nTraining complete — best validation accuracy: {best_val_acc:.2f}%"
        )
        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info(
            f"View this run: mlflow ui --backend-store-uri {args.mlflow_tracking_uri}"
        )
        logger.info(
            "Next step: run  python scripts/evaluate_checkpoint.py "
            f"--checkpoint {output_dir / 'best_model.pth'}  --dataset_root {dataset_root}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 3D-CNN for Sign Language Recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root",         required=True,  help="Root of dataset (class-per-folder).")
    parser.add_argument("--output_dir",  default="./checkpoints", help="Where to save model weights.")
    parser.add_argument("--results_dir", default="./results",     help="Where to save history + curves.")
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.5)
    parser.add_argument("--patience",    type=int,   default=10,  help="Early-stopping patience (epochs).")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--max_videos_per_class", type=int, default=None, help="Cap for quick experiments.")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--mixed_precision", action="store_true", help="Use FP16 AMP (GPU only).")
    parser.add_argument("--no_hand_detection", action="store_true", help="Disable MediaPipe ROI step.")

    # MLflow arguments
    parser.add_argument("--experiment_name", default="SLR-3DCNN",
                        help="MLflow experiment name.")
    parser.add_argument("--run_name", default=None,
                        help="MLflow run name (auto-generated if not set).")
    parser.add_argument("--mlflow_tracking_uri", default="./mlruns",
                        help="MLflow tracking URI (file path or server URL).")
    parser.add_argument("--run_test", action="store_true",
                        help="Evaluate on test set and log results within the same MLflow run.")

    args = parser.parse_args()
    main(args)
