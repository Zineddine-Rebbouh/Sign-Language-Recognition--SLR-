"""
Training template for Sign Language Recognition models.
This is a template that can be adapted for different model types (2D CNN, Sequence, etc.).

Usage:
  python train_template.py --config config_template.yaml
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import get_2d_cnn_model  # Or import your sequence/3D models
from src.evaluate import evaluate_model
# from src.dataset import SLRDataset # (Example: implement your own Dataset)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = 100.0 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = 100.0 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def load_datasets(config):
    """
    Placeholder for dataset loading.
    You must implement this to return train, val, and test DataLoaders.
    """
    logger.info("Initializing dataloaders...")
    # Example:
    # train_dataset = SLRDataset(config, split='train')
    # val_dataset = SLRDataset(config, split='val')
    # test_dataset = SLRDataset(config, split='test')
    #
    # train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    # return train_loader, val_loader, test_loader
    
    # Returning empty lists as placeholders so the script runs without NameError
    return [], [], []


def train_model(config_path: Path):
    """Main training function."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup paths (remove kaggle hardcoding in practice)
    output_dir = Path(config["output"]["output_dir"])
    model_dir = Path(config["output"]["model_dir"])
    log_dir = Path(config["output"]["log_dir"])
    experiment_name = config["output"]["experiment_name"]

    for d in [output_dir, model_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data loaders
    train_loader, val_loader, test_loader = load_datasets(config)

    # Model
    num_classes = config["model"]["num_classes"]
    model_name = config["model"]["name"]
    model = get_2d_cnn_model(model_name, num_classes, pretrained=config["model"]["pretrained"])
    model = model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    scaler = torch.cuda.amp.GradScaler() if (config.get("advanced", {}).get("mixed_precision") and device.type == "cuda") else None

    # History
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    num_epochs = config["training"]["num_epochs"]

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler) if train_loader else (0,0)
        val_loss, val_acc = validate(model, val_loader, criterion, device) if val_loader else (0,0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc >= best_val_acc and val_loader:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = model_dir / f"{experiment_name}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                },
                model_path,
            )
            logger.info(f"Saved best model: {model_path}")
        else:
            patience_counter += 1

        if patience_counter >= config["training"].get("early_stopping_patience", 10):
            logger.info("Early stopping triggered.")
            break

    # Save history
    history_path = output_dir / f"{experiment_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Evaluate on test set
    if test_loader:
         logger.info("Evaluating on test set...")
         # Need X_test, y_test for evaluate_model
         # y_true, y_pred = [], []
         # ... gather predictions ...
         # evaluate_model(model, X_test, y_test, model_name=experiment_name, save_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_template.yaml"), help="Path to config file")
    args = parser.parse_args()
    train_model(args.config)
