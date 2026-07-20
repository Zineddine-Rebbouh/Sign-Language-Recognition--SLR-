import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SLRDataset
from model_3dcnn import Improved3DCNN
from preprocessing import ImprovedVideoProcessor
from utils_video import split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

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


def main(args):
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Splitting dataset: {dataset_root}")
    train_items, val_items, test_items, idx_to_class = split_dataset(
        dataset_root,
        max_videos_per_class=args.max_videos_per_class,
        val_size=0.15,
        test_size=0.15,
        random_state=args.seed
    )
    
    # Save class mapping
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)
        
    logger.info(f"Found {len(idx_to_class)} classes.")
    
    processor = ImprovedVideoProcessor(
        target_size=(112, 112),
        num_frames=30,
        use_hand_detection=not args.no_hand_detection
    )
    
    train_dataset = SLRDataset(train_items, processor=processor, augment=True)
    val_dataset = SLRDataset(val_items, processor=processor, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = Improved3DCNN(num_classes=len(idx_to_class), dropout_rate=args.dropout)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_mapping": idx_to_class,
            }, model_path)
            logger.info(f"Saved new best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break
                
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    logger.info(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CNN for Sign Language Recognition")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./results/3dcnn", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_videos_per_class", type=int, default=None, help="Limit videos for quick testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--mixed_precision", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--no_hand_detection", action="store_true", help="Disable MediaPipe hand detection")
    
    args = parser.parse_args()
    main(args)
