"""
Training template for Sign Language Recognition models.
This is a template that can be adapted for different model types.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import json
from datetime import datetime

from models import get_2d_cnn_model  # Import appropriate model
from evaluate import evaluate_model_comprehensive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
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
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(config_path: Path):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    output_dir = Path(config['output']['output_dir'])
    model_dir = Path(config['output']['model_dir'])
    log_dir = Path(config['output']['log_dir'])
    experiment_name = config['output']['experiment_name']
    
    # Create directories
    for dir_path in [output_dir, model_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset (implement your dataset loading here)
    # train_loader, val_loader, test_loader = load_datasets(config)
    
    # Create model
    num_classes = config['model']['num_classes']
    model_name = config['model']['name']
    model = get_2d_cnn_model(model_name, num_classes, 
                            pretrained=config['model']['pretrained'])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['learning_rate'],
                          weight_decay=config['training']['weight_decay'])
    
    # Mixed precision
    scaler = None
    if config['advanced']['mixed_precision'] and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = model_dir / f"{experiment_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            logger.info(f"Saved best model: {model_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation
    logger.info("\nEvaluating on test set...")
    # results = evaluate_model_comprehensive(...)
    
    # Save history
    history_path = output_dir / f"{experiment_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    import sys
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    train_model(config_path)

