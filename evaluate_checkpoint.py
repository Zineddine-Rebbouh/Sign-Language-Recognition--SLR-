import argparse
import json
import logging
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SLRDataset
from evaluate import evaluate_model
from model_3dcnn import Improved3DCNN
from preprocessing import ImprovedVideoProcessor
from utils_video import split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
        
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    idx_to_class = checkpoint.get("class_mapping")
    if not idx_to_class:
        logger.error("No class mapping found in checkpoint.")
        return
        
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Improved3DCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info("Initializing dataset...")
    _, _, test_items, _ = split_dataset(
        args.dataset_root,
        max_videos_per_class=args.max_videos_per_class,
        val_size=0.15,
        test_size=0.15,
        random_state=args.seed
    )
    
    processor = ImprovedVideoProcessor(
        target_size=(112, 112),
        num_frames=30,
        use_hand_detection=not args.no_hand_detection
    )
    
    test_dataset = SLRDataset(test_items, processor=processor, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info("Running evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    
    # We create a dummy class with predict/predict_proba to use the evaluate_model function
    class DummyModel:
        def predict(self, X): return y_pred
        def predict_proba(self, X): return y_proba
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluate_model(
        model=DummyModel(),
        X_test=np.zeros((len(y_true), 1)), # Dummy X
        y_test=y_true,
        model_name="Improved3DCNN",
        class_names=class_names,
        save_dir=output_dir
    )
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained 3D CNN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./results/eval", help="Where to save eval metrics/plots")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_videos_per_class", type=int, default=None)
    parser.add_argument("--no_hand_detection", action="store_true")
    
    args = parser.parse_args()
    main(args)
