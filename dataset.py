import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from typing import List, Tuple, Callable

logger = logging.getLogger(__name__)

class SLRDataset(Dataset):
    """Sign Language Recognition Dataset.
    
    Can either process raw video paths on the fly using a processor,
    or load pre-extracted frame numpy arrays.
    """
    
    def __init__(
        self,
        items: List[Tuple[str, int]],
        processor: Callable = None,
        augment: bool = False
    ):
        """
        Args:
            items: List of (video_path, class_idx)
            processor: Function/object with `process_video(path) -> np.ndarray`.
            augment: Whether to apply data augmentation (not fully implemented).
        """
        self.items = items
        self.processor = processor
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.items)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.items[idx]
        
        if self.processor:
            # Process on the fly: returns (num_frames, H, W, C)
            try:
                frames = self.processor.process_video(video_path)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                # Fallback zero tensor
                frames = np.zeros((30, 112, 112, 3), dtype=np.float32)
        else:
            # Assumes video_path is already a processed numpy array
            frames = video_path
            
        if self.augment:
            # Optional: Add frame-level augmentations here (color jitter, etc.)
            pass
            
        # Convert to tensor and permute to (C, T, H, W) for 3D CNN
        # frames is (T, H, W, C)
        tensor = torch.from_numpy(frames).float()
        tensor = tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        return tensor, label
