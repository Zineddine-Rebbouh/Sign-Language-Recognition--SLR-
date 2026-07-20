import pytest
import numpy as np
import sys
import tempfile
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import ImprovedVideoProcessor

def create_dummy_video(path: str, num_frames: int = 10, size: tuple = (200, 200)):
    """Creates a short dummy video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, size)
    for i in range(num_frames):
        # Create a frame with a moving white square
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        x = (i * 10) % size[0]
        cv2.rectangle(frame, (x, x), (x+50, x+50), (255, 255, 255), -1)
        out.write(frame)
    out.release()

def test_processor_initialization():
    processor = ImprovedVideoProcessor(target_size=(112, 112), num_frames=30, use_hand_detection=False)
    assert processor.target_size == (112, 112)
    assert processor.num_frames == 30
    assert not processor.use_hand_detection
    
def test_process_video_shape():
    """Test that the processor outputs the exact required shape regardless of input video length."""
    processor = ImprovedVideoProcessor(target_size=(112, 112), num_frames=30, use_hand_detection=False)
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Create a short video (10 frames)
        create_dummy_video(temp_path, num_frames=10)
        
        frames = processor.process_video(temp_path)
        
        # Output should be padded/sampled to exactly 30 frames
        assert isinstance(frames, np.ndarray)
        assert frames.dtype == np.float32
        assert frames.shape == (30, 112, 112, 3)
        assert np.max(frames) <= 1.0
        assert np.min(frames) >= 0.0
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
