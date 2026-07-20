import pytest
import torch
import sys
from pathlib import Path

# Add project root to sys.path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_3dcnn import Improved3DCNN

def test_model_initialization():
    """Test that the model initializes correctly with default arguments."""
    model = Improved3DCNN(num_classes=20)
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
def test_model_parameter_count():
    """Test that the parameter count is roughly what we expect (~14.3M)."""
    model = Improved3DCNN(num_classes=20)
    param_count = model.get_parameter_count()
    # It should be around 14.3M. Let's assert it's between 14M and 15M.
    assert 14_000_000 < param_count < 15_000_000

def test_model_forward_pass_shape():
    """Test that the model accepts the correct input shape and produces the correct output shape."""
    model = Improved3DCNN(num_classes=20)
    model.eval()
    
    # Input shape: (Batch, Channels, Time/Frames, Height, Width)
    # Our preprocessor produces (C, T, H, W) = (3, 30, 112, 112)
    # So with batch size 2, it's (2, 3, 30, 112, 112)
    dummy_input = torch.randn(2, 3, 30, 112, 112)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Output shape should be (Batch, NumClasses)
    assert output.shape == (2, 20)
