import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import app after fixing sys.path
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test that the /health endpoint returns a 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_invalid_extension():
    """Test that uploading a file with an invalid extension returns 422."""
    # Create a dummy text file
    files = {"file": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/predict", files=files)
    
    # Depending on how the model_service is loaded (or not loaded in CI without checkpoint),
    # it might return 503 if model isn't loaded. We should account for both.
    assert response.status_code in (422, 503)
    
    if response.status_code == 422:
        assert "Unsupported file type" in response.json()["detail"]
        
def test_predict_empty_file():
    """Test that uploading an empty file returns 422."""
    files = {"file": ("test.mp4", b"", "video/mp4")}
    response = client.post("/predict", files=files)
    
    assert response.status_code in (422, 503)
    if response.status_code == 422:
        assert "Empty file" in response.json()["detail"]
