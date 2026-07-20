"""
FastAPI application entry point for Sign Language Recognition.

This defines the HTTP endpoints, handles file uploads, and manages
the model lifespan (loading it once at startup).
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.inference import ModelService
from api.schemas import ErrorResponse, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global reference to the model service
model_service: ModelService | None = None

# Default checkpoint path (can be overridden by env var for Docker)
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "./checkpoints/best_model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for the FastAPI app.
    
    Loads the model once when the server starts and cleans up when it shuts down.
    This prevents the overhead of loading the model on every request.
    """
    global model_service
    
    logger.info("Starting up FastAPI service...")
    
    try:
        if Path(CHECKPOINT_PATH).exists():
            model_service = ModelService(checkpoint_path=CHECKPOINT_PATH)
            logger.info("Model service initialized successfully.")
        else:
            logger.warning(
                f"Checkpoint not found at {CHECKPOINT_PATH}. "
                "The API will start, but /predict will return 503 until a model is provided."
            )
            model_service = None
    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}")
        model_service = None
        
    yield
    
    logger.info("Shutting down FastAPI service...")
    # Add any cleanup code here if needed
    model_service = None


# Create the FastAPI app
app = FastAPI(
    title="Sign Language Recognition API",
    description="Real-time 3D-CNN inference for ASL gesture recognition.",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    }
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check API health",
)
async def health_check():
    """Check if the API is running and the model is loaded."""
    if model_service is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            num_classes=model_service.num_classes,
            device=str(model_service.device),
            model_name="Improved3DCNN"
        )
    else:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            num_classes=0,
            device="unknown",
            model_name="unknown"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict sign from video",
    description="Upload a short video file to get a sign language prediction.",
)
async def predict_video(file: UploadFile = File(...)):
    """
    Run 3D-CNN inference on an uploaded video file.
    
    The video is temporarily saved to disk, processed (frames extracted,
    hands detected, enhanced, resized), and fed through the model.
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable."
        )

    # Read file content to memory to check size and write to temp file
    content = await file.read()
    file_size = len(content)
    
    # Input validation
    error_msg = model_service.validate_video_file(file.filename, file_size)
    if error_msg:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_msg
        )

    # Save to temporary file for OpenCV to read
    # Use a secure temp file that gets deleted automatically
    try:
        # Create temp file with the same extension so OpenCV knows how to decode it
        ext = Path(file.filename).suffix
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            
        # Run inference
        try:
            result = model_service.predict(Path(temp_path))
            return result
        except ValueError as ve:
            # Catch known validation errors (e.g., video too short, unreadable)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve)
            )
        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(e)}"
            )
            
    finally:
        # Always clean up the temp file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")

# ── Mount Gradio Frontend ───────────────────────────────────────────────
# We mount the Gradio app at the root (/) and move the API to /api/predict
# (Wait, we can just mount Gradio at / and keep the API paths as they are, 
#  but it's cleaner to have Gradio at /)
try:
    import gradio as gr
    from app import demo
    
    # Mount Gradio onto the FastAPI app
    app = gr.mount_gradio_app(app, demo, path="/")
    logger.info("Gradio frontend mounted at /")
except ImportError:
    logger.warning("Gradio not installed; running in API-only mode.")
except Exception as e:
    logger.error(f"Failed to mount Gradio: {e}")

