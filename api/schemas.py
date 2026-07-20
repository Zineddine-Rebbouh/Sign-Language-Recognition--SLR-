"""
Pydantic schemas for the SLR FastAPI service.

These models define the request/response contract for all API endpoints.
FastAPI uses them to:
  1. Validate incoming data automatically (returns 422 on bad input)
  2. Generate OpenAPI/Swagger docs at /docs with typed schemas
  3. Serialise responses to JSON with the exact fields documented here
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str = Field(..., examples=["healthy"], description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    num_classes: int = Field(..., examples=[20], description="Number of sign classes the model recognises")
    device: str = Field(..., examples=["cpu"], description="Compute device (cpu or cuda)")
    model_name: str = Field(..., examples=["Improved3DCNN"], description="Model architecture name")


class TopKPrediction(BaseModel):
    """A single class prediction with its confidence score."""
    class_name: str = Field(..., examples=["Hello"], description="Predicted sign/class name")
    class_index: int = Field(..., examples=[0], description="Class index (0-based)")
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.92], description="Softmax probability")


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    predicted_class: str = Field(..., examples=["Hello"], description="Top-1 predicted sign class")
    class_index: int = Field(..., examples=[0], description="Top-1 class index")
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.92], description="Top-1 confidence (softmax)")
    top_k_predictions: List[TopKPrediction] = Field(
        ..., description="Top-5 predictions ranked by confidence"
    )
    hand_detected: bool = Field(..., description="Whether a hand was detected in any frame")
    num_frames_extracted: int = Field(..., description="Number of frames extracted from the video")
    processing_time_ms: float = Field(..., examples=[3200.5], description="Total inference time in milliseconds")


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str = Field(..., description="Human-readable error message")
