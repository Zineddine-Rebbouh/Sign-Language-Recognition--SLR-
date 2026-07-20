"""
Gradio Frontend for Sign Language Recognition.

This script creates a web UI for the model. For deployment (e.g. Hugging Face Spaces),
we mount this Gradio app directly onto our FastAPI application. This gives us the
best of both worlds in a single container:
  1. A sleek web UI for recruiters/users at the root URL (/)
  2. A production REST API at /api/predict for developers
  3. Interactive API docs at /docs
"""

import gradio as gr
import os
from pathlib import Path

# Important: We import the ModelService directly to run inference in-process,
# bypassing HTTP overhead for the web UI, while still using the exact same logic.
from api.inference import ModelService
from api.main import CHECKPOINT_PATH

# Initialize the model service manually for the UI
# (FastAPI lifespan handles its own initialization)
ui_model_service = None

def init_service():
    global ui_model_service
    if ui_model_service is None and Path(CHECKPOINT_PATH).exists():
        ui_model_service = ModelService(checkpoint_path=CHECKPOINT_PATH)
    return ui_model_service


def predict_video(video_filepath):
    """Gradio inference callback."""
    service = init_service()
    if service is None:
        return "Error: Model checkpoint not found.", None, "Please train the model first."

    if not video_filepath:
        return "Error: No video uploaded.", None, ""

    try:
        # Run the exact same inference pipeline as the API
        file_size = os.path.getsize(video_filepath)
        error = service.validate_video_file(video_filepath, file_size)
        if error:
            return f"Error: {error}", None, ""

        result = service.predict(Path(video_filepath))
        
        # Format output for Gradio
        top_k_dict = {
            pred.class_name: pred.confidence 
            for pred in result.top_k_predictions
        }
        
        status = f"✅ Processed {result.num_frames_extracted} frames in {result.processing_time_ms}ms"
        if not result.hand_detected:
            status += "\n⚠️ Warning: No hand detected in video. Prediction may be unreliable."
            
        return result.predicted_class, top_k_dict, status

    except Exception as e:
        return f"Error: {str(e)}", None, "Inference failed."


# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🤟 Sign Language Recognition (3D-CNN)")
    gr.Markdown(
        "Upload a short video (e.g., ASL-20 dataset) to see the model in action. "
        "The model analyzes spatial and temporal features across 30 frames simultaneously."
    )
    
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Upload Sign Video", include_audio=False)
            submit_btn = gr.Button("Analyze Sign", variant="primary")
            
        with gr.Column():
            output_class = gr.Textbox(label="Top Prediction", text_align="center")
            output_probs = gr.Label(label="Confidence Scores", num_top_classes=5)
            output_status = gr.Textbox(label="Processing Details", interactive=False)

    submit_btn.click(
        fn=predict_video,
        inputs=[video_in],
        outputs=[output_class, output_probs, output_status]
    )

    gr.Markdown(
        """
        ---
        **MLOps Architecture**:
        - **Model**: Custom 14.3M parameter 3D-CNN (`Improved3DCNN`)
        - **Backend**: FastAPI + Uvicorn
        - **Container**: Multi-stage Docker
        - **Tracking**: MLflow
        
        *Developers: The REST API is available at `/api/predict` and OpenAPI docs at `/docs`.*
        """
    )

# When running locally as a script
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
