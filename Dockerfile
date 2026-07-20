# Stage 1: Builder
# We use a builder stage to compile any C-extensions and install packages
# into a virtual environment, keeping the final runtime image clean.
FROM python:3.11-slim as builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------

# Stage 2: Runtime
# This is the final slim image that will be deployed.
FROM python:3.11-slim as runtime

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create a non-root user with UID 1000 (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Copy application code (excluding files in .dockerignore)
COPY --chown=user:user api/ api/
COPY --chown=user:user src/ src/
COPY --chown=user:user app.py app.py

# Create a checkpoints directory (weights are usually mounted via volume locally)
RUN mkdir -p checkpoints

EXPOSE 7860

# Run the FastAPI server (which now hosts the Gradio frontend at /)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
