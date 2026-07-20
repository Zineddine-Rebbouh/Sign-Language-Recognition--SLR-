# Sign Language Recognition (SLR) & MLOps System

> Custom 3D-CNN achieving **90.71% validation accuracy** on 20 ASL gesture classes.
> Upgraded into a complete **MLOps** ecosystem featuring MLflow tracking, automated CI/CD testing, FastAPI serving, and a Gradio frontend — all running in a single lightweight Docker container.

[![CI Pipeline](https://github.com/your-username/sign-language-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/sign-language-recognition/actions/workflows/ci.yml)
[![Deployed on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/sign-language-api)

---

## Table of Contents
1. [Project Description](#project-description)
2. [MLOps & Pipeline Architecture](#mlops--pipeline-architecture)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)
6. [Deployment](#deployment)
7. [Model Architecture](#model-architecture)
8. [Known Limitations](#known-limitations)

---

## Project Description

This project implements a robust Sign Language Recognition (SLR) system, graduating from a research script into a production-ready application. 

### Core Machine Learning Principles
- **No data leakage** — the train/val/test split is performed at the *video* level. No two frames from the same clip appear in different partitions.
- **Preprocessing consistency** — the exact same `ImprovedVideoProcessor` class is used during training and real-time inference (via the API).
- **Reproducibility** — every experiment is seeded (Python, NumPy, PyTorch, cuDNN). The seed is stored inside the checkpoint.

### MLOps Principles Added
- **Experiment Tracking**: Integrated **MLflow** to log hyperparameters, metrics, and models.
- **Unified Serving**: **FastAPI** provides a robust REST API for developers, while **Gradio** provides a beautiful web UI for users. Both run on the same server.
- **Containerization**: A multi-stage **Docker** build separates compilation from runtime, keeping the image slim while satisfying tricky OpenCV system dependencies (`libgl1-mesa-glx`).
- **CI/CD**: **GitHub Actions** automatically runs Pytest suites and smoke-tests the Docker container on every pull request.

---

## MLOps & Pipeline Architecture

```mermaid
flowchart TD
    %% Define styles
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef process fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef deploy fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef infra fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000;

    subgraph Development Pipeline
        RawData[("Raw Videos<br>(ASL-20)")]:::data
        Split["Video-Level Split<br>(70/15/15)"]:::process
        Train["scripts/train.py<br>Model Training"]:::model
        MLflow[("MLflow Tracking<br>(Metrics, Params, Artifacts)")]:::infra
        
        RawData --> Split --> Train
        Train <--> MLflow
        Train -- "Saves best_model.pth" --> Checkpoints[("Checkpoints")]:::data
    end

    subgraph Production Docker Container (Port 7860)
        direction TB
        GradioUI["Gradio UI<br>(/ root)"]:::deploy
        FastAPI["FastAPI REST<br>(/api/predict)"]:::deploy
        ModelService["ModelService<br>(Loads Model Once)"]:::process
        Preprocessing["ImprovedVideoProcessor<br>(MediaPipe ROI, CLAHE, Resize)"]:::process
        
        GradioUI -- "In-process call" --> ModelService
        FastAPI -- "Dependency Injection" --> ModelService
        ModelService --> Preprocessing
    end

    subgraph CI/CD (GitHub Actions)
        Pytest["Unit Tests<br>(tests/)"]:::process
        SmokeTest["Docker Build & Smoke Test<br>(curl /health)"]:::process
    end

    Checkpoints -. "Mounted/Copied" .-> ModelService
    User((User/Recruiter)) --> GradioUI
    Dev((Developer)) --> FastAPI
```

---

## Repository Structure

```text
.
├── api/                        # FastAPI Serving Package
│   ├── __init__.py
│   ├── inference.py            # ModelService (connects API to ML logic)
│   ├── main.py                 # FastAPI & Gradio endpoints
│   └── schemas.py              # Pydantic validation models
├── checkpoints/                # Saved .pth weights
├── data/                       # Datasets
├── notebooks/                  # Research notebooks
├── scripts/                    # Entry-point runner scripts
│   ├── train.py                # Train 3D-CNN (MLflow enabled)
│   ├── evaluate_checkpoint.py  
│   └── realtime_inference.py   # Live webcam demo
├── src/                        # Core ML logic (Importable)
│   ├── dataset.py              
│   ├── evaluate.py             
│   ├── model_3dcnn.py          # 14.3M param architecture
│   ├── preprocessing.py        # Video Processor (Shared by API & Train)
│   └── utils_video.py          
├── tests/                      # Pytest suite
│   ├── test_api.py             # Endpoint tests
│   ├── test_model.py           # Architecture tensor tests
│   └── test_preprocessing.py   # Tensor shape validation
├── .github/workflows/          # CI/CD configuration
│   └── ci.yml                  
├── app.py                      # Gradio Frontend definition
├── Dockerfile                  # Multi-stage production container
├── docker-compose.yml          # Local development stack
├── requirements.txt            
└── README.md
```

---

## Setup & Installation

### Local Python Environment

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
pip install -r requirements.txt
```

*(Note: CPU users can save bandwidth by installing the CPU version of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)*

### Local Docker Environment (Recommended)

To spin up the API and MLflow tracking server locally:

```bash
docker compose up -d --build
```
- API & Gradio UI: `http://localhost:7860`
- API Docs (Swagger): `http://localhost:7860/docs`
- MLflow UI: `http://localhost:5000`

---

## Usage

### 1. Training with MLflow

Training is fully tracked by MLflow automatically.

```bash
python scripts/train.py \
    --dataset_root "./data/raw/Full Data" \
    --epochs 50 --batch_size 8 \
    --experiment_name "SLR-3DCNN" \
    --run_test  # Evaluates on the test set at the end
```

View the learning curves, hyperparameters, and confusion matrices by opening `http://localhost:5000` (if running docker compose) or `mlflow ui --backend-store-uri ./mlruns`.

### 2. Testing Locally

```bash
# Run unit tests
pytest tests/ -v
```

---

## Deployment

This system is designed for instant, free deployment to **Hugging Face Spaces** using the Docker SDK.

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Docker** as the SDK.
3. Link the remote and push:
   ```bash
   git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/sign-language-api
   git push huggingface main
   ```
4. HF Spaces will automatically build the `Dockerfile` and expose port `7860`. The Gradio UI will be visible immediately, and the REST API will be available for remote calls.

---

## Model Architecture

### Improved3DCNN (14.3 M parameters)

| Layer block | Output shape | Notes |
|-------------|-------------|-------|
| Input | (B, 3, 30, 112, 112) | RGB × frames × H × W |
| Conv3D block 1 | (B, 64, 30, 56, 56) | 3×3×3 conv, BN, ReLU, MaxPool(1,2,2) |
| Conv3D block 2 | (B, 128, 15, 28, 28) | MaxPool(2,2,2) |
| Conv3D block 3 | (B, 256, 7, 14, 14) | Two 3×3×3 conv layers |
| Conv3D block 4 | (B, 512, 3, 7, 7) | Two 3×3×3 conv layers |
| Conv3D block 5 | (B, 512, 1, 3, 3) | Two 3×3×3 conv layers |
| AdaptiveAvgPool3d | (B, 512) | Flattened |
| Classifier | (B, 20) | Final logits |

**Design decisions:**
- **3D convolutions** capture both spatial hand shape AND temporal motion trajectory simultaneously.
- **Label smoothing (ε=0.1)** prevents overconfident predictions.
- **Cosine LR annealing** smoothly reduces the learning rate.

> The model achieved **90.71% validation accuracy** on the ASL-20 dataset. To reproduce exactly, run `scripts/train.py` with `--seed 42`.

---

## Known Limitations

1. **No temporal augmentation** — The dataset is small (~20 classes × N videos). Augmentations like random frame drops or speed changes could improve generalisation.
2. **Single-hand only** — `HandDetector` extracts one bounding box encompassing all detected hands; two-handed signs may be sub-optimally cropped.
3. **Real-time CPU latency** — On CPU, 3D convolutions over 30 frames are slow (~2–4 FPS). For large scale deployment, consider model quantisation (`torch.quantization`) or converting to ONNX + TensorRT.
