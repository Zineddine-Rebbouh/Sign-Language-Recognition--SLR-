# Sign Language Recognition (SLR) using WLASL Dataset

A comprehensive Sign Language Recognition system implementing multiple model architectures for video-based sign language classification.

## 📋 Project Overview

This project implements a complete Sign Language Recognition pipeline using the WLASL (World Level American Sign Language) Video Dataset. It includes both baseline and advanced deep learning models for video classification.

**Dataset**: [WLASL Processed on Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

## 🎯 Project Structure

### Phase 1: Baseline Model (Mandatory)
- **HOG + SVM**: Histogram of Oriented Gradients with Support Vector Machine
  - Linear and RBF kernels
  - Comprehensive evaluation metrics

### Phase 2: Advanced Models

#### 1. Traditional CV Models
- HOG + CNN
- HOG + LBPH
- Inception + HOG + LBPH + KNN

#### 2. 2D Image-based Models
- VGG (11, 16, 19)
- ResNet (18, 34, 50, 101)
- MobileNet
- AlexNet
- InceptionV3

#### 3. Sequence Models
- CNN + LSTM
- CNN + GRU
- 2D CNN → LSTM

#### 4. Spatiotemporal Models
- 3D CNN
- C3D
- I3D

#### 5. Transformer Models
- CNN + Transformer Encoder
- TimeSformer
- Video Swin Transformer

#### 6. Keypoint Models
- GCN (Graph Convolutional Network)
- ST-GCN (Spatial-Temporal GCN)
- Keypoint LSTM (with MediaPipe)

## 📁 File Structure

```
.
├── av.ipynb                          # Main Kaggle notebook
├── baseline_hog_svm.py               # Baseline HOG+SVM implementation
├── utils_video.py                    # Video processing utilities
├── evaluate.py                       # Comprehensive evaluation module
├── compare_models.py                 # Model comparison script
├── config_template.yaml              # Configuration template
├── train_template.py                 # Training template
├── models/
│   ├── __init__.py
│   ├── 2d_cnn_models.py              # VGG, ResNet, MobileNet, etc.
│   ├── sequence_models.py            # CNN+LSTM, CNN+GRU
│   ├── spatiotemporal_models.py     # 3D CNN, C3D, I3D
│   ├── transformer_models.py         # Transformer-based models
│   ├── keypoint_models.py            # GCN, ST-GCN, Keypoint LSTM
│   └── traditional_cv_models.py     # HOG+CNN, HOG+LBPH, etc.
└── README.md
```

## 🚀 Quick Start (Kaggle)

1. **Upload the dataset** to your Kaggle notebook:
   - Add dataset: `risangbaskoro/wlasl-processed`

2. **Run the notebook**:
   - Open `av.ipynb` in Kaggle
   - Run all cells sequentially
   - The baseline model will train automatically

3. **For Phase 2 models**:
   - Use the provided model architectures in `models/`
   - Create training scripts based on `train_template.py`
   - Configure using `config_template.yaml`

## 📊 Evaluation Metrics

The project implements comprehensive evaluation metrics:

### Quantitative Metrics
- **Accuracy**: Standard classification accuracy
- **Top-k Accuracy**: Top-k prediction accuracy (k=1, 3, 5)
- **Precision, Recall, F1-Score**: Per-class and weighted averages
- **Confusion Matrix**: Visualized and saved
- **Sequence Accuracy**: Exact sequence match accuracy
- **mAP**: Mean Average Precision
- **Temporal Detection Accuracy**: Frame-level detection with tolerance
- **Frame-wise Accuracy**: Per-frame classification accuracy

### Qualitative Metrics
- Robustness tests
- Cross-signer generalization
- Smoothness of predicted sequences
- Keypoint correctness (for keypoint models)

## 🔧 Configuration

Each model can be configured using YAML files. See `config_template.yaml` for options:

```yaml
model:
  name: "resnet18"
  num_classes: 100
  pretrained: true

data:
  dataset_root: "/kaggle/input/wlasl-processed"
  batch_size: 32
  target_size: [224, 224]

training:
  num_epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
```

## 📈 Model Comparison

Use `compare_models.py` to compare all trained models:

```bash
python compare_models.py \
    --results_dir /kaggle/working/slr_results \
    --output /kaggle/working/model_comparison.csv \
    --plot /kaggle/working/model_comparison.png
```

## 🧪 Experiment Management

- **Runs Directory**: Auto-organized by model name + timestamp
- **Training Curves**: Loss and accuracy plots
- **Results Summary**: CSV/JSON files with all metrics
- **Model Checkpoints**: Best models saved automatically
- **Early Stopping**: Configurable patience

## 📝 Usage Examples

### Baseline Model (Phase 1)

```python
from baseline_hog_svm import train_baseline_hog_svm
from pathlib import Path

results = train_baseline_hog_svm(
    dataset_root=Path("/kaggle/input/wlasl-processed"),
    output_dir=Path("/kaggle/working/baseline"),
    kernel='linear',
    max_videos_per_class=50
)
```

### 2D CNN Model (Phase 2)

```python
from models import get_2d_cnn_model
import torch

model = get_2d_cnn_model("resnet18", num_classes=100, pretrained=True)
model = model.to(device)
```

### Sequence Model

```python
from models.sequence_models import CNNLSTM

model = CNNLSTM(
    num_classes=100,
    cnn_backbone='resnet18',
    hidden_size=256,
    num_layers=2
)
```

## 🔬 Requirements

### Core Dependencies
- Python 3.7+
- PyTorch 1.8+
- OpenCV
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

### Optional (for specific models)
- MediaPipe (for keypoint extraction)
- torchvision (for pretrained models)
- PyYAML (for configuration files)

## 📄 License

This project is for educational purposes. Please cite the WLASL dataset if you use it in your research.

## 🤝 Contributing

This is a course project. For improvements or bug fixes, please create issues or pull requests.

## 📚 References

- WLASL Dataset: [Paper](https://arxiv.org/abs/1910.11006)
- Evaluation Metrics: Based on standard SLR evaluation criteria

## ⚠️ Notes

- The notebook is designed for **Kaggle environment** (not local PC)
- Dataset paths are configured for Kaggle (`/kaggle/input/...`)
- GPU acceleration is recommended for Phase 2 models
- Some models (e.g., full I3D, Video Swin) are simplified versions
- MediaPipe integration requires additional setup

## 🎓 For Students

This project structure follows best practices:
- Clean, modular code
- Comprehensive evaluation
- Proper logging and error handling
- Experiment management
- Production-ready structure

Good luck with your Sign Language Recognition project! 🎉

