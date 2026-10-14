# Deepfake Detection Framework

A modern, high-performance web application, deep learning inference service, and systematic model evaluation suite for detecting manipulated and synthetic facial media. Allows users to upload an image of a human face and evaluate whether it is **REAL** or **FAKE** across four deep-learning convolutional neural network architectures.

![Deepfake Detection Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20PyTorch-blue)
![Frontend](https://img.shields.io/badge/Frontend-React%20%7C%20TypeScript%20%7C%20Tailwind-38bdf8)
![Inference](https://img.shields.io/badge/Inference-torch.no__grad%20%7C%20CUDA%2FCPU-emerald)
![Evaluation](https://img.shields.io/badge/Evaluation-MLflow%20%7C%20scikit--learn-orange)

---

## Supported Model Architectures

The framework is configured to interface with four binary image-classification models:

| Model | Architectural Role | Characteristics & Strengths |
| :--- | :--- | :--- |
| **Basic CNN** | Baseline | 4-block sequential ConvNet capturing low-to-mid spatial features with low latency. |
| **ResNet18** | Residual CNN | Deep residual network with identity skip connections preventing gradient degradation. |
| **EfficientNet-B0** | Efficient Architecture | Compound coefficient scaling with inverted residual MBConv blocks and Squeeze-and-Excitation. |
| **Xception** | Advanced CNN | Extreme Inception with depthwise separable convolutions decoupling spatial and cross-channel filters. |

### Binary Class Mapping
- **`0`**: Fake (Synthetic / Manipulated)
- **`1`**: Real (Authentic)

### Image Preprocessing Pipeline
All inputs are automatically processed through the exact training pipeline:
- **Spatial Resolution**: Resized to `256 × 256` RGB
- **Channel Normalization**:
  - `Mean = [0.485, 0.456, 0.406]`
  - `Std = [0.229, 0.224, 0.225]`
- **Inference Mode**: `model.eval()` executed inside `torch.no_grad()`
- **Device Support**: CUDA GPU when available, CPU fallback otherwise

---

## Model Evaluation Pipeline (MLflow + scikit-learn)

The framework includes a standardized offline evaluation suite that benchmarks all four models under identical conditions on a held-out test dataset, recording full lineage, metrics, and visual artifacts into **MLflow**.

### 1. Dataset Directory Setup
Place labeled test images inside `dataset/test/` organized by class:
```
dataset/
└── test/
    ├── fake/          # Class 0: Synthetic / Manipulated face images (.jpg, .png, .webp)
    └── real/          # Class 1: Authentic face images (.jpg, .png, .webp)
```

### 2. Run Model Evaluation
Run the evaluation suite using the Windows launcher:
```cmd
run_evaluation.bat
```
Or directly via Python CLI:
```powershell
python backend/evaluation/evaluate_models.py --dataset-dir dataset/test --batch-size 8
```

#### CLI Options:
- `--dataset-dir`: Path to folder containing `fake/` and `real/` subfolders (default: `dataset/test`).
- `--models`: List of models to evaluate (default: `basic_cnn resnet18 efficientnet_b0 xception`).
- `--threshold`: Decision threshold for class 1 REAL (default: `0.50`).
- `--batch-size`: Batch size for DataLoader (default: `8`).
- `--no-mlflow`: Disable MLflow tracking and save only local disk artifacts.

### 3. Computed Metrics
For each model, the evaluation suite computes:
- **Accuracy**: Overall fraction of correct classifications.
- **Precision**: Real prediction accuracy ($TP / (TP + FP)$).
- **Recall (Sensitivity)**: True Real detection rate ($TP / (TP + FN)$).
- **Specificity**: True Fake detection rate ($TN / (TN + FP)$).
- **False Positive Rate (FPR)**: Real falsely classified as Fake.
- **False Negative Rate (FNR)**: Fake falsely classified as Real.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve (with single-class presence guard).
- **Latency**: Mean per-image inference latency (ms).

### 4. Diagnostic Artifacts Generated
For each model run, artifacts are saved in `backend/evaluation/results/<model_id>/` and logged into MLflow:
- `predictions.csv`: Per-sample table with image path, true label, predicted label, raw logit, and probability.
- `metrics.json`: All computed scikit-learn scores.
- `classification_report.json`: Per-class precision, recall, f1, and support.
- `confusion_matrix.png`: Publication-grade annotated $2 \times 2$ heatmap with TN, FP, FN, TP cell counts.
- `roc_curve.png`: ROC curve plot with AUC annotation and diagonal baseline.
- `model_comparison.csv` / `model_comparison.json`: Cross-model comparative leaderboard.

### 5. Launch MLflow Dashboard
View experiments, compare curves, inspect parameters, and download artifacts interactively:
```cmd
run_mlflow_ui.bat
```
Open `http://127.0.0.1:5000` in your browser.

---

## Directory Structure

```
deepfake-detection-framework/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI endpoints & CORS
│   │   ├── config.py                # Environment settings, normalization, metadata
│   │   ├── schemas.py               # Pydantic schemas (PredictionResponse, ModelInfo, Health)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── basic_cnn.py         # Baseline ConvNet architecture
│   │   │   ├── resnet18.py          # ResNet18 adapted for 1-logit binary output
│   │   │   ├── efficientnet.py      # EfficientNet-B0 adapted for 1-logit binary output
│   │   │   ├── xception.py          # Xception builder (timm + separable fallback)
│   │   │   └── registry.py          # Model loader, caching, device & weights resolution
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── preprocessing.py     # 256x256 resize & ImageNet normalization
│   │       └── inference.py         # Sigmoid probability, latency & verdict computation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── dataset.py               # DeepfakeEvaluationDataset loader & Pillow verification
│   │   ├── metrics.py               # Scikit-learn metric calculations, CM & ROC plotters
│   │   ├── evaluate_models.py       # Benchmark runner, MLflow logger & comparison builder
│   │   └── results/                 # Evaluation output directory (CSV, JSON, PNG plots)
│   ├── mlruns/                      # Local MLflow experiment tracking store
│   ├── tests/
│   │   └── test_evaluation.py       # PyTest suite for metrics, single-class guard & loader
│   ├── weights/                     # Directory for .pth checkpoint files
│   ├── requirements.txt             # Core backend dependencies
│   ├── requirements-evaluation.txt  # MLflow, scikit-learn, matplotlib dependencies
│   └── run_mlflow.bat               # MLflow UI launcher (backend dir)
├── dataset/
│   ├── README.md                    # Dataset format and directory structure instructions
│   └── test/
│       ├── fake/                    # Test fake samples (.gitkeep)
│       └── real/                    # Test real samples (.gitkeep)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx           # Forensic title, status indicators, hardware device
│   │   │   ├── ImageUpload.tsx      # Clean drag-and-drop zone with preview & validation
│   │   │   ├── ModelSelector.tsx    # Radio model selector with weights status
│   │   │   ├── ResultsPanel.tsx     # Clean verdict (REAL/FAKE), confidence bar, model description
│   │   │   ├── HistoryPanel.tsx     # LocalStorage forensic history & clear controls
│   │   │   └── Footer.tsx           # Academic disclaimer & pipeline specs
│   │   ├── config/
│   │   │   └── models.ts            # Central model configuration & measured benchmarks
│   │   ├── services/
│   │   │   └── api.ts               # Typed REST API client
│   │   ├── hooks/
│   │   │   └── useHistory.ts        # Persistent local history hook
│   │   ├── types/
│   │   │   └── index.ts             # TypeScript interfaces
│   │   ├── App.tsx                  # Widescreen single-card layout
│   │   └── index.css                # Times New Roman typography & clean styling
│   ├── index.html
│   ├── package.json
│   └── tailwind.config.js
├── run_evaluation.bat               # Windows batch launcher for evaluation suite
├── run_mlflow_ui.bat                # Windows batch launcher for MLflow dashboard
├── run_backend.bat                  # Backend server launcher
├── run_frontend.bat                 # Frontend server launcher
├── start.bat                        # Full-stack launcher (backend + frontend)
└── README.md
```

---

## Connecting Trained Model Weights

Place your trained `.pth` files into `backend/weights/`:

```
backend/
└── weights/
    ├── basic_cnn.pth
    ├── resnet18.pth
    ├── efficientnet_b0.pth
    └── xception_current.pth       # or xception.pth
```

> [!IMPORTANT]
> The application **does not fabricate results** when weights are absent. If a model file is missing, the API returns a structured `503 Service Unavailable` response and the frontend displays clear instructions for where to drop the file.

---

## Quick Start Guide

### Option 1: One-Click Full-Stack Launch
Double-click `start.bat` in the root folder. It will launch both the backend API and frontend dev server in separate windows.

### Option 2: Manual Terminal Commands

#### 1. Start the Backend API
```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API is running at: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

#### 2. Start the Frontend Application
```powershell
cd frontend
npm install
npm run dev
```
- Dashboard opens at: `http://localhost:5173`

---

## API Endpoints

### `POST /predict`
Uploads a facial image and evaluates it using the selected deep learning model.

**Request (`multipart/form-data`)**:
- `image`: Image file (JPG, JPEG, PNG, WEBP)
- `model`: Model key (`basic_cnn`, `resnet18`, `efficientnet_b0`, `xception`)

**Response (`application/json`)**:
```json
{
  "prediction": "REAL",
  "class_id": 1,
  "confidence": 0.9362,
  "model": "basic_cnn",
  "processing_time_ms": 68.87,
  "probabilities": {
    "fake": 0.0638,
    "real": 0.9362
  },
  "is_mock": false,
  "device_used": "cpu"
}
```

### `GET /models`
Returns list of the 4 models, architecture descriptions, and disk weight availability.

### `GET /health`
Returns system status, PyTorch version, timm version, and active execution device (`cuda` or `cpu`).

---

## License & Advisory

This application is provided for academic, research, and media forensic verification purposes. Results generated by deep learning models are probabilistic estimates and should be corroborated by multimodal forensics before making consequential determinations.
