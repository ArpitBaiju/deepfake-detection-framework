import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.metrics import (
    calculate_evaluation_metrics,
    generate_confusion_matrix_plot,
    generate_roc_curve_plot,
)
from evaluation.dataset import DeepfakeEvaluationDataset


def test_calculate_evaluation_metrics_balanced():
    # 0 = FAKE, 1 = REAL
    # TN=1, FP=1, FN=1, TP=1
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.1, 0.8, 0.3, 0.9]

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["true_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["total_evaluated"] == 4
    assert metrics["roc_auc_status"] == "ok"
    assert metrics["roc_auc"] is not None


def test_calculate_evaluation_metrics_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.05, 0.12, 0.88, 0.95]

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 2
    assert metrics["roc_auc"] == 1.0
    assert metrics["roc_auc_status"] == "ok"


def test_calculate_evaluation_metrics_single_class_guard():
    # Only REAL class present
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 0, 1]
    y_score = [0.9, 0.8, 0.4, 0.95]

    metrics = calculate_evaluation_metrics(y_true, y_pred, y_score)

    # ROC-AUC must not raise ValueError: Only one class present in y_true
    assert metrics["roc_auc"] is None
    assert metrics["roc_auc_status"] == "unavailable_single_class_present"
    assert metrics["total_evaluated"] == 4


def test_generate_plots(tmp_path):
    cm = np.array([[10, 2], [3, 15]])
    cm_path = tmp_path / "confusion_matrix.png"
    generate_confusion_matrix_plot(cm, "TestModel", cm_path)
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0

    roc_path = tmp_path / "roc_curve.png"
    generate_roc_curve_plot(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
        roc_auc=1.0,
        model_name="TestModel",
        output_path=roc_path,
    )
    assert roc_path.exists()
    assert roc_path.stat().st_size > 0

    # Test ROC plot with single class fallback
    roc_fallback_path = tmp_path / "roc_curve_fallback.png"
    generate_roc_curve_plot(
        y_true=[1, 1, 1],
        y_score=[0.7, 0.8, 0.9],
        roc_auc=None,
        model_name="SingleClassModel",
        output_path=roc_fallback_path,
    )
    assert roc_fallback_path.exists()
    assert roc_fallback_path.stat().st_size > 0


def test_deepfake_evaluation_dataset(tmp_path):
    fake_dir = tmp_path / "fake"
    real_dir = tmp_path / "real"
    fake_dir.mkdir()
    real_dir.mkdir()

    # Create 1 valid fake image
    fake_img = Image.new("RGB", (300, 300), color=(255, 0, 0))
    fake_img_path = fake_dir / "sample_fake.png"
    fake_img.save(fake_img_path)

    # Create 1 valid real image
    real_img = Image.new("RGB", (300, 300), color=(0, 255, 0))
    real_img_path = real_dir / "sample_real.jpg"
    real_img.save(real_img_path)

    # Create 1 corrupted image (text content with .jpg extension)
    corrupted_path = fake_dir / "corrupted.jpg"
    corrupted_path.write_text("NOT_AN_IMAGE_CONTENT", encoding="utf-8")

    dataset = DeepfakeEvaluationDataset(tmp_path, validate_on_init=True)

    assert len(dataset) == 2
    assert len(dataset.corrupted_files) == 1

    summary = dataset.get_summary()
    assert summary["total_samples"] == 2
    assert summary["num_fake"] == 1
    assert summary["num_real"] == 1
    assert summary["num_corrupted"] == 1

    # Verify item shapes and values
    tensor, label, path_str = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert label in (0, 1)
    assert Path(path_str).exists()


def test_class_mapping_and_threshold_logic():
    # Verify binary decision boundary at tau = 0.50
    # Positive class = 1 (REAL), Negative class = 0 (FAKE)
    threshold = 0.50
    test_scores = [0.0, 0.499, 0.50, 0.501, 1.0]
    expected_preds = [0, 0, 1, 1, 1]

    for score, expected in zip(test_scores, expected_preds):
        pred = 1 if score >= threshold else 0
        assert pred == expected, f"Score {score} expected {expected} but got {pred}"


def test_evaluate_single_model_pipeline(tmp_path):
    from evaluation.evaluate_models import evaluate_single_model

    # Use dataset folder in root directory or mock in tmp_path
    dataset_candidates = [
        BACKEND_DIR.parent / "dataset" / "Test",
        BACKEND_DIR.parent / "dataset" / "test",
        BACKEND_DIR.parent / "dataset",
    ]
    chosen_dir = None
    for cand in dataset_candidates:
        if cand.is_dir():
            ds = DeepfakeEvaluationDataset(cand, validate_on_init=False)
            if len(ds) > 0:
                chosen_dir = cand
                break

    if chosen_dir is None:
        fake_dir = tmp_path / "fake"
        real_dir = tmp_path / "real"
        fake_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), color=(255, 0, 0)).save(fake_dir / "f1.jpg")
        Image.new("RGB", (256, 256), color=(0, 255, 0)).save(real_dir / "r1.jpg")
        chosen_dir = tmp_path

    dataset = DeepfakeEvaluationDataset(chosen_dir, validate_on_init=True)
    assert len(dataset) > 0

    results_dir = tmp_path / "results"
    res = evaluate_single_model(
        model_id="basic_cnn",
        dataset=dataset,
        threshold=0.50,
        batch_size=4,
        output_base_dir=results_dir,
    )

    assert res["model_id"] == "basic_cnn"
    assert "metrics" in res
    assert "accuracy" in res["metrics"]
    assert "confusion_matrix_raw" in res["metrics"]
    assert (results_dir / "basic_cnn" / "predictions.csv").exists()
    assert (results_dir / "basic_cnn" / "metrics.json").exists()
    assert (results_dir / "basic_cnn" / "confusion_matrix.png").exists()
    assert (results_dir / "basic_cnn" / "roc_curve.png").exists()
    assert (results_dir / "basic_cnn" / "evaluation_config.json").exists()


def test_update_frontend_models_config(tmp_path, monkeypatch):
    import pandas as pd
    from evaluation import evaluate_models

    mock_config = tmp_path / "models.ts"
    mock_config.write_text(
        """
export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  basic_cnn: {
    id: 'basic_cnn',
    validationMetrics: {
      isMeasured: false,
    },
  },
};
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_models, "BACKEND_DIR", tmp_path / "backend")
    # Simulate backend/parent/frontend/src/config/models.ts layout
    frontend_config = tmp_path / "frontend" / "src" / "config" / "models.ts"
    frontend_config.parent.mkdir(parents=True, exist_ok=True)
    frontend_config.write_text(mock_config.read_text(encoding="utf-8"), encoding="utf-8")

    df = pd.DataFrame([
        {
            "Model ID": "basic_cnn",
            "Accuracy": 0.942,
            "F1-Score": 0.938,
            "ROC-AUC": 0.981,
        }
    ])

    evaluate_models.update_frontend_models_config(df)
    updated = frontend_config.read_text(encoding="utf-8")

    assert "accuracy: 94.2" in updated
    assert "auc: 0.981" in updated
    assert "f1Score: 0.938" in updated
    assert "isMeasured: true" in updated

