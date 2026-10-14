import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

# Ensure MLflow file store backend is permitted and agent hint suppressed
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add backend directory to sys.path so app modules are resolvable
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.models.registry import registry
from app.config import MODEL_METADATA, MODEL_WEIGHT_FILENAMES, WEIGHTS_DIR
from evaluation.dataset import DeepfakeEvaluationDataset
from evaluation.metrics import (
    calculate_evaluation_metrics,
    generate_confusion_matrix_plot,
    generate_roc_curve_plot,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_engine")

MODELS_ORDER = ["basic_cnn", "resnet18", "efficientnet_b0", "xception"]


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file for experiment reproducibility."""
    if not file_path.is_file():
        return "not_found"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evaluate_single_model(
    model_id: str,
    dataset: DeepfakeEvaluationDataset,
    threshold: float = 0.50,
    batch_size: int = 8,
    output_base_dir: Path = None,
) -> Dict[str, Any]:
    """
    Evaluates one model on the test dataset and saves local diagnostic artifacts.
    """
    logger.info("==================================================")
    logger.info("Evaluating model: %s", model_id)
    logger.info("==================================================")

    # 1. Load model instance from registry
    model, is_mock = registry.get_model(model_id)
    model.eval()
    device = registry.device

    weight_path = registry.resolve_weight_path(model_id)
    checkpoint_hash = compute_file_sha256(weight_path) if weight_path else "mock_no_checkpoint"
    total_params, trainable_params = count_parameters(model)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    y_score_list: List[float] = []
    sample_paths: List[str] = []

    start_eval_time = time.perf_counter()

    with torch.no_grad():
        for batch_tensors, batch_labels, batch_paths in loader:
            batch_tensors = batch_tensors.to(device)
            outputs = model(batch_tensors)

            # Extract positive-class (Real = 1) probabilities
            if outputs.shape[-1] == 1 or outputs.numel() == len(batch_labels):
                scores = torch.sigmoid(outputs).squeeze(-1).cpu().numpy().tolist()
            elif outputs.shape[-1] == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                scores = probs.cpu().numpy().tolist()
            else:
                raise ValueError(f"Unexpected model output shape: {outputs.shape}")

            if isinstance(scores, float):
                scores = [scores]

            for score, label, p in zip(scores, batch_labels.numpy().tolist(), batch_paths):
                pred = 1 if score >= threshold else 0
                y_true_list.append(int(label))
                y_pred_list.append(int(pred))
                y_score_list.append(float(score))
                sample_paths.append(str(p))

    eval_duration = time.perf_counter() - start_eval_time
    avg_latency_ms = (eval_duration / len(dataset) * 1000.0) if len(dataset) > 0 else 0.0

    # 2. Compute metrics
    metrics = calculate_evaluation_metrics(y_true_list, y_pred_list, y_score_list)
    metrics["evaluation_duration_seconds"] = round(eval_duration, 2)
    metrics["avg_inference_latency_ms"] = round(avg_latency_ms, 2)

    # 3. Create local artifacts directory
    model_output_dir = output_base_dir / model_id
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions.csv
    pred_df = pd.DataFrame({
        "image_path": sample_paths,
        "true_label": ["REAL" if y == 1 else "FAKE" for y in y_true_list],
        "predicted_label": ["REAL" if y == 1 else "FAKE" for y in y_pred_list],
        "true_class_id": y_true_list,
        "predicted_class_id": y_pred_list,
        "real_probability": [round(s, 5) for s in y_score_list],
        "fake_probability": [round(1.0 - s, 5) for s in y_score_list],
        "is_correct": [t == p for t, p in zip(y_true_list, y_pred_list)],
    })
    predictions_csv_path = model_output_dir / "predictions.csv"
    pred_df.to_csv(predictions_csv_path, index=False)

    # Save metrics.json
    metrics_json_path = model_output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save classification_report.json
    report_json_path = model_output_dir / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics["classification_report"], f, indent=2)

    # Save confusion_matrix.png
    cm_plot_path = model_output_dir / "confusion_matrix.png"
    generate_confusion_matrix_plot(
        np.array(metrics["confusion_matrix_raw"]),
        model_name=MODEL_METADATA.get(model_id, {}).get("name", model_id),
        output_path=cm_plot_path,
    )

    # Save roc_curve.png
    roc_plot_path = model_output_dir / "roc_curve.png"
    generate_roc_curve_plot(
        y_true_list,
        y_score_list,
        metrics["roc_auc"],
        model_name=MODEL_METADATA.get(model_id, {}).get("name", model_id),
        output_path=roc_plot_path,
    )

    # Save evaluation_config.json
    eval_config = {
        "model_id": model_id,
        "model_name": MODEL_METADATA.get(model_id, {}).get("name", model_id),
        "architecture_details": MODEL_METADATA.get(model_id, {}).get("architecture_details", ""),
        "checkpoint_path": str(weight_path) if weight_path else "missing",
        "checkpoint_sha256": checkpoint_hash,
        "classification_threshold": threshold,
        "positive_class": "REAL (1)",
        "negative_class": "FAKE (0)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "torch_version": torch.__version__,
        "dataset_summary": dataset.get_summary(),
        "is_mock": is_mock,
    }
    config_json_path = model_output_dir / "evaluation_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_config, f, indent=2)

    return {
        "model_id": model_id,
        "model_name": MODEL_METADATA.get(model_id, {}).get("name", model_id),
        "metrics": metrics,
        "config": eval_config,
        "artifacts_dir": model_output_dir,
    }


def log_run_to_mlflow(eval_result: Dict[str, Any], experiment_name: str):
    """Logs evaluation parameters, metrics, tags and artifacts to local MLflow store."""
    try:
        import mlflow

        mlruns_dir = BACKEND_DIR / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.as_uri())
        mlflow.set_experiment(experiment_name)

        model_id = eval_result["model_id"]
        run_name = f"{model_id.replace('_', '-')}-evaluation"

        with mlflow.start_run(run_name=run_name) as run:
            # 1. Log tags
            mlflow.set_tags({
                "task": "binary_deepfake_classification",
                "framework": "pytorch",
                "evaluation_type": "held_out_test_set",
                "model_id": model_id,
                "model_name": eval_result["model_name"],
            })

            # 2. Log parameters
            cfg = eval_result["config"]
            mlflow.log_params({
                "model_id": model_id,
                "model_name": cfg["model_name"],
                "checkpoint_sha256": cfg["checkpoint_sha256"][:16],
                "total_parameters": cfg["total_parameters"],
                "classification_threshold": cfg["classification_threshold"],
                "positive_class": "REAL",
                "negative_class": "FAKE",
                "device": cfg["device"],
                "total_test_samples": cfg["dataset_summary"]["total_samples"],
                "num_real": cfg["dataset_summary"]["num_real"],
                "num_fake": cfg["dataset_summary"]["num_fake"],
            })

            # 3. Log metrics
            m = eval_result["metrics"]
            metrics_dict = {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "specificity": m["specificity"],
                "false_positive_rate": m["false_positive_rate"],
                "false_negative_rate": m["false_negative_rate"],
                "true_positives": m["true_positives"],
                "true_negatives": m["true_negatives"],
                "false_positives": m["false_positives"],
                "false_negatives": m["false_negatives"],
                "evaluation_duration_seconds": m["evaluation_duration_seconds"],
                "avg_inference_latency_ms": m["avg_inference_latency_ms"],
            }
            if m["roc_auc"] is not None:
                metrics_dict["roc_auc"] = m["roc_auc"]

            mlflow.log_metrics(metrics_dict)

            # 4. Log artifacts
            artifacts_dir = eval_result["artifacts_dir"]
            mlflow.log_artifacts(str(artifacts_dir))
            logger.info("Successfully logged MLflow run: %s (Run ID: %s)", run_name, run.info.run_id)

    except ImportError:
        logger.warning("MLflow not installed in environment. Artifacts saved locally only.")
    except Exception as exc:
        logger.error("Failed to log to MLflow: %s", exc)


def update_frontend_models_config(comparison_df: pd.DataFrame):
    """
    Exports verified benchmark numbers directly to frontend/src/config/models.js (or .ts).
    """
    import re
    config_path = BACKEND_DIR.parent / "frontend" / "src" / "config" / "models.js"
    if not config_path.is_file():
        config_path = BACKEND_DIR.parent / "frontend" / "src" / "config" / "models.ts"

    if not config_path.is_file():
        logger.warning("Frontend config not found at %s", config_path)
        return

    logger.info("Exporting measured benchmarks to %s", config_path)
    try:
        content = config_path.read_text(encoding="utf-8")
        for _, row in comparison_df.iterrows():
            mid = str(row["Model ID"])
            acc = round(float(row["Accuracy"]) * 100, 1)
            f1 = round(float(row["F1-Score"]), 3)
            auc_val = row["ROC-AUC"]
            auc_str = None
            if pd.notna(auc_val) and auc_val != "N/A":
                try:
                    auc_str = f"{float(auc_val):.3f}"
                except (ValueError, TypeError):
                    auc_str = None

            # Build the validationMetrics replacement
            metrics_lines = [
                "    validationMetrics: {",
                f"      accuracy: {acc},",
            ]
            if auc_str is not None:
                metrics_lines.append(f"      auc: {auc_str},")
            metrics_lines.append(f"      f1Score: {f1},")
            metrics_lines.append("      isMeasured: true,")
            metrics_lines.append("    }")
            new_block = "\n".join(metrics_lines)

            # Match model object block in models.ts
            pattern = re.compile(
                rf"({re.escape(mid)}:\s*\{{.*?validationMetrics:\s*\{{[^}}]*\}})",
                re.DOTALL,
            )

            def repl(match):
                sub_text = match.group(1)
                sub_updated = re.sub(r"validationMetrics:\s*\{[^}]*\}", new_block.strip(), sub_text)
                return sub_updated

            content, count = pattern.subn(repl, content, count=1)
            if count > 0:
                logger.info("Updated frontend validationMetrics for %s", mid)
            else:
                logger.warning("Model ID %s not found in frontend config", mid)

        config_path.write_text(content, encoding="utf-8")
        logger.info("Frontend models config updated successfully.")
    except Exception as exc:
        logger.error("Failed to export to frontend: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models and log experiments with MLflow.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(BACKEND_DIR.parent / "dataset" / "test"),
        help="Path to labeled evaluation test directory containing fake/ and real/ subdirectories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_ORDER,
        help="Model identifiers to evaluate (default: basic_cnn resnet18 efficientnet_b0 xception)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Classification threshold for positive class REAL (default: 0.50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model inference (default: 8)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="deepfake-model-evaluation",
        help="MLflow experiment name (default: deepfake-model-evaluation)",
    )
    parser.add_argument(
        "--export-to-frontend",
        action="store_true",
        help="Export measured metrics to frontend/src/config/models.ts",
    )
    args = parser.parse_args()

    test_dir = Path(args.dataset_dir)
    if not test_dir.is_dir():
        for candidate in [
            BACKEND_DIR.parent / "dataset" / "Test",
            BACKEND_DIR.parent / "dataset" / "test",
            BACKEND_DIR.parent / "dataset",
        ]:
            if candidate.is_dir():
                test_dir = candidate
                break

    if not test_dir.is_dir():
        logger.error("Evaluation dataset directory not found: %s", test_dir)
        logger.error("Please provide a dataset folder with 'Fake/' and 'Real/' subdirectories.")
        sys.exit(1)

    logger.info("Loading test dataset from %s ...", test_dir)
    dataset = DeepfakeEvaluationDataset(test_dir, validate_on_init=True)
    summary = dataset.get_summary()
    logger.info("Dataset statistics: %s", json.dumps(summary, indent=2))

    if len(dataset) == 0:
        logger.error("No valid test images found in %s/fake or %s/real.", test_dir, test_dir)
        logger.error("Place .jpg or .png images in these folders before running evaluation.")
        sys.exit(1)

    results_base_dir = BACKEND_DIR / "evaluation" / "results"
    results_base_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for model_id in args.models:
        try:
            res = evaluate_single_model(
                model_id=model_id,
                dataset=dataset,
                threshold=args.threshold,
                batch_size=args.batch_size,
                output_base_dir=results_base_dir,
            )
            log_run_to_mlflow(res, experiment_name=args.experiment_name)
            all_results.append(res)
        except Exception as exc:
            logger.exception("Failed to evaluate model %s: %s", model_id, exc)

    if not all_results:
        logger.error("No models were successfully evaluated.")
        sys.exit(1)

    # Generate Comparison Table
    comparison_rows = []
    for r in all_results:
        m = r["metrics"]
        cfg = r["config"]
        comparison_rows.append({
            "Model ID": r["model_id"],
            "Model Name": r["model_name"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall (Sensitivity)": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"] if m["roc_auc"] is not None else "N/A",
            "Specificity": m["specificity"],
            "False Positive Rate": m["false_positive_rate"],
            "False Negative Rate": m["false_negative_rate"],
            "Total Params": cfg["total_parameters"],
            "Latency (ms)": m["avg_inference_latency_ms"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save CSV and JSON
    comparison_csv_path = results_base_dir / "model_comparison.csv"
    comparison_json_path = results_base_dir / "model_comparison.json"
    comparison_df.to_csv(comparison_csv_path, index=False)
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    logger.info("================================================================================")
    logger.info("MODEL EVALUATION COMPARISON REPORT")
    logger.info("================================================================================")
    print("\n" + comparison_df.to_string(index=False) + "\n")

    # Best-performer analysis (without declaring absolute universal winner)
    print("--- Performance Highlights ---")
    best_acc = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    best_f1 = comparison_df.loc[comparison_df["F1-Score"].idxmax()]
    best_rec = comparison_df.loc[comparison_df["Recall (Sensitivity)"].idxmax()]
    best_spec = comparison_df.loc[comparison_df["Specificity"].idxmax()]

    print(f"* Best Overall Accuracy : {best_acc['Model Name']} ({best_acc['Accuracy']*100:.2f}%)")
    print(f"* Best F1-Score         : {best_f1['Model Name']} ({best_f1['F1-Score']:.4f})")
    print(f"* Best Recall (Real)    : {best_rec['Model Name']} ({best_rec['Recall (Sensitivity)']*100:.2f}%)")
    print(f"* Best Specificity (Fake): {best_spec['Model Name']} ({best_spec['Specificity']*100:.2f}%)")

    logger.info("Artifacts and comparison table written to: %s", results_base_dir)
    logger.info("Run 'run_mlflow.bat' to inspect runs in the interactive dashboard.")

    if args.export_to_frontend:
        update_frontend_models_config(comparison_df)


if __name__ == "__main__":
    main()
