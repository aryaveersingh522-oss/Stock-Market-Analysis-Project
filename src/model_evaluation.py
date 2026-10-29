"""
Model Evaluation and Comparison Suite
====================================
Evaluates the 5 syllabus-specified machine learning models across:
- Accuracy
- Logarithmic Loss (Log Loss)
- Area Under ROC Curve (ROC-AUC)
- Precision
- Recall
- F1 Score
- Confusion Matrix (TP, FP, TN, FN breakdown)

Generates simple, clean visualizations:
- Bar Charts for metrics comparison, confusion matrix elements, feature importances.
- Line Chart for ROC Curves.

Syllabus Alignment:
- Unit III: Supervised Learning: Classification (Accuracy, Log Loss, ROC-AUC, Precision, Recall, F1, Confusion Matrix).
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard aesthetic styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def evaluate_all_models(
    predictions_path: str = "models/test_predictions.json",
    models_path: str = "models/all_models.joblib",
    output_dir: str = "reports/figures",
    metadata_dir: str = "models"
) -> pd.DataFrame:
    """Compute all evaluation metrics and generate standard simple visual charts."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    with open(predictions_path, "r") as f:
        data = json.load(f)

    y_true = np.array(data["y_true"])
    predictions = data["predictions"]
    probabilities = data["probabilities"]
    feature_names = data["feature_names"]

    metrics_list = []
    cm_dict = {}

    for name in predictions.keys():
        y_pred = np.array(predictions[name])
        y_prob = np.array(probabilities[name])

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)

        # Clip probability for robust numerical stability in log loss
        clipped_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
        loss = log_loss(y_true, clipped_prob)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cm_dict[name] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": auc,
            "Log_Loss": loss
        })

    metrics_df = pd.DataFrame(metrics_list)
    logger.info("Evaluation Performance Table:\n" + metrics_df.to_string(index=False))

    # Identify Best-Fit Model (highest F1-score with high accuracy)
    best_model_row = metrics_df.sort_values(by=["F1_Score", "Accuracy"], ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    logger.info(f"==> Best-Fit Model Selected: [{best_model_name}] with F1-Score: {best_model_row['F1_Score']:.4f}")

    # Serialize best model separately
    all_models = joblib.load(models_path)
    joblib.dump(all_models[best_model_name], os.path.join(metadata_dir, "best_model.joblib"))

    # -------------------------------------------------------------
    # CHART 1: Model Metrics Comparison (Bar Chart)
    # -------------------------------------------------------------
    logger.info("Plotting Chart 7: Model Metrics Comparison (Bar Chart)...")
    plot_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 5.2))
    ax = sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    )
    plt.title("Model Performance Comparison (Test Set)", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Machine Learning Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_model_metrics_comparison_barchart.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # CHART 2: Confusion Matrix Elements Breakdown (Bar Chart)
    # -------------------------------------------------------------
    logger.info("Plotting Chart 8: Confusion Matrix Breakdown (Bar Chart)...")
    cm_rows = []
    for model_name, counts in cm_dict.items():
        for elem, val in counts.items():
            cm_rows.append({"Model": model_name, "Element": elem, "Count": val})
    cm_df = pd.DataFrame(cm_rows)

    plt.figure(figsize=(10, 5.2))
    sns.barplot(
        data=cm_df,
        x="Model",
        y="Count",
        hue="Element",
        palette={"TP": "#27ae60", "TN": "#2980b9", "FP": "#e74c3c", "FN": "#e67e22"}
    )
    plt.title("Confusion Matrix Breakdown by Model (TP, TN, FP, FN)", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Machine Learning Model", fontsize=11)
    plt.ylabel("Number of Predictions", fontsize=11)
    plt.legend(title="Prediction Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_confusion_matrix_breakdown_barchart.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # CHART 3: ROC Curves (Line Chart)
    # -------------------------------------------------------------
    logger.info("Plotting Chart 9: ROC Curves (Line Chart)...")
    plt.figure(figsize=(8, 6))
    for name in predictions.keys():
        y_prob = np.array(probabilities[name])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=1.8)

    plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline (AUC = 0.500)", linewidth=1.2)
    plt.title("ROC Curves Comparison Across 5 Models", fontsize=13, weight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_roc_curves_linechart.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # CHART 4: Feature Importance (Bar Chart from Decision Tree)
    # -------------------------------------------------------------
    logger.info("Plotting Chart 10: Feature Importance (Bar Chart)...")
    dt_model = all_models.get("Decision Tree")
    if dt_model and hasattr(dt_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": dt_model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        plt.figure(figsize=(9, 5.5))
        bars = plt.barh(fi_df["Feature"], fi_df["Importance"], color="#16a085", edgecolor="black", alpha=0.85)
        plt.title("Feature Importance from Decision Tree Classifier", fontsize=13, weight='bold', pad=12)
        plt.xlabel("Gini Importance Score", fontsize=11)
        plt.ylabel("Engineered Feature", fontsize=11)

        for bar in bars:
            w = bar.get_width()
            if w > 0.01:
                plt.text(w + 0.005, bar.get_y() + bar.get_height()/2., f"{w:.3f}",
                         va='center', ha='left', fontsize=8.5, weight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "10_feature_importances_barchart.png"), dpi=200)
        plt.close()

    # Save summary json
    summary_data = {
        "best_model": best_model_name,
        "metrics_table": metrics_df.to_dict(orient="records"),
        "confusion_matrices": cm_dict
    }
    with open(os.path.join(metadata_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Model evaluation complete. Figures saved in {output_dir}")
    return metrics_df


if __name__ == "__main__":
    evaluate_all_models()
