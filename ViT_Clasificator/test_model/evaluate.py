import numpy as np
import csv
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)


def evaluate(y_bin, y_bin_hat, y_bin_proba, y_class, y_class_hat, y_class_proba):
    report_class = classification_report(y_class, y_class_hat, output_dict=True, zero_division=0)
    cm_class = confusion_matrix(y_class, y_class_hat)
    bal_acc_class = balanced_accuracy_score(y_class, y_class_hat)

    n_labels = y_class_proba.shape[1]
    try:
        auc_class_ovr = roc_auc_score(y_class, y_class_proba, multi_class="ovr", average="macro",
                                       labels=list(range(n_labels)))
    except ValueError:
        auc_class_ovr = float("nan")

    f1_macro = f1_score(y_class, y_class_hat, average="macro", zero_division=0)
    f1_weighted = f1_score(y_class, y_class_hat, average="weighted", zero_division=0)
    recall_macro = recall_score(y_class, y_class_hat, average="macro", zero_division=0)
    recall_weighted = recall_score(y_class, y_class_hat, average="weighted", zero_division=0)
    precision_macro = precision_score(y_class, y_class_hat, average="macro", zero_division=0)
    precision_weighted = precision_score(y_class, y_class_hat, average="weighted", zero_division=0)

    metrics_class = {
        "balanced_accuracy": float(bal_acc_class),
        "auc_roc_ovr": float(auc_class_ovr),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
    }

    metrics_bin = {}
    report_bin = {}
    cm_bin = np.array([])
    if len(y_bin) > 0:
        report_bin = classification_report(y_bin, y_bin_hat, output_dict=True, zero_division=0)
        cm_bin = confusion_matrix(y_bin, y_bin_hat)
        bal_acc_bin = balanced_accuracy_score(y_bin, y_bin_hat)
        try:
            auc_bin = roc_auc_score(y_bin, y_bin_proba)
        except ValueError:
            auc_bin = float("nan")
        f1_bin = f1_score(y_bin, y_bin_hat, average="binary", zero_division=0)
        recall_bin = recall_score(y_bin, y_bin_hat, average="binary", zero_division=0)
        precision_bin = precision_score(y_bin, y_bin_hat, average="binary", zero_division=0)
        metrics_bin = {
            "balanced_accuracy": float(bal_acc_bin),
            "auc_roc": float(auc_bin),
            "f1": float(f1_bin),
            "recall": float(recall_bin),
            "precision": float(precision_bin),
        }

    return {
        "report_class": report_class,
        "cm_class": cm_class,
        "metrics_class": metrics_class,
        "report_bin": report_bin,
        "cm_bin": cm_bin,
        "metrics_bin": metrics_bin,
    }


def save_results(results, eval_metrics, save_dir, run_id=0, class_names=None, bin_names=None):
    os.makedirs(save_dir, exist_ok=True)
    row = results.copy()
    row["run"] = run_id
    for k, v in eval_metrics["metrics_class"].items():
        row[f"class_{k}"] = v
    for k, v in eval_metrics["metrics_bin"].items():
        row[f"bin_{k}"] = v
    csv_path = os.path.join(save_dir, "test_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    json_path = os.path.join(save_dir, f"report_run_{run_id}.json")
    with open(json_path, "w") as f:
        json.dump({
            "class_report": eval_metrics["report_class"],
            "bin_report": eval_metrics["report_bin"],
            "metrics_class": eval_metrics["metrics_class"],
            "metrics_bin": eval_metrics["metrics_bin"],
        }, f, indent=2, default=str)

    cm_class = eval_metrics["cm_class"]
    np.save(os.path.join(save_dir, f"cm_class_run_{run_id}.npy"), cm_class)
    plot_confusion_matrix(
        cm_class,
        labels=class_names,
        title=f"Confusion Matrix — Multiclass (run {run_id})",
        save_path=os.path.join(save_dir, f"cm_class_run_{run_id}.png"),
    )

    cm_bin = eval_metrics["cm_bin"]
    if cm_bin.size > 0:
        np.save(os.path.join(save_dir, f"cm_bin_run_{run_id}.npy"), cm_bin)
        plot_confusion_matrix(
            cm_bin,
            labels=bin_names or ["Benigno", "Maligno"],
            title=f"Confusion Matrix — Binaria (run {run_id})",
            save_path=os.path.join(save_dir, f"cm_bin_run_{run_id}.png"),
        )


def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path=None, figsize=None, normalize=False):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    n = cm.shape[0]
    if figsize is None:
        size = max(6, n * 0.8)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels if labels is not None else range(n),
        yticklabels=labels if labels is not None else range(n),
        linewidths=0.5, linecolor="gray",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
