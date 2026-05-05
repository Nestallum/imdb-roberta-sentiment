"""Evaluation utilities: predictions, metrics, and confusion matrix plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel


@torch.no_grad()
def predict(
    model: PreTrainedModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, ground_truth) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_report(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> str:
    """Return a formatted classification report (precision, recall, F1)."""
    return classification_report(labels, preds, target_names=class_names, digits=4)


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot and save a confusion matrix as a PNG image."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)