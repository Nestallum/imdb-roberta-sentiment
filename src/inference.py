"""Inference utilities for predicting sentiment on raw text."""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def predict_sentiment(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int,
    class_names: list[str],
) -> tuple[str, float]:
    """Predict the sentiment of a single text.

    Returns:
        A tuple (predicted_class_name, confidence) where confidence is the
        softmax probability of the predicted class.
    """
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    ).to(device)

    logits = model(**encoding).logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)
    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())

    return class_names[pred_idx], confidence