"""Entry point for predicting sentiment on raw text using a fine-tuned RoBERTa.

Usage:
    python predict.py --text "This movie was incredible."
    python predict.py --text "Worst film ever." --checkpoint checkpoints/best
"""

import argparse
from pathlib import Path

from transformers import RobertaForSequenceClassification

from src.data import load_tokenizer
from src.inference import predict_sentiment
from src.utils import get_device, load_config


CLASS_NAMES = ["negative", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment on a piece of text.")
    parser.add_argument("--text", type=str, required=True, help="Text to classify.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = RobertaForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    label, confidence = predict_sentiment(
        text=args.text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=cfg["data"]["max_length"],
        class_names=CLASS_NAMES,
    )

    print(f"Prediction: {label} ({confidence:.4f})")


if __name__ == "__main__":
    main()