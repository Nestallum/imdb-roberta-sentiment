"""Entry point for evaluating a fine-tuned RoBERTa checkpoint on IMDb test set.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best
"""

import argparse
from pathlib import Path

from transformers import RobertaForSequenceClassification

from src.data import build_dataloaders, load_imdb_splits, load_tokenizer, tokenize_dataset
from src.evaluation import compute_report, plot_confusion_matrix, predict
from src.utils import get_device, get_logger, load_config, set_seed


CLASS_NAMES = ["negative", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned RoBERTa on IMDb test set.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("evaluate")

    set_seed(cfg["training"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Loading tokenizer: {cfg['model']['name']}")
    tokenizer = load_tokenizer(cfg["model"]["name"])

    logger.info("Loading IMDb test set")
    raw = load_imdb_splits(seed=cfg["training"]["seed"])
    tokenized = tokenize_dataset(raw, tokenizer, max_length=cfg["data"]["max_length"])
    loaders = build_dataloaders(
        tokenized,
        tokenizer,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = RobertaForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    logger.info("Running inference on test set")
    preds, labels = predict(model, loaders["test"], device)

    report = compute_report(preds, labels, CLASS_NAMES)
    logger.info(f"\n{report}")

    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm_path = Path("docs/images/confusion_matrix.png")
    plot_confusion_matrix(preds, labels, CLASS_NAMES, cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    logger.info(f"Classification report saved to {results_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()