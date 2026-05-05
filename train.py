"""Entry point for fine-tuning RoBERTa on IMDb sentiment classification.

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
from pathlib import Path

from src.data import build_dataloaders, load_imdb_splits, load_tokenizer, tokenize_dataset
from src.model import build_model
from src.training import train
from src.utils import get_device, get_logger, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on IMDb.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("train")

    set_seed(cfg["training"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Loading tokenizer: {cfg['model']['name']}")
    tokenizer = load_tokenizer(cfg["model"]["name"])

    logger.info("Loading and splitting IMDb dataset")
    raw = load_imdb_splits(seed=cfg["training"]["seed"])
    logger.info(f"Splits: { {k: len(v) for k, v in raw.items()} }")

    logger.info("Tokenizing dataset")
    tokenized = tokenize_dataset(raw, tokenizer, max_length=cfg["data"]["max_length"])

    loaders = build_dataloaders(
        tokenized,
        tokenizer,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    logger.info(f"Building model: {cfg['model']['name']}")
    model = build_model(cfg["model"]["name"], num_labels=cfg["data"]["num_labels"])

    logger.info("Starting training")
    train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        device=device,
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        log_dir=Path(cfg["logging"]["log_dir"]),
        checkpoint_dir=Path(cfg["logging"]["checkpoint_dir"]),
    )


if __name__ == "__main__":
    main()