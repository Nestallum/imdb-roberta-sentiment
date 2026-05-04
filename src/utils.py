"""utils.py — Utilities for reproducibility, device setup, logging, and config loading."""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Note: setting `torch.backends.cudnn.deterministic = True` is intentionally
    omitted here. It would slow down training significantly with limited
    benefit for a single-GPU fine-tuning run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str) -> logging.Logger:
    """Return a stdout logger with a standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)