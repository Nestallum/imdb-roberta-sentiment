"""Data loading and tokenization for the IMDb sentiment classification task."""

from typing import Any

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the pre-trained tokenizer associated with the given model."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def load_imdb_splits(val_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """Load IMDb and create a train/validation/test split.

    The original dataset has only train (25k) and test (25k). We carve a
    validation set out of the train split to monitor overfitting honestly,
    keeping the test set untouched until final evaluation.
    """
    raw = load_dataset("imdb")
    split = raw["train"].train_test_split(test_size=val_ratio, seed=seed, stratify_by_column="label")
    return DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    )


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    """Tokenize all splits and return a DatasetDict of tensor-ready examples."""

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding handled by the data collator
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def build_dataloaders(
    tokenized: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders with dynamic padding via DataCollator."""
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    return {
        "train": DataLoader(
            tokenized["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        ),
        "validation": DataLoader(
            tokenized["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        ),
        "test": DataLoader(
            tokenized["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        ),
    }