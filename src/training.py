"""Training and validation loops for sequence classification."""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup


def train_one_epoch(
    model: PreTrainedModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> int:
    """Run one training epoch and return the updated global step."""
    model.train()
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for batch in progress:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        global_step += 1

    return global_step


@torch.no_grad()
def validate(
    model: PreTrainedModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (mean loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item() * batch["labels"].size(0)
        preds = outputs.logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_examples += batch["labels"].size(0)

    return total_loss / total_examples, total_correct / total_examples


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    log_dir: Path,
    checkpoint_dir: Path,
) -> None:
    """Fine-tune the model and save the best checkpoint based on val accuracy."""
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    writer = SummaryWriter(log_dir=str(log_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            max_grad_norm, epoch, writer, global_step,
        )
        val_loss, val_acc = validate(model, val_loader, device)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        print(f"Epoch {epoch} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(checkpoint_dir / "best")
            print(f"  ✓ new best (val_acc={val_acc:.4f}), saved to {checkpoint_dir / 'best'}")

    writer.close()
    print(f"Training complete. Best val accuracy: {best_acc:.4f}")