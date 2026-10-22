from __future__ import annotations
import sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` from the repository root without installing the package.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from medical_cnn.data import build_transforms, save_classes
from medical_cnn.model import RegularizedCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a regularized CNN on ImageFolder data.")
    parser.add_argument("--data", type=Path, required=True, help="Root containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    train_dir, val_dir = args.data / "train", args.data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        parser.error("Expected --data/train and --data/val directories")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(args.image_size, train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(args.image_size))
    if train_ds.classes != val_ds.classes:
        parser.error("Train and val must have the same class subfolders")
    if len(train_ds.classes) < 2:
        parser.error("At least two classes are required")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = RegularizedCNN(len(train_ds.classes), args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    args.output.mkdir(parents=True, exist_ok=True)
    save_classes(train_ds.classes, args.output / "classes.json")
    best_loss, stale_epochs, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "num_classes": len(train_ds.classes),
                        "image_size": args.image_size, "dropout": args.dropout,
                        "classes": train_ds.classes}, args.output / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {args.output / 'best_model.pt'}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
