from __future__ import annotations
import sys
from pathlib import Path as _Path

# Allow `python scripts/<name>.py` from the repository root without installing the package.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
from medical_cnn.data import build_transforms, load_checkpoint
from medical_cnn.model import RegularizedCNN


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN on held-out ImageFolder data.")
    parser.add_argument("--data", type=Path, required=True, help="Test directory with class subfolders")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.data.is_dir() or not args.checkpoint.is_file():
        parser.error("Test data directory and checkpoint must both exist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    classes = checkpoint["classes"]
    ds = datasets.ImageFolder(args.data, transform=build_transforms(checkpoint["image_size"]))
    if ds.classes != classes:
        parser.error(f"Test classes {ds.classes} do not match checkpoint classes {classes}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = RegularizedCNN(checkpoint["num_classes"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            y_pred.extend(model(images.to(device)).argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred).tolist())


if __name__ == "__main__":
    main()
