"""Training script for the road-surface-condition classifier.

Fine-tunes a pretrained CNN (transfer learning) and logs the run to MLflow.

Every run logs:
  * code version      : current git commit SHA
  * dataset version   : DVC md5 hash of the data directory (.dvc file)
  * hyperparameters   : backbone, learning rate, batch size, epochs, ...
  * metrics           : accuracy, macro-F1, per-class F1
  * artifacts         : best model weights (.pt) and confusion matrix (.png)

Usage:
  python ml/train.py --data-dir data/raw --experiment-name baseline
  python ml/train.py --data-dir data/processed --experiment-name improved
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
)

CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def get_dvc_md5(data_dir: str) -> str:
    """Read the md5 that DVC recorded for ``data_dir`` in its .dvc file."""
    dvc_file = Path(f"{data_dir}.dvc")
    if not dvc_file.exists():
        return "unknown"
    text = dvc_file.read_text(encoding="utf-8")
    m = re.search(r"md5:\s*([0-9a-f]+)", text)
    return m.group(1) if m else "unknown"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RoadSurfaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = default_loader(str(self.root / row["filename"]))
        if self.transform is not None:
            img = self.transform(img)
        label = CLASS2IDX[row["condition"]]
        return img, label


def make_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    """Split per-class so each split keeps the same label distribution."""
    train, val, test = [], [], []
    rng = np.random.default_rng(seed)
    for _, group in df.groupby("condition"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test.extend(idx[:n_test])
        val.extend(idx[n_test : n_test + n_val])
        train.extend(idx[n_test + n_val :])
    return (
        df.loc[train].reset_index(drop=True),
        df.loc[val].reset_index(drop=True),
        df.loc[test].reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {backbone}")
    return model


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class_f1 = f1_score(labels, preds, average=None, labels=list(range(len(CLASSES))))
    return acc, macro_f1, per_class_f1, labels, preds


def save_confusion_matrix(labels, preds, path: Path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="road-surface")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_dir = Path(args.data_dir)
    root = Path(".")

    # --- data ---
    df = pd.read_csv(args.manifest)
    df = df[df["filename"].isin(p.name for p in data_dir.glob("*.jpg"))]
    df = df[df["condition"].isin(CLASSES)].reset_index(drop=True)
    print(f"[data] {args.data_dir}: {len(df)} images")
    train_df, val_df, test_df = stratified_split(
        df, args.val_frac, args.test_frac, args.seed
    )
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = RoadSurfaceDataset(train_df, data_dir, make_transforms(args.img_size, True))
    val_ds = RoadSurfaceDataset(val_df, data_dir, make_transforms(args.img_size, False))
    test_ds = RoadSurfaceDataset(test_df, data_dir, make_transforms(args.img_size, False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # --- model ---
    model = build_model(args.backbone, len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_version = get_dvc_md5(args.data_dir)
    git_sha = get_git_sha()
    run_name = args.run_name or f"{args.backbone}-{args.data_dir.replace('/', '-')}-{int(time.time())}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "data_dir": args.data_dir,
                "data_version_dvc": data_version,
                "code_version_git": git_sha,
                "backbone": args.backbone,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "img_size": args.img_size,
                "seed": args.seed,
                "device": str(device),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            }
        )

        best_val_acc = 0.0
        best_state = None
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_acc, val_f1, per_f1, _, _ = evaluate(model, val_loader, device)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                },
                step=epoch,
            )
            print(
                f"[epoch {epoch}/{args.epochs}] "
                f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
                f"({time.time() - t0:.1f}s)"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # --- final evaluation on test set with the best weights ---
        model.load_state_dict(best_state)
        test_acc, test_f1, per_f1, labels, preds = evaluate(model, test_loader, device)
        mlflow.log_metrics({"test_acc": test_acc, "test_macro_f1": test_f1})
        for cls, f1 in zip(CLASSES, per_f1):
            mlflow.log_metric(f"test_f1_{cls}", float(f1))
        print(f"[test] acc={test_acc:.4f} macro_f1={test_f1:.4f}")

        # --- artifacts ---
        out_weights = Path(args.output_dir) / f"{run_name}_best.pt"
        torch.save(best_state, out_weights)
        cm_path = Path(args.output_dir) / f"{run_name}_confusion.png"
        save_confusion_matrix(labels, preds, cm_path)
        mlflow.log_artifact(str(out_weights), "model")
        mlflow.log_artifact(str(cm_path), "plots")
        for cls, f1 in zip(CLASSES, per_f1):
            print(f"    {cls:12s} test_f1={f1:.4f}")
        print(f"[artifacts] {out_weights}")
        print(f"[mlflow] run_id={run.info.run_id}")

    # persist the "which model is prod" pointer for the app
    prod_path = Path("ml/registry/current_model.md")
    prod_path.write_text(
        f"# Current production model\n\n"
        f"- run_id: {run.info.run_id}\n"
        f"- run_name: {run_name}\n"
        f"- backbone: {args.backbone}\n"
        f"- data_dir: {args.data_dir}\n"
        f"- data_version_dvc: {data_version}\n"
        f"- test_acc: {test_acc:.4f}\n"
        f"- test_macro_f1: {test_f1:.4f}\n"
        f"- weights: {out_weights}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
