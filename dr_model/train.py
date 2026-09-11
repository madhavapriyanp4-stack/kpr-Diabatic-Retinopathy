"""
Train the DR severity classifier.

Usage:
    python train.py --csv train.csv --img_dir train_images --epochs 20

Saves the checkpoint with the best validation Cohen's Kappa to
best_model.pt (Kappa is used instead of accuracy because the classes
are imbalanced -- see the class distribution note in dataset.py).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import APTOSDataset, get_transforms
from model import build_model


def compute_class_weights(dataset, num_classes=5):
    counts = dataset.df["diagnosis"].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=1)
    weights = 1.0 / counts.values
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / len(loader.dataset), kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=380)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_ds = APTOSDataset(args.csv, args.img_dir, size=args.size, transform=get_transforms(args.size, True))
    idx = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, stratify=full_ds.df["diagnosis"], random_state=42)

    val_ds = APTOSDataset(args.csv, args.img_dir, size=args.size, transform=get_transforms(args.size, False))
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=5, device=device)
    class_weights = compute_class_weights(full_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_kappa = -1.0
    for epoch in range(args.epochs):
        train_loss, train_kappa = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_kappa = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss {train_loss:.4f} kappa {train_kappa:.4f} "
              f"| val_loss {val_loss:.4f} kappa {val_kappa:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> saved new best model (kappa={best_kappa:.4f})")

    print(f"Training done. Best validation Kappa: {best_kappa:.4f}")


if __name__ == "__main__":
    main()
