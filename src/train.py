"""Module 3b/4: train a fire+smoke classifier (two yes/no outputs).
Examples
  python src/train.py --model simple_cnn --epochs 1 --max-batches 3 --run smoke_test
  python src/train.py --model simple_cnn --epochs 25 --run cnn_scratch
  python src/train.py --model resnet18 --epochs 15 --run resnet18
Never uses the TEST split.
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torchvision import models

from dataset import get_loaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """4 conv blocks (conv -> batchnorm -> ReLU -> pool) + global pooling + 2 outputs."""

    def __init__(self, n_out: int = 2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN()
    if name == "resnet18":
        m = models.resnet18(weights="DEFAULT")            # pretrained on ImageNet
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    total, probs, targets = 0.0, [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * len(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.cpu().numpy())
    P, Y = np.concatenate(probs), np.concatenate(targets)
    return total / len(P), P, Y


def metrics(Y, P):
    out = {}
    for i, name in enumerate(["fire", "smoke"]):
        pred = (P[:, i] >= 0.5).astype(int)
        try:
            out[f"{name}_auc"] = roc_auc_score(Y[:, i], P[:, i])
        except ValueError:                       # only happens on tiny smoke-test subsets
            out[f"{name}_auc"] = float("nan")
        out[f"{name}_f1"] = f1_score(Y[:, i], pred, zero_division=0)
        out[f"{name}_recall"] = recall_score(Y[:, i], pred, zero_division=0)
    out["mean_auc"] = float(np.nanmean([out["fire_auc"], out["smoke_auc"]]))
    out["macro_f1"] = (out["fire_f1"] + out["smoke_f1"]) / 2
    return out


def save_curves(history: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping curves")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["epoch"], history["train_loss"], label="train")
    ax[0].plot(history["epoch"], history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["epoch"], history["mean_auc"], label="val mean AUC")
    ax[1].plot(history["epoch"], history["macro_f1"], label="val macro F1")
    ax[1].set_title("Validation scores")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="simple_cnn", choices=["simple_cnn", "resnet18", "efficientnet_b0"])
    p.add_argument("--run", default=None, help="name used for saved files (default: model name)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    p.add_argument("--max-batches", type=int, default=0, help="quick test: only use N batches")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run = a.run or a.model

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    loaders = get_loaders(batch_size=a.batch, size=a.size, num_workers=a.workers)
    smoke_pos = int(loaders["train"].dataset.df["smoke"].sum())
    smoke_neg = len(loaders["train"].dataset) - smoke_pos
    pos_weight = torch.tensor([1.0, smoke_neg / smoke_pos], device=device)   # fire fixed at 1.0
    print("pos_weight (fire, smoke):", [round(v, 2) for v in pos_weight.tolist()])

    model = build_model(a.model).to(device)
    lr = a.lr or (1e-3 if a.model == "simple_cnn" else 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    history, best, bad = [], -1.0, 0

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for i, (x, y) in enumerate(loaders["train"]):
            if a.max_batches and i >= a.max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(x)
            seen += len(x)
        scheduler.step()

        val_loss, P, Y = evaluate(model, loaders["val"], criterion, device, a.max_batches)
        m = metrics(Y, P)
        row = {"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss, **m,
               "seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:02d} | train {row['train_loss']:.3f} | val {val_loss:.3f} | "
              f"AUC fire {m['fire_auc']:.3f} smoke {m['smoke_auc']:.3f} | macroF1 {m['macro_f1']:.3f} | "
              f"recall fire {m['fire_recall']:.2f} smoke {m['smoke_recall']:.2f} | {row['seconds']:.0f}s")

        score = 0.0 if np.isnan(m["mean_auc"]) else m["mean_auc"]
        if score > best:
            best, bad = score, 0
            torch.save({"model": a.model, "size": a.size, "state": model.state_dict()}, f"models/{run}.pt")
            np.save(f"outputs/{run}_val_probs.npy", P)
            np.save(f"outputs/{run}_val_targets.npy", Y)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stopping: no improvement for {a.patience} epochs")
                break

    hist = pd.DataFrame(history)
    hist.to_csv(f"outputs/{run}_history.csv", index=False)
    save_curves(hist, f"outputs/{run}_curves.png")
    print(f"\nbest validation mean AUC: {best:.3f}  -> models/{run}.pt")


if __name__ == "__main__":
    main()