"""Module 7: predict fire / smoke for ONE image with the finalist model.
Used by app_gradio.py and app_tk.py. Runs on CPU.
Usage (quick test): python src/predict.py path\\to\\photo.jpg
Needs: models/thresholds.json (written by final_eval.py) and the model files it names.
"""
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models as tvm

from dataset import get_transforms
from features import extract          # hand-made features (only needed if the finalist includes the classical model)
from train import SimpleCNN

MODELS_DIR = Path("models")
MARGIN = 0.10   # probability this close to its threshold -> "needs human check"
DEVICE = torch.device("cpu")


def build_for_inference(name):
    if name == "simple_cnn":
        return SimpleCNN()
    if name == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
        return m
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    raise ValueError(name)


@lru_cache(maxsize=1)
def load_system():
    """Load thresholds and the models named in models/thresholds.json (done once)."""
    cfg = json.loads((MODELS_DIR / "thresholds.json").read_text())
    nets, clf = [], None
    for member in cfg["members"]:
        if member == "classical_best":
            clf = joblib.load(MODELS_DIR / "classical_best.joblib")
            continue
        ckpt = torch.load(MODELS_DIR / f"{member}.pt", map_location=DEVICE)
        net = build_for_inference(ckpt["model"])
        net.load_state_dict(ckpt["state"])
        net.eval()
        nets.append((net, get_transforms(train=False, size=ckpt["size"])))
    return cfg, nets, clf


def prepare(img):
    """Same steps as data preparation: upright, RGB, longest side <= 512, JPEG quality 90."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def decide(p_fire, p_smoke, thr_fire, thr_smoke, margin=MARGIN):
    """Small rule layer on top of the model's probabilities."""
    fire, smoke = p_fire >= thr_fire, p_smoke >= thr_smoke
    unsure = abs(p_fire - thr_fire) < margin or abs(p_smoke - thr_smoke) < margin
    if fire and smoke:
        label = "FIRE + SMOKE"
    elif fire:
        label = "FIRE"
    elif smoke:
        label = "SMOKE"
    else:
        label = "NONE DETECTED"
    return label, unsure


@torch.no_grad()
def predict(img):
    """img: a PIL image. Returns a dict with probabilities, decision and human-check flag."""
    cfg, nets, clf = load_system()
    img = prepare(img)
    probs = [torch.sigmoid(net(tf(img).unsqueeze(0))).squeeze(0).numpy() for net, tf in nets]
    if clf is not None:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        buf.seek(0)
        feats = np.array([extract(buf)], dtype=np.float32)
        probs.append(np.array([p[0, 1] for p in clf.predict_proba(feats)]))
    p = np.mean(probs, axis=0)
    decision, unsure = decide(float(p[0]), float(p[1]), cfg["fire"], cfg["smoke"])
    return {"p_fire": float(p[0]), "p_smoke": float(p[1]), "decision": decision,
            "needs_human_check": bool(unsure), "thr_fire": cfg["fire"], "thr_smoke": cfg["smoke"],
            "model": cfg["model"]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python src/predict.py path\\to\\photo.jpg")
    result = predict(Image.open(sys.argv[1]))
    for key, value in result.items():
        print(f"{key}: {value}")
