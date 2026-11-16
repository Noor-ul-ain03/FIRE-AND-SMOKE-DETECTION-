"""Module 3a: handcrafted image features for the classical ML baselines.
Usage: python src/features.py        Output: data/features.npz
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SIZE = 128
BINS = 16


def extract(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = rgb.mean(axis=2)
    feats = []

    # 1) mean and spread of each colour channel (RGB + HSV)
    for ch in (rgb[..., 0], rgb[..., 1], rgb[..., 2], h, s, v):
        feats += [ch.mean(), ch.std()]

    # 2) colour histograms of hue, saturation, brightness
    for ch in (h, s, v):
        hist, _ = np.histogram(ch, bins=BINS, range=(0, 1))
        feats += (hist / hist.sum()).tolist()

    # 3) "fire-coloured" pixels: red/orange/yellow hue, vivid and bright
    fire_mask = ((h < 0.17) | (h > 0.94)) & (s > 0.5) & (v > 0.6)
    # "smoke-coloured" pixels: greyish (low saturation), mid brightness
    smoke_mask = (s < 0.2) & (v > 0.25) & (v < 0.85)
    feats += [fire_mask.mean(), smoke_mask.mean(), (v > 0.9).mean()]

    # 4) texture: smoke blurs edges, so edge strength helps
    feats += [np.abs(np.diff(gray, axis=1)).mean(),
              np.abs(np.diff(gray, axis=0)).mean(),
              gray.std()]

    # 5) where in the picture: fire/smoke fraction in each quarter
    half = SIZE // 2
    for mask in (fire_mask, smoke_mask):
        for rows in (slice(0, half), slice(half, SIZE)):
            for cols in (slice(0, half), slice(half, SIZE)):
                feats.append(mask[rows, cols].mean())
    return feats


if __name__ == "__main__":
    df = pd.read_csv("data/splits.csv")
    X = np.array([extract(f"data/images/{f}") for f in tqdm(df["file"], desc="features")],
                 dtype=np.float32)
    np.savez("data/features.npz",
             X=X,
             Y=df[["fire", "smoke"]].to_numpy(dtype=int),
             split=df["split"].to_numpy(dtype=str),
             cluster=df["cluster"].to_numpy(dtype=int),
             stratum=df["stratum"].to_numpy(dtype=int))
    print("saved data/features.npz  shape:", X.shape)