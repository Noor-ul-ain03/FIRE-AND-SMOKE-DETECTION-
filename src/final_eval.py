"""Module 5: final evaluation.
The finalist and the thresholds are chosen on VALIDATION. Test labels are used only for reporting.
Usage: python src/final_eval.py
Needs: models/*.pt (from Colab), models/classical_best.joblib, data/features.npz, data/splits.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import models as tvm

from dataset import FireSmokeDataset, get_transforms
from train import SimpleCNN

FIRE_RECALL_TARGET = 0.98     # at least this share of real fires must be caught on validation
LABELS = ["fire", "smoke"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@torch.no_grad()
def deep_probs(run, split):
    ckpt_path = Path(f"models/{run}.pt")
    cache = Path(f"outputs/probs_{run}_{split}.npy")
    if cache.exists() and cache.stat().st_mtime > ckpt_path.stat().st_mtime:
        return np.load(cache)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_for_inference(ckpt["model"])
    model.load_state_dict(ckpt["state"])
    model.to(device).eval()
    ds = FireSmokeDataset(split, get_transforms(train=False, size=ckpt["size"]))
    out = [torch.sigmoid(model(x.to(device))).cpu().numpy()
           for x, _ in DataLoader(ds, batch_size=32, shuffle=False)]
    P = np.concatenate(out)
    np.save(cache, P)
    return P


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, i], P[:, i]) for i in range(2)]))


def tune_thresholds(Y, P):
    grid = np.linspace(0.05, 0.95, 91)
    feasible = []
    for t in grid:
        pred = (P[:, 0] >= t).astype(int)
        if recall_score(Y[:, 0], pred) >= FIRE_RECALL_TARGET:
            feasible.append((f1_score(Y[:, 0], pred), t))
    t_fire = max(feasible)[1] if feasible else float(grid[0])
    t_smoke = max(grid, key=lambda t: f1_score(Y[:, 1], (P[:, 1] >= t).astype(int)))
    return float(t_fire), float(t_smoke)


def report(Y, P, thr):
    out = {}
    for i, lab in enumerate(LABELS):
        pred = (P[:, i] >= thr[i]).astype(int)
        out[f"{lab}_auc"] = roc_auc_score(Y[:, i], P[:, i])
        out[f"{lab}_precision"] = precision_score(Y[:, i], pred, zero_division=0)
        out[f"{lab}_recall"] = recall_score(Y[:, i], pred, zero_division=0)
        out[f"{lab}_f1"] = f1_score(Y[:, i], pred, zero_division=0)
    out["macro_f1"] = (out["fire_f1"] + out["smoke_f1"]) / 2
    out["mean_auc"] = (out["fire_auc"] + out["smoke_auc"]) / 2
    return out


# ---------- load labels and classical model ----------
d = np.load("data/features.npz")
va, te = d["split"] == "val", d["split"] == "test"
Yva, Yte = d["Y"][va], d["Y"][te]
clf = joblib.load("models/classical_best.joblib")


def classical_probs(X):
    return np.column_stack([p[:, 1] for p in clf.predict_proba(X)])


runs = sorted(p.stem for p in Path("models").glob("*.pt") if p.stem != "smoke_test")
if not runs:
    raise SystemExit("No models/*.pt found. Download your Colab results into models/ first.")

val, test, members = {}, {}, {}
for r in runs:
    val[r], test[r], members[r] = deep_probs(r, "val"), deep_probs(r, "test"), [r]
    saved = Path(f"outputs/{r}_val_targets.npy")
    if saved.exists():
        assert (np.load(saved) == Yva).all(), f"{r}: validation order mismatch"
val["classical_best"] = classical_probs(d["X"][va])
test["classical_best"] = classical_probs(d["X"][te])
members["classical_best"] = ["classical_best"]

# ---------- ensembles (averages of probabilities) ----------
ranked = sorted(runs, key=lambda r: mean_auc(Yva, val[r]), reverse=True)
if len(ranked) >= 2:
    top2 = ranked[:2]
    for extra in ([], ["classical_best"]):
        group = top2 + extra
        name = "ens_" + "+".join(group)
        val[name] = np.mean([val[g] for g in group], axis=0)
        test[name] = np.mean([test[g] for g in group], axis=0)
        members[name] = group

# ---------- thresholds on validation, scores on test ----------
rows, thresholds = [], {}
for name in val:
    thr = tune_thresholds(Yva, val[name])
    thresholds[name] = thr
    rep = report(Yte, test[name], thr)
    rows.append({"model": name,
                 "val_mean_auc": round(mean_auc(Yva, val[name]), 3),
                 "test_mean_auc": round(rep["mean_auc"], 3),
                 "test_fire_auc": round(rep["fire_auc"], 3),
                 "test_smoke_auc": round(rep["smoke_auc"], 3),
                 "thr_fire": round(thr[0], 2), "thr_smoke": round(thr[1], 2),
                 "test_fire_recall": round(rep["fire_recall"], 3),
                 "test_fire_precision": round(rep["fire_precision"], 3),
                 "test_smoke_recall": round(rep["smoke_recall"], 3),
                 "test_smoke_precision": round(rep["smoke_precision"], 3),
                 "test_macro_f1": round(rep["macro_f1"], 3)})

res = pd.DataFrame(rows).sort_values("val_mean_auc", ascending=False)
print(res.to_string(index=False))
res.to_csv("outputs/final_results.csv", index=False)

# ---------- finalist = best on VALIDATION ----------
final = res.iloc[0]["model"]
thr = thresholds[final]
print(f"\nFinalist (chosen on validation): {final}")
print(f"Thresholds tuned on validation: fire={thr[0]:.2f}, smoke={thr[1]:.2f}")
for i, lab in enumerate(LABELS):
    pred = (test[final][:, i] >= thr[i]).astype(int)
    tn, fp, fn, tp = confusion_matrix(Yte[:, i], pred, labels=[0, 1]).ravel()
    print(f"TEST {lab}: TN={tn} FP={fp} FN={fn} TP={tp}  (missed {lab}: {fn} of {fn + tp})")

Path("models").mkdir(exist_ok=True)
with open("models/thresholds.json", "w") as f:
    json.dump({"model": final, "members": members[final],
               "fire": thr[0], "smoke": thr[1]}, f, indent=2)
np.save("outputs/final_test_probs.npy", test[final])
print("\nsaved outputs/final_results.csv, models/thresholds.json")
