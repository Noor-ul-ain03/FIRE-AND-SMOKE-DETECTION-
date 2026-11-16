"""Module 6: figures and a small decision layer for the report. CPU only, reads saved results.
Usage: python src/make_figures.py
Needs (all produced earlier): outputs/final_results.csv, outputs/final_test_probs.npy,
       models/thresholds.json, data/splits.csv, data/images/, outputs/*_history.csv
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)
MARGIN = 0.10   # a probability this close to its threshold -> "needs human check"


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


def label_text(fire, smoke):
    return "+".join(n for n, v in (("fire", fire), ("smoke", smoke)) if v) or "neither"


# ---------- load saved results ----------
res = pd.read_csv("outputs/final_results.csv")
cfg = json.load(open("models/thresholds.json"))
thr_f, thr_s = cfg["fire"], cfg["smoke"]
df = pd.read_csv("data/splits.csv")
test = df[df["split"] == "test"].reset_index(drop=True)
P = np.load("outputs/final_test_probs.npy")
assert len(P) == len(test), "probability rows do not match the test split"
Y = test[["fire", "smoke"]].to_numpy().astype(int)
pred_f, pred_s = P[:, 0] >= thr_f, P[:, 1] >= thr_s
wrong = (pred_f != Y[:, 0].astype(bool)) | (pred_s != Y[:, 1].astype(bool))

# ---------- figure 1: all models on the test set ----------
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(res)), 0.38
ax.bar(x - w / 2, res["test_mean_auc"], w, label="test mean AUC")
ax.bar(x + w / 2, res["test_macro_f1"], w, label="test macro F1")
ax.set_xticks(x)
ax.set_xticklabels(res["model"], rotation=25, ha="right", fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.set_title("All models on the test set (thresholds tuned on validation)")
fig.tight_layout()
fig.savefig(FIG / "fig1_model_comparison.png", dpi=150)
plt.close(fig)

# ---------- figure 2: confusion matrices of the finalist ----------
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for i, lab in enumerate(["fire", "smoke"]):
    pred = (P[:, i] >= (thr_f, thr_s)[i]).astype(int)
    cm = confusion_matrix(Y[:, i], pred, labels=[0, 1])
    axes[i].imshow(cm, cmap="Blues")
    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, cm[r, c], ha="center", va="center", fontsize=16,
                         color="white" if cm[r, c] > cm.max() / 2 else "black")
    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels([f"no {lab}", lab])
    axes[i].set_yticklabels([f"no {lab}", lab])
    axes[i].set_xlabel("predicted")
    axes[i].set_ylabel("actual")
    axes[i].set_title(f"{lab} (test set)")
fig.suptitle(f"Finalist: {cfg['model']}", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_confusion_matrices.png", dpi=150)
plt.close(fig)

# ---------- figure 3: training curves of the deep runs ----------
fig, ax = plt.subplots(figsize=(7, 4))
for f in sorted(Path("outputs").glob("*_history.csv")):
    run = f.name.replace("_history.csv", "")
    if run == "smoke_test":
        continue
    h = pd.read_csv(f)
    ax.plot(h["epoch"], h["mean_auc"], marker="o", label=run)
ax.set_xlabel("epoch")
ax.set_ylabel("validation mean AUC")
ax.set_title("Validation AUC per epoch")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig3_training_curves.png", dpi=150)
plt.close(fig)


# ---------- figures 4 and 5: prediction grids ----------
def grid(idx, path, title):
    if len(idx) == 0:
        return
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.9))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        ax.imshow(Image.open(f"data/images/{test.loc[i, 'file']}").convert("RGB"))
        decision, unsure = decide(P[i, 0], P[i, 1], thr_f, thr_s)
        ax.set_title(f"actual: {label_text(Y[i, 0], Y[i, 1])}\n"
                     f"fire {P[i, 0]:.2f} | smoke {P[i, 1]:.2f}\n"
                     f"{decision}{' (HUMAN CHECK)' if unsure else ''}",
                     fontsize=8, color="red" if wrong[i] else "green")
    fig.suptitle(title)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


rng = np.random.default_rng(0)
grid(rng.choice(len(test), size=min(12, len(test)), replace=False),
     FIG / "fig4_random_test_predictions.png",
     "Random test photos (green = all answers correct, red = at least one wrong)")
grid(np.where(wrong)[0][:12], FIG / "fig5_mistakes.png",
     f"Mistakes on the test set ({int(wrong.sum())} of {len(test)} images have a wrong answer)")

# ---------- per-image decisions and a check of the human-check rule ----------
rows, flagged = [], []
for i in range(len(test)):
    decision, unsure = decide(P[i, 0], P[i, 1], thr_f, thr_s)
    flagged.append(unsure)
    rows.append({"file": test.loc[i, "file"], "actual_fire": Y[i, 0], "actual_smoke": Y[i, 1],
                 "p_fire": round(float(P[i, 0]), 3), "p_smoke": round(float(P[i, 1]), 3),
                 "decision": decision, "needs_human_check": unsure})
pd.DataFrame(rows).to_csv("outputs/test_decisions.csv", index=False)
flagged = np.array(flagged)
print(f"Test images: {len(test)} | with a wrong answer: {int(wrong.sum())}")
print(f"Flagged 'needs human check': {int(flagged.sum())} ({flagged.mean():.0%})")
if flagged.any() and (~flagged).any():
    print(f"  error rate among flagged images:   {wrong[flagged].mean():.0%}")
    print(f"  error rate among unflagged images: {wrong[~flagged].mean():.0%}")
print("Figures saved in outputs/figures/, decisions in outputs/test_decisions.csv")
