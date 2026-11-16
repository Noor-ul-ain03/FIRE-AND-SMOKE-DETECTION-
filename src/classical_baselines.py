"""Module 3a: compare classical models with grouped cross-validation.
Usage: python src/classical_baselines.py     Output: outputs/classical_results.csv
The TEST split is never used here.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ["fire", "smoke"]

d = np.load("data/features.npz")
X, Y, split, cluster, stratum = d["X"], d["Y"], d["split"], d["cluster"], d["stratum"]
tr, va = split == "train", split == "val"
Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
groups_tr, strata_tr = cluster[tr], stratum[tr]


def make_models():
    base = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "SVM_RBF": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "HistGradBoosting": HistGradientBoostingClassifier(random_state=42),
    }
    # one copy of each model per label (fire, smoke)
    return {name: MultiOutputClassifier(m) for name, m in base.items()}


def proba(model, X):
    return np.column_stack([p[:, 1] for p in model.predict_proba(X)])


def scores(Y, P):
    out = {}
    for i, name in enumerate(LABELS):
        pred = (P[:, i] >= 0.5).astype(int)
        out[f"{name}_auc"] = roc_auc_score(Y[:, i], P[:, i])
        out[f"{name}_f1"] = f1_score(Y[:, i], pred)
        out[f"{name}_recall"] = recall_score(Y[:, i], pred)
    out["macro_f1"] = (out["fire_f1"] + out["smoke_f1"]) / 2
    return out


rows = []
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(Xtr, strata_tr, groups=groups_tr))

for name in make_models():
    t0 = time.time()
    fold_scores = []
    for tr_i, te_i in folds:
        model = make_models()[name]
        model.fit(Xtr[tr_i], Ytr[tr_i])
        fold_scores.append(scores(Ytr[te_i], proba(model, Xtr[te_i])))
    cv_f1 = [s["macro_f1"] for s in fold_scores]

    model = make_models()[name]
    model.fit(Xtr, Ytr)
    val = scores(Yva, proba(model, Xva))

    rows.append({"model": name,
                 "cv_macro_f1": f"{np.mean(cv_f1):.3f} +/- {np.std(cv_f1):.3f}",
                 **{f"val_{k}": round(v, 3) for k, v in val.items()}})
    print(f"done {name} in {time.time() - t0:.0f}s")

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("outputs/classical_results.csv", index=False)
print("\nsaved outputs/classical_results.csv")