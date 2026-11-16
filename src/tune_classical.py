"""Module 3a (part 2): tune the two best classical models with grouped CV.
Usage: python src/tune_classical.py
Outputs: outputs/classical_tuning.csv, models/classical_best.joblib
The TEST split is never used here.
"""
import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")

d = np.load("data/features.npz")
X, Y, split, cluster, stratum = d["X"], d["Y"], d["split"], d["cluster"], d["stratum"]
tr, va = split == "train", split == "val"
Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
             .split(Xtr, stratum[tr], groups=cluster[tr]))


def proba(model, X):
    return np.column_stack([p[:, 1] for p in model.predict_proba(X)])


def macro_f1(Y, P):
    return float(np.mean([f1_score(Y[:, i], (P[:, i] >= 0.5).astype(int)) for i in range(2)]))


SEARCH = {
    "RandomForest": (
        lambda **p: RandomForestClassifier(n_jobs=-1, random_state=42, **p),
        {"n_estimators": [400], "max_depth": [None, 15],
         "min_samples_leaf": [1, 3], "max_features": ["sqrt", 0.3]},
    ),
    "HistGradBoosting": (
        lambda **p: HistGradientBoostingClassifier(random_state=42, **p),
        {"learning_rate": [0.03, 0.06, 0.1], "max_iter": [200, 400], "max_depth": [None, 6]},
    ),
}

records, best = [], {}
for name, (factory, grid) in SEARCH.items():
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid, combo))
        cv = []
        for tr_i, te_i in folds:
            m = MultiOutputClassifier(factory(**params)).fit(Xtr[tr_i], Ytr[tr_i])
            cv.append(macro_f1(Ytr[te_i], proba(m, Xtr[te_i])))
        records.append({"model": name, "params": str(params),
                        "cv_macro_f1": round(np.mean(cv), 4), "cv_std": round(np.std(cv), 4)})
        print(f"{name} {params} -> CV {np.mean(cv):.3f} +/- {np.std(cv):.3f}")
        if name not in best or np.mean(cv) > best[name][0]:
            best[name] = (np.mean(cv), params)

pd.DataFrame(records).to_csv("outputs/classical_tuning.csv", index=False)

print("\n=== Best setting per model, refit on all training data, scored on validation ===")
final = None
for name, (cv_score, params) in best.items():
    model = MultiOutputClassifier(SEARCH[name][0](**params)).fit(Xtr, Ytr)
    P = proba(model, Xva)
    print(f"\n{name}: params={params}  (CV {cv_score:.3f})")
    print(f"  val macro F1 = {macro_f1(Yva, P):.3f}")
    for i, lab in enumerate(["fire", "smoke"]):
        pred = (P[:, i] >= 0.5).astype(int)
        print(f"  {lab}: AUC={roc_auc_score(Yva[:, i], P[:, i]):.3f} "
              f"F1={f1_score(Yva[:, i], pred):.3f} recall={recall_score(Yva[:, i], pred):.3f}")
    if final is None or cv_score > final[0]:
        final = (cv_score, name, model)

Path("models").mkdir(exist_ok=True)
joblib.dump(final[2], "models/classical_best.joblib")
print(f"\nSaved best classical model ({final[1]}) to models/classical_best.joblib")