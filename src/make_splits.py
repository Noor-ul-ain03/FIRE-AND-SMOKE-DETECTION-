"""Module 2 (part 1): leakage-free, stratified train/val/test split.
Usage: python src/make_splits.py
Input: data/manifest.csv   Output: data/splits.csv
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("data/manifest.csv")
df["stratum"] = df["fire"] + 2 * df["smoke"]      # 1=fire only, 2=smoke only, 3=both

# 7 folds: fold 0 -> test, fold 1 -> val, the other 5 -> train  (~71/14/14)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
folds = [test_idx for _, test_idx in sgkf.split(df, df["stratum"], groups=df["cluster"])]

df["split"] = "train"
df.loc[folds[0], "split"] = "test"
df.loc[folds[1], "split"] = "val"

# safety check: no duplicate cluster may appear in more than one split
assert df.groupby("cluster")["split"].nunique().max() == 1, "leakage: a cluster spans splits"

names = {1: "fire only", 2: "smoke only", 3: "both"}
print("Images per split:\n", df["split"].value_counts(), "\n")
print("Counts by split and label mix:\n", pd.crosstab(df["split"], df["stratum"].map(names)), "\n")
print("Share by split:\n", pd.crosstab(df["split"], df["stratum"].map(names), normalize="index").round(3), "\n")

train = df[df["split"] == "train"]
for col in ["fire", "smoke"]:
    pos = int(train[col].sum())
    neg = len(train) - pos
    print(f"{col}: positives={pos}, negatives={neg}, pos_weight (neg/pos)={neg / pos:.2f}")

df.to_csv("data/splits.csv", index=False)
print("\nSaved data/splits.csv")