"""Predict every image in a folder and save a table.
Usage: python src/predict_folder.py "test images"
Output: printed table and outputs/folder_predictions.csv
Only .jpg, .jpeg, .png and .bmp files are read; other files (for example .avif) are skipped and listed.
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from predict import predict

if len(sys.argv) < 2:
    sys.exit('Usage: python src/predict_folder.py "test images"')

folder = Path(sys.argv[1])
if not folder.is_dir():
    sys.exit(f"Folder not found: {folder.resolve()}")

rows, skipped = [], []
for f in sorted(folder.iterdir()):
    if not f.is_file():
        continue
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        skipped.append(f.name)
        continue
    try:
        r = predict(Image.open(f))
    except Exception as error:
        skipped.append(f"{f.name} ({error})")
        continue
    rows.append({"file": f.name, "p_fire": round(r["p_fire"], 3), "p_smoke": round(r["p_smoke"], 3),
                 "decision": r["decision"], "needs_human_check": r["needs_human_check"]})

if rows:
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    Path("outputs").mkdir(exist_ok=True)
    table.to_csv("outputs/folder_predictions.csv", index=False)
    print("\nsaved outputs/folder_predictions.csv")
if skipped:
    print("\nskipped:", skipped)
