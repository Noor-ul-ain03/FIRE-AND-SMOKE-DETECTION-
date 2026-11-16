"""Module 1: visual spot-check of duplicate clusters.
Usage: python src/preview_clusters.py     Output: data/cluster_check.png
"""
import pandas as pd
from PIL import Image, ImageDraw

df = pd.read_csv("data/manifest.csv")
sizes = df["cluster"].value_counts()
rows = sizes.head(3).index.tolist() + sizes[sizes == 2].sample(3, random_state=0).index.tolist()

THUMB, MAX_COLS, LABEL_H = 160, 8, 18
canvas = Image.new("RGB", (MAX_COLS * THUMB, len(rows) * (THUMB + LABEL_H)), "white")
draw = ImageDraw.Draw(canvas)

for r, cid in enumerate(rows):
    files = df.loc[df["cluster"] == cid, "file"].tolist()[:MAX_COLS]
    y = r * (THUMB + LABEL_H)
    draw.text((4, y + 2), f"cluster {cid} (n={sizes[cid]})", fill="black")
    for c, f in enumerate(files):
        img = Image.open(f"data/images/{f}").convert("RGB")
        img.thumbnail((THUMB, THUMB))
        canvas.paste(img, (c * THUMB, y + LABEL_H))

canvas.save("data/cluster_check.png")
print("saved data/cluster_check.png")