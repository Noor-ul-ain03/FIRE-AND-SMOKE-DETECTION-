"""Module 1 (final): build the master manifest from the YOLO zip.
Usage: python src/build_manifest.py "archive (1).zip"
Outputs: data/manifest.csv, data/excluded.csv, data/images/*.jpg
"""
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import imagehash
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MAX_SIDE = 512      # longest side of saved images
HASH_THRESH = 4     # max differing bits (of 64) to treat two images as near-duplicates
OUT_DIR = Path("data")
IMG_DIR = OUT_DIR / "images"


def count_boxes(text: str):
    """Return (n_fire, n_smoke) from a YOLO label file: class 0=fire, 1=smoke."""
    n_fire = n_smoke = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        if cid == 0:
            n_fire += 1
        elif cid == 1:
            n_smoke += 1
    return n_fire, n_smoke


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def main(zip_path: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        labels = {}
        for n in names:
            p = PurePosixPath(n)
            if n.lower().endswith(".txt") and len(p.parts) >= 2:
                labels[(p.parts[0], p.stem)] = n   # works for 'labels' and 'label' folders

        image_names = [n for n in names if n.lower().endswith(IMG_EXT)]
        for n in tqdm(image_names, desc="processing images"):
            p = PurePosixPath(n)
            split, stem = p.parts[0], p.stem

            label_path = labels.get((split, stem))
            if label_path is None:
                excluded.append((n, "no_label_file"))
                continue
            n_fire, n_smoke = count_boxes(z.read(label_path).decode("utf-8", errors="replace"))
            if n_fire + n_smoke == 0:
                excluded.append((n, "no_boxes"))
                continue

            try:
                img = Image.open(io.BytesIO(z.read(n)))
                img.load()                                  # raises on truncated files
                img = ImageOps.exif_transpose(img).convert("RGB")
            except Exception as e:
                excluded.append((n, f"unreadable: {e}"))
                continue

            orig_w, orig_h = img.size
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            out_name = f"{split}_{stem}.jpg"
            img.save(IMG_DIR / out_name, quality=90)

            rows.append({
                "file": out_name,
                "orig_split": split,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "fire": int(n_fire > 0),
                "smoke": int(n_smoke > 0),
                "n_fire_boxes": n_fire,
                "n_smoke_boxes": n_smoke,
                "phash": int(str(imagehash.phash(img)), 16),
            })

    df = pd.DataFrame(rows)

    # --- group near-duplicate images into clusters (union-find) ---
    hashes = df["phash"].tolist()
    parent = list(range(len(hashes)))
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if (hashes[i] ^ hashes[j]).bit_count() <= HASH_THRESH:
                ri, rj = find(parent, i), find(parent, j)
                if ri != rj:
                    parent[rj] = ri
    df["cluster"] = [find(parent, i) for i in range(len(hashes))]
    df["phash"] = df["phash"].apply(lambda h: f"{h:016x}")

    df.to_csv(OUT_DIR / "manifest.csv", index=False)
    pd.DataFrame(excluded, columns=["file", "reason"]).to_csv(OUT_DIR / "excluded.csv", index=False)

    # --- summary ---
    print(f"\nKept images: {len(df)} | excluded: {len(excluded)}")
    for f, reason in excluded:
        print("   excluded:", f, "->", reason[:60])
    combo = df["fire"] + 2 * df["smoke"]
    print("\nLabel mix:", dict(combo.map({1: "fire only", 2: "smoke only", 3: "both"}).value_counts()))
    sizes = df["cluster"].value_counts()
    print(f"\nClusters: {sizes.size} | singletons: {(sizes == 1).sum()} | largest: {sizes.max()}")
    print("Cluster-size distribution:", dict(Counter(sizes.tolist()).most_common(8)))
    print("\nLargest clusters (open these in data/images to spot-check):")
    for cid in sizes.head(3).index:
        print("  cluster", cid, "->", df.loc[df.cluster == cid, "file"].head(6).tolist())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/build_manifest.py "archive (1).zip"')
    main(sys.argv[1])