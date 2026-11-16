"""Module 1 / Step B3: find duplicate and near-duplicate images.
Usage: python src/check_duplicates.py "archive (1).zip" "archive (2).zip"
"""
import io
import sys
import zipfile
from pathlib import PurePosixPath

import imagehash
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
THRESH = 4  # max differing bits (of 64) to call two images near-identical


def hash_zip(zip_path: str):
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.endswith("/") or not n.lower().endswith(IMG_EXT):
                continue
            split = PurePosixPath(n).parts[0]
            try:
                img = Image.open(io.BytesIO(z.read(n)))
                img.draft("RGB", (256, 256))
                h = int(str(imagehash.phash(img.convert("RGB"))), 16)
            except Exception as e:
                print("unreadable:", n, e)
                continue
            rows.append((zip_path, split, n, h))
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/check_duplicates.py "a.zip" "b.zip"')
    rows = []
    for p in sys.argv[1:]:
        print("hashing", p, "...")
        rows += hash_zip(p)
    print("total images hashed:", len(rows))

    same_split, cross_split, cross_zip = [], [], []
    for i in range(len(rows)):
        zi, si, ni, hi = rows[i]
        for j in range(i + 1, len(rows)):
            zj, sj, nj, hj = rows[j]
            if bin(hi ^ hj).count("1") <= THRESH:
                pair = (ni, nj)
                if zi != zj:
                    cross_zip.append(pair)
                elif si != sj:
                    cross_split.append(pair)
                else:
                    same_split.append(pair)

    for title, pairs in [("Near-duplicates inside the same split", same_split),
                         ("Near-duplicates ACROSS splits (leakage)", cross_split),
                         ("Matches BETWEEN the two zips", cross_zip)]:
        print(f"\n{title}: {len(pairs)}")
        for a, b in pairs[:8]:
            print("   ", a, "<->", b)