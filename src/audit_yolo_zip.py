"""Module 1 / Step B2: audit a YOLO-format dataset inside a zip.
Usage: python src/audit_yolo_zip.py "archive (1).zip"
"""
import io
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def audit(zip_path: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]

        for n in names:
            if n.lower().endswith((".yaml", ".yml")):
                print(f"--- {n} ---")
                print(z.read(n).decode("utf-8", errors="replace"))

        images, labels = defaultdict(dict), defaultdict(dict)
        for n in names:
            p = PurePosixPath(n)
            if len(p.parts) < 2:
                continue
            split = p.parts[0]
            if n.lower().endswith(IMG_EXT):
                images[split][p.stem] = n
            elif n.lower().endswith(".txt"):
                labels[split][p.stem] = n

        for split in sorted(set(images) | set(labels)):
            imgs, labs = images[split], labels[split]
            print("=" * 60)
            print(f"SPLIT: {split} | images={len(imgs)} | label files={len(labs)}")
            print("  images without label:", sorted(set(imgs) - set(labs))[:5] or "none")
            print("  labels without image:", sorted(set(labs) - set(imgs))[:5] or "none")

            boxes_per_class = Counter()
            areas = defaultdict(list)
            image_sets = Counter()
            odd_lines = 0
            for stem, path in labs.items():
                classes = set()
                text = z.read(path).decode("utf-8", errors="replace")
                for line in text.splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        odd_lines += 1
                        continue
                    if len(parts) != 5:
                        odd_lines += 1  # polygon or malformed line
                        continue
                    classes.add(cid)
                    boxes_per_class[cid] += 1
                    areas[cid].append(float(parts[3]) * float(parts[4]))
                key = "+".join(str(c) for c in sorted(classes)) or "(empty)"
                image_sets[key] += 1

            print("  boxes per class id:", dict(sorted(boxes_per_class.items())))
            print("  median box area (fraction of image):",
                  {c: round(statistics.median(a), 3) for c, a in sorted(areas.items())})
            print("  images by class-id set:", dict(image_sets))
            print("  non-box / malformed lines:", odd_lines)

            sizes, bad = Counter(), []
            for stem, path in imgs.items():
                try:
                    img = Image.open(io.BytesIO(z.read(path)))
                    sizes[img.size] += 1
                    img.verify()
                except Exception:
                    bad.append(path)
            print("  most common image sizes:", sizes.most_common(4))
            print("  unreadable images:", bad[:5] or "none", f"(total {len(bad)})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/audit_yolo_zip.py "archive (1).zip"')
    for p in sys.argv[1:]:
        audit(p)