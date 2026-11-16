import sys
import zipfile
from collections import Counter

def explore_zip(zip_path):
    print("=" * 60)
    print("ZIP FILE:", zip_path)
    print("=" * 60)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if not n.endswith("/")]  # files only

        print("Total files:", len(names))

        # Count files per folder
        folder_counts = Counter()
        for n in names:
            parts = n.split("/")
            folder = "/".join(parts[:-1]) if len(parts) > 1 else "(root)"
            folder_counts[folder] += 1

        print("\nFiles per folder:")
        for folder, count in sorted(folder_counts.items()):
            print("  ", folder, "->", count)

        # Count file types
        ext_counts = Counter()
        for n in names:
            if "." in n.split("/")[-1]:
                ext_counts[n.rsplit(".", 1)[-1].lower()] += 1
            else:
                ext_counts["(no extension)"] += 1

        print("\nFile types:")
        for ext, count in ext_counts.most_common():
            print("  ", ext, "->", count)

        print("\nFirst 5 files:")
        for n in names[:5]:
            print("  ", n)
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/explore_structure.py "file1.zip" "file2.zip"')
        sys.exit()
    for path in sys.argv[1:]:
        explore_zip(path)