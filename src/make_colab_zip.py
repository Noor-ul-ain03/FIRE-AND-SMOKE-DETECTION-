"""Pack code + data for Google Colab. Usage: python src/make_colab_zip.py"""
import zipfile
from pathlib import Path

out = "fire_project.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in Path("src").glob("*.py"):
        z.write(f, f.as_posix())
    z.write("data/splits.csv", "data/splits.csv")
    for f in Path("data/images").glob("*.jpg"):
        z.write(f, f.as_posix())
print(out, f"{Path(out).stat().st_size / 1e6:.0f} MB")