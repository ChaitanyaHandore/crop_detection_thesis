import os
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def clean(folder):
    for subdir, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(subdir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                print("Removing corrupt image:", path)
                os.remove(path)

if __name__ == "__main__":
    for split in ("train","val","test"):
        print(f"Cleaning {split}...")
        clean(os.path.join(ROOT, split))
    print("Done.")