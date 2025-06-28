import os
import random
import shutil
import csv
from PIL import Image

RAW_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'Raw Data', 'CCMT Dataset'
)
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
SAMPLES_DIR = os.path.join(DATA_DIR, 'samples')
COUNT_CSV   = os.path.join(DATA_DIR, 'class_counts.csv')

# Prepare samples folder
if os.path.exists(SAMPLES_DIR):
    shutil.rmtree(SAMPLES_DIR)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Iterate crops → diseases → count & sample
with open(COUNT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['crop', 'disease', 'count'])

    for crop in sorted(os.listdir(RAW_DIR)):
        crop_path = os.path.join(RAW_DIR, crop)
        if not os.path.isdir(crop_path):
            continue

        for disease in sorted(os.listdir(crop_path)):
            disease_path = os.path.join(crop_path, disease)
            if not os.path.isdir(disease_path):
                continue

            # List valid image files
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            images = [f for f in os.listdir(disease_path)
                      if f.lower().endswith(valid_exts)]
            count = len(images)
            writer.writerow([crop, disease, count])

            # Sample up to 2 images for thumbnails
            for img_name in random.sample(images, min(2, count)):
                src = os.path.join(disease_path, img_name)
                try:
                    img = Image.open(src)
                    img.thumbnail((256, 256))
                    save_name = f"{crop}___{disease}___{os.path.splitext(img_name)[0]}.png"
                    img.save(os.path.join(SAMPLES_DIR, save_name))
                except Exception as e:
                    print(f"Failed {src}: {e}")

print("✅ Done!")
print(f"• Counts written to: {COUNT_CSV}")
print(f"• Thumbnails in:    {SAMPLES_DIR}")