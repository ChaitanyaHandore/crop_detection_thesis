import os, random, shutil

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, '..', 'data', 'Raw Data', 'CCMT Dataset')
PROC_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

# Create processed directories
for split in ('train', 'val', 'test'):
    for crop in os.listdir(RAW_DIR):
        crop_dir = os.path.join(RAW_DIR, crop)
        if not os.path.isdir(crop_dir):
            continue
        for disease in os.listdir(crop_dir):
            disease_dir = os.path.join(crop_dir, disease)
            if not os.path.isdir(disease_dir):
                continue
            os.makedirs(os.path.join(PROC_DIR, split, crop, disease), exist_ok=True)

# Split ratios
ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
random.seed(42)

# Perform split
for crop in os.listdir(RAW_DIR):
    crop_dir = os.path.join(RAW_DIR, crop)
    if not os.path.isdir(crop_dir):
        continue

    for disease in os.listdir(crop_dir):
        disease_dir = os.path.join(crop_dir, disease)
        if not os.path.isdir(disease_dir):
            continue

        # Gather images
        images = [f for f in os.listdir(disease_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        random.shuffle(images)
        n = len(images)
        t_end = int(n * ratios['train'])
        v_end = t_end + int(n * ratios['val'])

        splits = {
            'train': images[:t_end],
            'val':   images[t_end:v_end],
            'test':  images[v_end:]
        }

        for split, img_list in splits.items():
            for img in img_list:
                src = os.path.join(disease_dir, img)
                dst = os.path.join(PROC_DIR, split, crop, disease, img)
                shutil.copy(src, dst)

print("✔ Data split complete — check data/processed/[train|val|test]/")