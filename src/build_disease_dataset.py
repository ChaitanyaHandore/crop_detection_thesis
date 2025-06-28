import os, shutil


SRC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
DST = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_data')

for split in ['train', 'val', 'test']:
    for crop in os.listdir(os.path.join(SRC, split)):
        crop_dir = os.path.join(SRC, split, crop)
        if not os.path.isdir(crop_dir): continue
        for disease in os.listdir(crop_dir):
            src_folder = os.path.join(crop_dir, disease)
            if not os.path.isdir(src_folder): continue

            cls = f"{crop}_{disease}".replace(' ', '_')
            dst_folder = os.path.join(DST, split, cls)
            os.makedirs(dst_folder, exist_ok=True)
            for img in os.listdir(src_folder):
                shutil.copy(os.path.join(src_folder, img),
                            os.path.join(dst_folder, img))
    print(f"✔ Built split {split}")

print("✅ disease_data ready with 22 folders per split")