import os

# Point this to the folder that contains Cashew, Cassava, Maize, Tomato
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'Raw Data', 'CCMT Dataset')

print("Dataset layout under:", RAW_DIR)
for crop in sorted(os.listdir(RAW_DIR)):
    crop_path = os.path.join(RAW_DIR, crop)
    if not os.path.isdir(crop_path):
        continue
    diseases = [d for d in os.listdir(crop_path)
                if os.path.isdir(os.path.join(crop_path, d))]
    print(f"- {crop}: {len(diseases)} disease folders → {diseases}")