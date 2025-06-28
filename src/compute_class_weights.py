import os
import numpy as np

def compute_weights(data_dir, crop):

    train_root = os.path.join(data_dir, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Expected train/ under {data_dir}, got none")

    # gather all subfolders starting with "{crop}_"
    class_dirs = sorted(
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
        and d.startswith(f"{crop}_")
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"No subfolders matching '{crop}_*' under {train_root}"
        )

    counts = []
    for d in class_dirs:
        path = os.path.join(train_root, d)
        # count only files
        n = sum(
            os.path.isfile(os.path.join(path, f))
            for f in os.listdir(path)
        )
        counts.append(n)

    total = sum(counts)
    K = len(counts)
    # inverse-frequency unnormalized
    weights = [ total/(K * c) for c in counts ]

    # scale so that sum(weights) == K
    factor = K / sum(weights)
    weights = [ w * factor for w in weights ]

    # strip the "crop_" prefix for clarity
    classes = [ d[len(crop)+1:] for d in class_dirs ]
    return classes, weights

if __name__ == "__main__":
    import argparse
    # compute own script folder, then data/disease_data
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_dd = os.path.normpath(os.path.join(this_dir, "..", "data", "disease_data"))

    p = argparse.ArgumentParser(
        description="Compute per-class weights for an imbalanced single-crop dataset"
    )
    p.add_argument(
        "--data_dir", default=default_dd,
        help="root of your disease_data (containing train/, val/, test/)"
    )
    p.add_argument(
        "--crop", required=True,
        help="prefix of your crop folders, e.g. 'Maize' to grab Maize_* subfolders"
    )
    args = p.parse_args()

    classes, weights = compute_weights(args.data_dir, args.crop)
    print("Classes (in order):", classes)
    print("Computed class weights:", [f"{w:.3f}" for w in weights])
    print("\nUse these values with --class_weights in train_single_crop.py")