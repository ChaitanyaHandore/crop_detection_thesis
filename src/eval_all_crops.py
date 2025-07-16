#!/usr/bin/env python3
"""
eval_all_crops.py

Evaluate the six backbones trained by train_all_crops.py
(on the full multi-crop test set), printing metrics
and saving confusion‐matrix heatmaps.
"""
import os, sys, argparse
import numpy as np
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, UnidentifiedImageError

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/disease_data",
                   help="root folder containing train/ val/ test/")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def resolve_root(data_dir):
    # if data_dir/test exists, OK; otherwise look one level deeper
    if os.path.isdir(os.path.join(data_dir, "test")):
        return data_dir
    for sub in sorted(os.listdir(data_dir)):
        cand = os.path.join(data_dir, sub)
        if os.path.isdir(os.path.join(cand, "test")):
            return cand
    raise FileNotFoundError(f"Could not find test/ under {data_dir}")

def pil_loader(path):
    try:
        from PIL import Image
        img = Image.open(path)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        from PIL import Image
        return Image.new("RGB",(224,224),(0,0,0))

def build_model(name, num_classes):
    """Instantiate backbone + custom Dropout+Linear head."""
    if name=="alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        in_f = m.classifier[6].in_features
        m.classifier[6] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, num_classes)
        )
    elif name=="vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, num_classes)
        )
    elif name=="resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, num_classes)
        )
    elif name=="convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_f = m.classifier[2].in_features
        m.classifier[2] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, num_classes)
        )
    elif name=="vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_f = m.heads.head.in_features
        m.heads.head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, num_classes)
        )
    elif name=="mobilevit_s":
        m = timm.create_model("mobilevit_s", pretrained=False)
        in_f = m.head.fc.in_features
        m.head.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, num_classes)
        )
    else:
        raise ValueError(f"Unknown model {name}")
    return m

def main():
    args = parse_args()
    root = resolve_root(args.data_dir)
    print(f"→ Using data root: {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test-time transforms + loader
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    test_ds = datasets.ImageFolder(
        os.path.join(root,"test"),
        transform=test_tf,
        loader=pil_loader
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    classes = test_ds.classes
    print(f"→ Loaded test set: {len(test_ds)} images, {len(classes)} classes")

    # Six models & their checkpoint filenames
    MODELS = [
        ("alexnet",      "best_all_alexnet.pth"),
        ("vgg16",        "best_all_vgg16.pth"),
        ("resnet50",     "best_all_resnet50.pth"),
        ("convnext_tiny","best_all_convnext_tiny.pth"),
        ("vit_b_16",     "best_all_vit_b_16.pth"),
        ("mobilevit_s",  "best_all_mobilevit_s.pth"),
    ]

    for name, ckpt in MODELS:
        print(f"\n=== Evaluating {name} ===")
        model = build_model(name, len(classes))
        sd = torch.load(os.path.join(os.getcwd(), ckpt), map_location="cpu")
        model.load_state_dict(sd)
        model.to(device).eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb,yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        acc = (y_pred==y_true).mean()
        print(f"Accuracy: {acc:.4f}\n")
        print(classification_report(y_true, y_pred,
                                    target_names=classes, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix – {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        out_fn = f"all_crop_cm_{name}.png"
        plt.savefig(out_fn, bbox_inches="tight")
        plt.close()
        print(f"→ saved confusion matrix to {out_fn}")

if __name__=="__main__":
    main()
