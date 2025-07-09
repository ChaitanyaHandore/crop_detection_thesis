#!/usr/bin/env python3
import os, sys
# ensure src/ is on sys.path
sys.path.append(os.path.dirname(__file__))
#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import timm
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# make sure we can import your dataset class
sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
from train_single_crop import SingleCropDataset

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "/content/data/disease_data"
CROP     = "Maize"
BATCH    = 32
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("alexnet",     lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT),     "best_alexnet.pth"),
    ("convnext_tiny", lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT), "best_convnext_tiny.pth"),
    ("vit_b_16",    lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),       "best_vit_b_16.pth"),
    ("mobilevit_s", lambda: timm.create_model("mobilevit_s", pretrained=False),                  "best_mobilevit_s.pth"),
    ("resnet50",    lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),            "best_resnet50.pth"),
    ("vgg16",       lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),                  "best_vgg16.pth"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2) Test‐time transforms & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
test_ds = SingleCropDataset(DATA_DIR, CROP, "test", transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=2, pin_memory=True)
classes = test_ds.idx_to_class

print(f"→ Loaded test set: {len(test_ds)} images, {len(classes)} classes")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
for name, build_fn, ckpt in MODELS:
    print(f"\n\n=== Evaluating {name} ===")

    # 3a) Build & inject the SAME head you used in training
    model = build_fn()
    n_cls = len(classes)

    if name == "alexnet":
        in_f = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, n_cls)
        )

    elif name == "vgg16":
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, n_cls)
        )

    elif name == "resnet50":
        in_f = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, n_cls)
        )

    elif name == "convnext_tiny":
        in_f = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, n_cls)
        )

    elif name == "vit_b_16":
        in_f = model.heads.head.in_features
        model.heads.head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, n_cls)
        )

    elif name == "mobilevit_s":
        in_f = model.head.fc.in_features
        model.head.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_f, n_cls)
        )

    # 3b) Load checkpoint
    ckpt_path = os.path.join(os.getcwd(), ckpt)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    # 3c) Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # 3d) Metrics
    acc = (y_pred == y_true).mean()
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # 3e) Confusion‐matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Confusion Matrix – {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 3f) Save
    out_fn = f"cm_{name}.png"
    fig.savefig(out_fn, bbox_inches="tight")
    print(f"→ saved confusion matrix to {out_fn}")
    plt.close(fig)
