#!/usr/bin/env python3
"""
train_single_crop.py

Train/evaluate a single‐crop disease classifier with:

  • Modern:    convnext_tiny, vit_b_16, mobilevit_s  
  • Classic:   resnet50, vgg16, alexnet  

Includes augmentation, dropout, weight‐decay, early stopping,
measures total training time, and saves a PNG of train/val loss & accuracy curves.
"""
import os, argparse, random, time
from PIL import Image, UnidentifiedImageError
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1) Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SingleCropDataset(Dataset):
    def __init__(self, data_dir, crop, split, transform=None):
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(data_dir, split)
        for cls_name in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_path): continue
            if not cls_name.startswith(crop + "_"): continue
            label = cls_name.split("_",1)[1]
            for fname in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, fname), label))
        labels = sorted({lbl for _, lbl in self.samples})
        self.class_to_idx = {lbl:i for i,lbl in enumerate(labels)}
        self.idx_to_class = labels

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path,label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx+1)%len(self))
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


# ─────────────────────────────────────────────────────────────────────────────
# 2) Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data/disease_data",
                   help="root folder with train/ val/ test/")
    p.add_argument("--crop",         required=True,
                   help="crop name prefix, e.g. Maize or Tomato")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2,
                   help="L2 regularization for AdamW")
    p.add_argument("--class_weights", type=float, nargs="+", required=True,
                   help="one float per disease subclass")
    p.add_argument("--pretrained",   action="store_true",
                   help="load ImageNet pretrained weights")
    p.add_argument("--model", choices=[
        "convnext_tiny","vit_b_16","mobilevit_s",
        "resnet50","vgg16","alexnet"
    ], default="convnext_tiny", help="architecture to use")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 3) Training/Eval + Plotting
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3a) Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 3b) DataLoaders
    splits = ["train","val"]
    loaders = {}
    for split in splits:
        ds = SingleCropDataset(args.data_dir, args.crop, split,
                               transform=(train_tf if split=="train" else val_tf))
        loaders[split] = DataLoader(ds,
            batch_size=args.batch_size,
            shuffle=(split=="train"),
            num_workers=4,
            pin_memory=True
        )

    num_classes = len(loaders["train"].dataset.idx_to_class)
    print(f"→ `{args.crop}` has {num_classes} subclasses:",
          loaders["train"].dataset.idx_to_class)

    # 3c) Loss & optimizer
    cw = torch.tensor(args.class_weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # 3d) Build model + replace head
    print(f"→ Building `{args.model}`, pretrained={args.pretrained}")
    if args.model=="convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if args.pretrained else None
        model = models.convnext_tiny(weights=w)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.5),
                                             nn.Linear(in_f, num_classes))

    elif args.model=="vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = models.vit_b_16(weights=w)
        in_f = model.heads.head.in_features
        model.heads.head = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_f, num_classes))

    elif args.model=="mobilevit_s":
        model = timm.create_model("mobilevit_s", pretrained=args.pretrained)
        in_f = model.head.fc.in_features
        model.head.fc = nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(in_f, num_classes))

    elif args.model=="resnet50":
        w = models.ResNet50_Weights.DEFAULT if args.pretrained else None
        model = models.resnet50(weights=w)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(in_f, num_classes))

    elif args.model=="vgg16":
        w = models.VGG16_Weights.DEFAULT if args.pretrained else None
        model = models.vgg16(weights=w)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.5),
                                              nn.Linear(in_f, num_classes))

    else:  # alexnet
        w = models.AlexNet_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = models.alexnet(weights=w)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.5),
                                              nn.Linear(in_f, num_classes))

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    # 3e) Train/validate loop
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc, no_improve = 0.0, 0

    # record start time
    t_start = time.time()

    for epoch in range(1, args.epochs+1):
        # — train —
        model.train()
        run_loss=correct=total=0
        for imgs, labels in loaders["train"]:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()

            run_loss += loss.item()*imgs.size(0)
            preds    = out.argmax(1)
            correct += (preds==labels).sum().item()
            total   += imgs.size(0)

        train_loss = run_loss/total
        train_acc  = correct/total

        # — validate —
        model.eval()
        run_loss=correct=total=0
        with torch.no_grad():
            for imgs, labels in loaders["val"]:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                run_loss += criterion(out, labels).item()*imgs.size(0)
                preds    = out.argmax(1)
                correct += (preds==labels).sum().item()
                total   += imgs.size(0)

        val_loss = run_loss/total
        val_acc  = correct/total

        # record & print
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train {train_loss:.4f}/{train_acc:.4f}  "
              f"Val   {val_loss:.4f}/{val_acc:.4f}")

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{args.model}.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"→ Early stopping after epoch {epoch}")
                break

    # compute and print elapsed time
    elapsed = time.time() - t_start
    print(f"🏁 Done. Best Val Acc: {best_val_acc:.4f}")
    print(f"⏱️ Total training time: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")

    # 3f) Plot & save curves
    epochs = range(1, len(history["train_loss"])+1)
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"],   label="val")
    plt.title("Loss");    plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"],   label="val")
    plt.title("Accuracy");plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()
    out_fn = f"training_curves_{args.model}.png"
    plt.savefig(out_fn)
    print(f"✅ Saved curves to {out_fn}")


if __name__=="__main__":
    main()
