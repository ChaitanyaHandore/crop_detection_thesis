#!/usr/bin/env python3
import os
import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class SingleCropDataset(Dataset):
    def __init__(self, data_dir, crop, split, transform=None):
        """
        data_dir/
          train/
            Maize_fall_armyworm/
            Maize_grasshoper/
            ...
          val/
          test/
        """
        split_dir = os.path.join(data_dir, split)
        # find only those folders starting with e.g. "Maize_"
        all_dirs = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
               and d.startswith(crop + "_")
        ]
        if not all_dirs:
            raise FileNotFoundError(f"No '{crop}_*' folders in {split_dir}")

        # strip prefix for human-readable class names
        self.classes = sorted(d[len(crop)+1:] for d in all_dirs)
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        self.samples = []
        for folder in all_dirs:
            cls_name = folder[len(crop)+1:]
            cls_idx  = self.class_to_idx[cls_name]
            folder_path = os.path.join(split_dir, folder)
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                # skip non-images
                if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png")):
                    continue
                self.samples.append((fpath, cls_idx))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # handle broken images gracefully
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # return a random valid sample instead
            return self.__getitem__(random.randrange(len(self)))
        if self.transform:
            img = self.transform(img)
        return img, label

def make_loaders(data_dir, crop, batch_size, num_workers=4):
    # common image transforms
    tf = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(  # ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ]),
        "test": None  # we'll reuse val transforms
    }
    tf["test"] = tf["val"]

    datasets = {
        split: SingleCropDataset(data_dir, crop, split, transform=tf[split])
        for split in ("train","val","test")
    }

    loaders = {
        split: DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split=="train"),
            num_workers=num_workers, pin_memory=True
        )
        for split,ds in datasets.items()
    }
    num_classes = len(datasets["train"].classes)
    print(f"→ `{crop}` classes ({num_classes}): {datasets['train'].classes}")
    print(f"→ train batches: {len(loaders['train'])}, val: {len(loaders['val'])}")
    return loaders, num_classes

def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, num_classes = make_loaders(
        args.data_dir, args.crop, args.batch_size, args.num_workers
    )

    # load ConvNeXt-Tiny
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if args.pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, num_classes)
    model = model.to(device)

    # class weights
    cw = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    if cw.numel() != num_classes:
        raise ValueError(f"Expected {num_classes} weights, got {cw.numel()}")
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        # Training
        model.train()
        running_loss = running_correct = 0
        for x,y in loaders["train"]:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss   += loss.item() * x.size(0)
            running_correct+= (out.argmax(1)==y).sum().item()
        train_loss = running_loss/len(loaders["train"].dataset)
        train_acc  = running_correct/len(loaders["train"].dataset)

        # Validation
        model.eval()
        val_loss = val_correct = 0
        with torch.no_grad():
            for x,y in loaders["val"]:
                x,y = x.to(device), y.to(device)
                out = model(x)
                val_loss    += criterion(out,y).item() * x.size(0)
                val_correct += (out.argmax(1)==y).sum().item()
        val_loss /= len(loaders["val"].dataset)
        val_acc  = val_correct/len(loaders["val"].dataset)

        print(f"Epoch {epoch}/{args.epochs}"
              f"  Train {train_loss:.4f}/{train_acc:.4f}"
              f"  Val   {val_loss:.4f}/{val_acc:.4f}")

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # save a checkpoint
            torch.save(model.state_dict(), args.output)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"→ Early stopping after {epoch} epochs")
                break

    print(f"🏁 Best Val Acc: {best_val_acc:.4f}  (checkpoint saved to {args.output})")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      required=True,
                   help="root folder containing train/ val/ test/")
    p.add_argument("--crop",          required=True,
                   help="e.g. Maize")
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--batch_size",type=int, default=32)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--class_weights", nargs="+", type=float, required=True,
                   help="one weight per subclass under <crop>_*")
    p.add_argument("--pretrained", action="store_true",
                   help="use ImageNet pretrained weights")
    p.add_argument("--patience", type=int, default=5,
                   help="early-stop after this many val checks without improvement")
    p.add_argument("--output", default="best_convnext.pth",
                   help="where to save best model")
    args = p.parse_args()

    train_and_eval(args)
