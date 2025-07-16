#!/usr/bin/env python3
"""
train_all_crops.py

Train one of [resnet50, vgg16, alexnet, convnext_tiny, vit_b_16, mobilevit_s]
on ALL crops/diseases (22 classes), with class-balanced sampling, data-cleaning,
augmentation, dropout, weight-decay, early stopping, and plotting of loss/accuracy curves.
"""
import os, argparse, time
from PIL import Image, UnidentifiedImageError, ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
import timm
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Train one CNN across ALL crops/diseases"
    )
    p.add_argument("--data_dir",    required=True,
                   help="root folder with train/ val/ test/")
    p.add_argument("--model",       choices=[
                       "resnet50","vgg16","alexnet",
                       "convnext_tiny","vit_b_16","mobilevit_s"
                   ], required=True,
                   help="which backbone to train")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--pretrained",  action="store_true",
                   help="use ImageNet-pretrained weights")
    return p.parse_args()

def pil_loader(path):
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return Image.new("RGB", (224,224), (0,0,0))

def make_loaders(root, bs):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
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

    train_ds = datasets.ImageFolder(
        os.path.join(root,"train"), transform=train_tf, loader=pil_loader
    )
    val_ds   = datasets.ImageFolder(
        os.path.join(root,"val"),   transform=val_tf, loader=pil_loader
    )

    counts = [0]*len(train_ds.classes)
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    total = sum(counts)
    inv = [total/c for c in counts]
    mean_inv = sum(inv)/len(inv)
    class_weights = [v/mean_inv for v in inv]

    sample_w = [class_weights[label] for _,label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, class_weights, train_ds.classes

def build_backbone(name, num_classes, pretrained):
    if name=="resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.5),
                             nn.Linear(in_f, num_classes))

    elif name=="vgg16":
        w = models.VGG16_Weights.DEFAULT if pretrained else None
        m = models.vgg16(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_f, num_classes))

    elif name=="alexnet":
        w = models.AlexNet_Weights.DEFAULT if pretrained else None
        m = models.alexnet(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_f, num_classes))

    elif name=="convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_f, num_classes))

    elif name=="vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vit_b_16(weights=w)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Sequential(nn.Dropout(0.5),
                                     nn.Linear(in_f, num_classes))

    else:  # mobilevit_s
        m = timm.create_model("mobilevit_s", pretrained=pretrained)
        in_f = m.head.fc.in_features
        m.head.fc = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(in_f, num_classes))

    return m

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, cw, classes = make_loaders(
        args.data_dir, args.batch_size
    )
    print(f"→ Classes ({len(classes)}): {classes}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, device=device)
    )
    model = build_backbone(
        args.model, len(classes), args.pretrained
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_acc, no_improve = 0.0, 0

    # start timing
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        model.train()
        t_loss = t_corr = t_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            t_loss  += loss.item()*xb.size(0)
            preds    = out.argmax(1)
            t_corr  += (preds==yb).sum().item()
            t_total += xb.size(0)

        train_loss = t_loss/t_total
        train_acc  = t_corr/t_total

        model.eval()
        v_loss = v_corr = v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                l = criterion(out, yb)
                v_loss += l.item()*xb.size(0)
                v_corr += (out.argmax(1)==yb).sum().item()
                v_total+= xb.size(0)

        val_loss = v_loss/v_total
        val_acc  = v_corr/v_total

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train {train_loss:.4f}/{train_acc:.4f}  "
              f"Val   {val_loss:.4f}/{val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), f"best_all_{args.model}.pth")
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"→ Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print(f"🏁 Best Val Acc: {best_acc:.4f}")
    print(f"⏱️ Total run time: {total_time/60:.2f} minutes ({total_time:.0f} sec)")

    epochs = range(1, len(history["train_loss"])+1)
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"],   label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"],   label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()
    out_fn = f"{args.model}_loss_acc.png"
    plt.savefig(out_fn)
    print(f"→ Saved curves to {out_fn}")

if __name__=="__main__":
    main()
