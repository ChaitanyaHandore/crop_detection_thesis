from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler

def filter_corrupt(dataset):
    good = []
    for path, label in dataset.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, label))
        except Exception:
            print("Filtering out corrupt:", path)
    dataset.samples = good
    dataset.targets = [l for _, l in good]

def compute_class_weights(dataset):
    counts = np.zeros(len(dataset.classes), dtype=np.int64)
    for _, label in dataset.samples:
        counts[label] += 1
    inv = 1.0 / counts
    return inv / inv.sum()

def main():
    # dirs
    BASE   = os.path.dirname(__file__)
    DATA   = os.path.join(BASE, '..', 'data', 'processed')
    OUTDIR = os.path.join(BASE, '..', 'models')
    os.makedirs(OUTDIR, exist_ok=True)

    # hyperparams
    BS, EPOCHS = 32, 10
    LR_BACKBONE, LR_HEAD = 1e-5, 1e-4

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # datasets
    train_ds = datasets.ImageFolder(os.path.join(DATA,'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA,'val'),   transform=val_tf)

    filter_corrupt(train_ds)
    filter_corrupt(val_ds)

    # sampler + loader
    weights = compute_class_weights(train_ds)
    sample_w = [weights[l] for _,l in train_ds.samples]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BS, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False,       num_workers=0)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = models.resnet18(pretrained=True)
    # freeze all but layer4+fc
    for name,param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False
    # adjust head
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    # loss & opt
    class_w = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': LR_BACKBONE},
        {'params': model.fc.parameters(),    'lr': LR_HEAD}
    ])

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        running_loss = running_corr = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            opt.step()
            preds = out.argmax(1)
            running_loss += loss.item()*x.size(0)
            running_corr += (preds==y).sum().item()
        train_loss = running_loss/len(train_ds)
        train_acc  = running_corr/len(train_ds)

        # val
        model.eval()
        v_loss=v_corr=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss=criterion(out,y)
                v_loss += loss.item()*x.size(0)
                v_corr += (out.argmax(1)==y).sum().item()
        val_loss = v_loss/len(val_ds)
        val_acc  = v_corr/len(val_ds)

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  "
              f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(), os.path.join(OUTDIR,'best_cnn.pth'))

    print(f"Done. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    from PIL import Image
    main()