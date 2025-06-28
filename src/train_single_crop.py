import os, argparse
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class SingleCropDataset(Dataset):
    def __init__(self, data_root, crop, split, transform=None):
        self.transform = transform
        self.samples = []

        split_root = os.path.join(data_root, split)
        for entry in sorted(os.listdir(split_root)):
            if not entry.startswith(crop + "_"):
                continue
            cls_name = entry[len(crop)+1:]
            cls_dir  = os.path.join(split_root, entry)
            if not os.path.isdir(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                try:
                    # fully load to catch any broken streams
                    with Image.open(path) as img:
                        img.load()
                    self.samples.append((path, cls_name))
                except Exception:
                    print(f"⚠️  Skipping corrupt: {path}")
                    continue

        self.classes = sorted({cls for _, cls in self.samples})
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        print(f"→ `{split}` classes: {self.classes}\n  Total: {len(self.classes)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[cls_name]

def make_loaders(data_dir, crop, batch_size):
    tf = {
      "train": transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
      ]),
      "val": transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
      ]),
    }

    loaders, num_classes = {}, None
    for split in ("train","val"):
        ds = SingleCropDataset(data_dir, crop, split, transform=tf[split])
        if num_classes is None:
            num_classes = len(ds.classes)
            print(f"→ `{split}` classes: {ds.classes}\n  Total: {num_classes}\n")
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split=="train"),
            num_workers=2, pin_memory=False, drop_last=(split=="train")
        )
    return loaders, num_classes

def train_and_eval(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loaders, num_classes = make_loaders(args.data_dir, args.crop, args.batch_size)

    model = models.convnext_tiny(
      weights=(models.ConvNeXt_Tiny_Weights.DEFAULT if args.pretrained else None)
    )
    in_f = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_f, num_classes)
    model.to(device)

    if len(args.class_weights)!=num_classes:
        raise ValueError("class_weights length mismatch")
    weights = torch.tensor(args.class_weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val, stop_cnt = 0.0, 0
    for ep in range(1, args.epochs+1):
        # train
        model.train()
        t_loss=t_corr=t_tot=0
        for x,y in loaders["train"]:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()*x.size(0)
            preds = out.argmax(1)
            t_corr += (preds==y).sum().item()
            t_tot += y.size(0)
        tr_l, tr_a = t_loss/t_tot, t_corr/t_tot

        # validate every 2 epochs
        if ep==1 or ep%2==0:
            model.eval()
            v_loss=v_corr=v_tot=0
            with torch.no_grad():
                for x,y in loaders["val"]:
                    x,y = x.to(device), y.to(device)
                    out = model(x)
                    l = criterion(out,y)
                    v_loss += l.item()*x.size(0)
                    p = out.argmax(1)
                    v_corr += (p==y).sum().item()
                    v_tot += y.size(0)
            v_l, v_a = v_loss/v_tot, v_corr/v_tot
            print(f"Epoch {ep}/{args.epochs}  "
                  f"Train {tr_l:.4f}/{tr_a:.4f}  "
                  f" Val  {v_l:.4f}/{v_a:.4f}")
            if v_a>best_val:
                best_val, stop_cnt = v_a, 0
            else:
                stop_cnt+=1
                if stop_cnt>=5:
                    print("Early stopping.")
                    break
        else:
            print(f"Epoch {ep}/{args.epochs}  Train {tr_l:.4f}/{tr_a:.4f}  (val skipped)")

    print(f"🏁 Done. Best Val Acc: {best_val:.4f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/disease_data")
    p.add_argument("--crop",        required=True)
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch_size",type=int, default=32)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--class_weights", nargs="+", type=float, required=True)
    p.add_argument("--pretrained", action="store_true")
    args = p.parse_args()
    train_and_eval(args)