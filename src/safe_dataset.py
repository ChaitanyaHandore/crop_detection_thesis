# src/safe_dataset.py
import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class SingleCropDataset(Dataset):
    def __init__(self, data_dir, crop, split, transform=None):
        """
        Expects directory structure:
          data_dir/
            train/val/test/
              <CROP>_<CLASS>/
                img1.jpg, img2.jpg, ...
        """
        super().__init__()
        split_dir = os.path.join(data_dir, split)
        # gather all (path,label) pairs
        self.samples = []
        self.classes = sorted(d for d in os.listdir(split_dir)
                              if os.path.isdir(os.path.join(split_dir, d))
                              and d.startswith(crop + "_"))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                self.samples.append((path, self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Skip this corrupt image by picking the next one
            # (modulo to avoid overflow)
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            img = self.transform(img)
        return img, label
