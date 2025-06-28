import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

def filter_corrupt(dataset):
    """Remove corrupt images from an ImageFolder dataset in place."""
    good = []
    for path, label in dataset.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, label))
        except Exception:
            print("Filtering out corrupt test image:", path)
    dataset.samples = good
    dataset.targets = [lab for _, lab in good]

def main():
    # Paths
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(base_dir, '..', 'data', 'processed')
    model_path = os.path.join(base_dir, '..', 'models', 'best_cnn.pth')

    # Transforms
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    # Build dataset + filter corrupt
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                   transform=test_tf)
    filter_corrupt(test_ds)

    # Device & loader args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_kwargs = dict(batch_size=32,
                         shuffle=False,
                         num_workers=4)

    if device.type == 'cuda':
        loader_kwargs['pin_memory'] = True

    test_loader = DataLoader(test_ds, **loader_kwargs)

    # Model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(test_ds.classes))
    model.load_state_dict(torch.load(model_path,
                                     map_location=device))
    model = model.to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Metrics
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds,
                                target_names=test_ds.classes))

    print("Confusion Matrix:\n")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    main()