import os 
import torch
from torch.utils.data import Dataset
from PIL import Image
class Dataset(Dataset):
    def __init__(self, image_dir, labels, labels_bin, transform=None, ignore_value=-1):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels
        self.labels_bin = dict(labels_bin)
        self.ignore_value = ignore_value
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fname, y_class = self.labels[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y_bin = self.labels_bin.get(fname, self.ignore_value)
        return (
            img,
            torch.tensor(y_bin, dtype=torch.float32),
            torch.tensor(y_class, dtype=torch.long),
        )