"""Module 2 (part 2): PyTorch dataset, transforms and dataloaders.
Reads data/splits.csv and data/images/. Each target is [fire, smoke] as floats.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMG_DIR = Path("data/images")
SPLITS_CSV = Path("data/splits.csv")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats, needed for pretrained models


def get_transforms(train: bool, size: int = 320):
    if train:
        return T.Compose([
            # mild crop only: fires can be tiny, and an aggressive crop could remove the fire but keep the label
            T.RandomResizedCrop(size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(),
            # no hue shift: colour is a key fire cue
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class FireSmokeDataset(Dataset):
    def __init__(self, split: str, transform=None, csv_path=SPLITS_CSV, img_dir=IMG_DIR):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row["file"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = torch.tensor([row["fire"], row["smoke"]], dtype=torch.float32)
        return img, target


def get_loaders(batch_size: int = 32, size: int = 320, num_workers: int = 0):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = FireSmokeDataset(split, get_transforms(train=(split == "train"), size=size))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers)
    return loaders


if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image

    loaders = get_loaders(batch_size=16)
    for name, loader in loaders.items():
        x, y = next(iter(loader))
        print(f"{name}: images {tuple(x.shape)}, targets {tuple(y.shape)}, "
              f"fire rate {y[:, 0].mean():.2f}, smoke rate {y[:, 1].mean():.2f}")

    x, y = next(iter(loaders["train"]))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    Path("outputs").mkdir(exist_ok=True)
    save_image(make_grid((x * std + mean).clamp(0, 1), nrow=4), "outputs/augment_check.png")
    print("saved outputs/augment_check.png")