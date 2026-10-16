import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path


class CatsDogsDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']}.jpg"

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)

        target = torch.tensor(row[self.labels].to_numpy(dtype="float32"))
        return image, target


def get_dataloaders(fold_idx=0, batch_size=32, num_workers=4):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    data_dir = project_root / "data"
    folds_csv_path = project_root / "folds.csv"
    train_csv_path = data_dir / "train.csv"
    img_dir = data_dir / "train"

    df = pd.read_csv(train_csv_path)
    folds = pd.read_csv(folds_csv_path)
    df = df.merge(folds, on="image_id", validate="one_to_one")

    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    labels = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
    class_counts = train_df[labels].sum().to_numpy()
    total_train_samples = len(train_df)

    pos_weights = torch.tensor(
        [(total_train_samples - count) / max(count, 1) for count in class_counts],
        dtype=torch.float32
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CatsDogsDataset(train_df, img_dir, train_transform)
    val_dataset = CatsDogsDataset(val_df, img_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, pos_weights
