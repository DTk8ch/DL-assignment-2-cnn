import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

if __package__:
    from .near_duplicate_audit import LABELS, find
else:
    from near_duplicate_audit import LABELS, find


def load_train_folds():
    root = Path(__file__).resolve().parent.parent
    audit_path = root / "outputs" / "train_duplicate_audit.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Missing outputs/train_duplicate_audit.csv. From the project root, run: "
            ".venv/bin/python cnn/near_duplicate_audit.py"
        )
    train = pd.read_csv(root / "data" / "train.csv")
    folds = pd.read_csv(root / "folds.csv")
    df = train.merge(folds, on="image_id", how="outer", validate="one_to_one", indicator=True)
    if not df["_merge"].eq("both").all():
        raise ValueError("folds.csv must contain every train ID exactly once")
    if not df.fold.isin(range(5)).all() or not df.excluded.isin([True, False]).all():
        raise ValueError("Every image needs a fold 0-4 and an explicit excluded flag")
    df["excluded"] = df.excluded.astype(bool)
    if set(df.loc[~df.excluded, "fold"]) != set(range(5)):
        raise ValueError("All five folds must contain active images")

    matches = pd.read_csv(audit_path)
    by_id = df.set_index("image_id")
    exact_parent = {image_id: image_id for image_id in df.image_id}
    group_folds = {}
    for row in matches.itertuples(index=False):
        a, b = by_id.loc[row.image_id_a], by_id.loc[row.image_id_b]
        group_folds.setdefault(row.group_id, set()).update([a.fold, b.fold])
        if row.sha256_equal:
            if not a[LABELS].equals(b[LABELS]):
                raise ValueError("Exact duplicates have conflicting labels")
            exact_parent[find(exact_parent, row.image_id_b)] = find(exact_parent, row.image_id_a)
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A duplicate group crosses folds")
    exact_groups = [find(exact_parent, image_id) for image_id in df.image_id]
    kept_counts = (~df.excluded).groupby(exact_groups).sum()
    if not kept_counts.eq(1).all():
        raise ValueError("Keep exactly one record per exact SHA-256 group")
    return df.loc[~df.excluded].drop(columns="_merge").reset_index(drop=True)


class CatsDogsDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = LABELS

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
    if fold_idx not in range(5):
        raise ValueError("fold_idx must be 0-4")
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    data_dir = project_root / "data"
    img_dir = data_dir / "train"

    df = load_train_folds()

    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    class_counts = train_df[LABELS].sum().to_numpy()
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)

    return train_loader, val_loader, pos_weights
