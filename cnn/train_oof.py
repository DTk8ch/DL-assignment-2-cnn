import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from engine import PetClassifier
from ensemble import TestImages, predict
from loader import CatsDogsDataset, load_train_folds
from near_duplicate_audit import sha256


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


class EpochLog(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["split_sha256"] = trainer.split_sha256
        checkpoint["training_ids"] = trainer.training_ids
        checkpoint["validation_ids"] = trainer.validation_ids

    def on_validation_epoch_end(self, trainer, model):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val_macro_f1")
        if value is not None:
            print(f"{model.hparams.mode} fold {trainer.datamodule_fold}: "
                  f"epoch {trainer.current_epoch + 1}, val F1 {float(value):.4f}", flush=True)


def train_one(fold, mode, args):
    data = load_train_folds()
    split_hash = sha256(ROOT / "folds.csv")
    train = data.loc[data.fold.ne(fold)].reset_index(drop=True)
    val = data.loc[data.fold.eq(fold)].reset_index(drop=True)
    out = args.output_dir / f"{mode}_fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    probabilities_file = out / "probabilities.npz"
    if probabilities_file.exists():
        with np.load(probabilities_file) as cached:
            if ("split_sha256" not in cached.files or cached["split_sha256"].item() != split_hash or
                    "training_ids" not in cached.files or
                    not np.array_equal(cached["training_ids"], train.image_id.to_numpy(dtype=str)) or
                    not np.array_equal(cached["validation_ids"], val.image_id.to_numpy(dtype=str))):
                raise ValueError("Cached OOF uses a different split; choose a new --output-dir")
        print(f"Reusing completed {mode} fold {fold}", flush=True)
        return

    pl.seed_everything(42 + fold, workers=True)
    torch.set_float32_matmul_precision("medium")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), norm,
    ])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=.5),
        transforms.ToTensor(), norm,
    ])
    train_loader = DataLoader(
        CatsDogsDataset(train, ROOT / "data" / "train", train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        CatsDogsDataset(val, ROOT / "data" / "train", val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    counts = train[LABELS].sum().to_numpy(dtype=np.float32)
    pos_weight = torch.tensor((len(train) - counts) / np.maximum(counts, 1))
    model = PetClassifier(mode=mode, pos_weights=pos_weight, lr=1e-4, pretrained=True)
    checkpoint = ModelCheckpoint(
        dirpath=out, filename="best", monitor="val_macro_f1", mode="max",
        save_top_k=1, save_weights_only=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="16-mixed", max_epochs=args.epochs,
        logger=CSVLogger(save_dir=out, name="training"),
        callbacks=[checkpoint, EarlyStopping(monitor="val_macro_f1", mode="max", patience=3),
                   EpochLog()],
        enable_progress_bar=False, log_every_n_steps=20,
    )
    trainer.datamodule_fold = fold
    trainer.split_sha256 = split_hash
    trainer.training_ids = train.image_id.tolist()
    trainer.validation_ids = val.image_id.tolist()
    trainer.fit(model, train_loader, val_loader)
    print(f"Best {mode} fold {fold}: {checkpoint.best_model_score} at "
          f"{checkpoint.best_model_path}", flush=True)
    del model, trainer
    torch.cuda.empty_cache()

    model = PetClassifier.load_from_checkpoint(
        checkpoint.best_model_path, mode=mode, pos_weights=pos_weight,
        pretrained=False, strict=True, map_location="cpu",
    ).to("cuda")
    val_probs = predict(model, val_loader, torch.device("cuda"))
    test_loader = DataLoader(
        TestImages(test.image_id, ROOT / "data" / "test", val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_probs = predict(model, test_loader, torch.device("cuda"))
    np.savez_compressed(
        probabilities_file,
        split_sha256=split_hash,
        training_ids=train.image_id.to_numpy(dtype=str),
        validation_ids=val.image_id.to_numpy(dtype=str),
        validation_probs=val_probs,
        test_ids=test.image_id.to_numpy(dtype=str),
        test_probs=test_probs,
        fold=fold, mode=mode, checkpoint=str(checkpoint.best_model_path),
    )
    print(f"Saved {mode} fold {fold} probabilities", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Train folds with cleaned data")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--mode", choices=["se", "cbam", "baseline"], default="se")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "oof_clean")
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds):
        parser.error("Folds must be 0-4")
    for fold in args.folds:
        train_one(fold, args.mode, args)


if __name__ == "__main__":
    main()
