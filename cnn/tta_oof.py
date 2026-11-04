"""Cache horizontal-flip test-time predictions for fold-specific models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from engine import PetClassifier
from ensemble import TestImages, load_model, predict


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="se", choices=["se", "cbam", "baseline"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one"
    )
    test = pd.read_csv(ROOT / "data" / "test.csv")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in args.folds:
        directory = ROOT / "outputs" / "oof" / f"{args.mode}_fold{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "horizontal_flip.npz"
        if output.exists():
            print(f"Reusing flip predictions for {args.mode} fold {fold}", flush=True)
            continue
        validation = data.loc[data.fold.eq(fold)].reset_index(drop=True)
        if fold == 0:
            source = Path(f"D:/downloads/model_{args.mode}.ckpt")
            model = load_model(args.mode, source)
        else:
            source = directory / "best.ckpt"
            model = PetClassifier.load_from_checkpoint(
                source, mode=args.mode, pretrained=False, pos_weights=torch.ones(5),
                strict=True, map_location="cpu",
            )
        model = model.to(device)
        val_loader = DataLoader(
            TestImages(validation.image_id, ROOT / "data" / "train", transform),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            TestImages(test.image_id, ROOT / "data" / "test", transform),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        print(f"Flipped inference for {args.mode} fold {fold}", flush=True)
        val_probs = predict(model, val_loader, device)
        test_probs = predict(model, test_loader, device)
        np.savez_compressed(output,
                            validation_ids=validation.image_id.to_numpy(dtype=str),
                            validation_probs=val_probs,
                            test_ids=test.image_id.to_numpy(dtype=str),
                            test_probs=test_probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
