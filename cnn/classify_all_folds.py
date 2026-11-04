"""Run the supplied fold-0 checkpoints on all training folds.

Scores for fold 1-4 are in-sample: those images were used to train these
checkpoints. Only fold 0 is held-out validation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ensemble import TestImages, load_model, predict


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("D:/downloads/model_baseline.ckpt"))
    parser.add_argument("--se", type=Path, default=Path("D:/downloads/model_se.ckpt"))
    parser.add_argument("--cbam", type=Path, default=Path("D:/downloads/model_cbam.ckpt"))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "all_folds")
    args = parser.parse_args()

    data = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one"
    ).sort_values(["fold", "image_id"]).reset_index(drop=True)
    ids = data.image_id.to_numpy(dtype=str)
    targets = data[LABELS].to_numpy(dtype=np.int64)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ["baseline", "se", "cbam"]:
        output = args.output_dir / f"probabilities_{name}.npz"
        if output.exists():
            with np.load(output) as saved:
                if np.array_equal(saved["image_ids"], ids):
                    print(f"Reusing {name} all-fold predictions", flush=True)
                    continue
            raise ValueError(f"Existing {output} has different image IDs")

        probs = np.empty((len(data), len(LABELS)), dtype=np.float32)
        heldout = data.fold.eq(0).to_numpy()
        with np.load(ROOT / "outputs" / "ensemble" / f"probabilities_{name}.npz") as saved:
            original_ids = saved["validation_ids"]
            positions = pd.Index(original_ids).get_indexer(ids[heldout])
            if np.any(positions < 0):
                raise ValueError("Cached fold-0 probabilities do not match fold IDs")
            probs[heldout] = saved["validation_probs"][positions]
        model = load_model(name, getattr(args, name)).to(device)
        remaining = ~heldout
        loader = DataLoader(
            TestImages(ids[remaining], ROOT / "data" / "train", transform),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
            pin_memory=device.type == "cuda",
        )
        print(f"Classifying {remaining.sum()} in-sample images with {name} on {device}", flush=True)
        probs[remaining] = predict(model, loader, device)
        np.savez_compressed(output, image_ids=ids, folds=data.fold.to_numpy(dtype=int),
                            targets=targets, probabilities=probs)
        print(f"Saved {output}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
