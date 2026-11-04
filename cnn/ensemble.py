"""Compare probability-averaging ensembles on fold 0 and write Kaggle submissions.

Example:
    python cnn/ensemble.py --data-dir data \
        --baseline D:/downloads/model_baseline.ckpt \
        --se D:/downloads/model_se.ckpt \
        --cbam D:/downloads/model_cbam.ckpt
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from engine import PetClassifier
from loader import CatsDogsDataset
from constraints import apply_label_constraints


LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
ROOT = Path(__file__).resolve().parent.parent


class TestImages(Dataset):
    def __init__(self, image_ids, image_dir, transform):
        self.image_ids = list(image_ids)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        with Image.open(self.image_dir / f"{image_id}.jpg") as image:
            return self.transform(image.convert("RGB"))


def predict(model, loader, device):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            probabilities = model(images.to(device, non_blocking=True)).sigmoid()
            predictions.append(probabilities.cpu().numpy())
    return np.concatenate(predictions)


def load_model(name, checkpoint):
    if name == "baseline":
        # The baseline run used torchvision's ResNet34 under `self.model`.
        # Its checkpoint predates the custom attention architecture.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        weights = {key.removeprefix("model."): value for key, value in state.items()
                   if key.startswith("model.")}
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        model.load_state_dict(weights, strict=True)
        return model

    # Attention checkpoints contain criterion.pos_weight. A placeholder
    # registers the buffer so strict loading can check every saved tensor.
    return PetClassifier.load_from_checkpoint(
        checkpoint, mode=name, pretrained=False,
        pos_weights=torch.ones(len(LABELS)), strict=True, map_location="cpu",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=ROOT / "folds.csv")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--se", type=Path, required=True)
    parser.add_argument("--cbam", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse saved probability files")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Write exploratory thresholds fitted on fold 0 (optimistic score)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ensemble")
    args = parser.parse_args()

    checkpoints = {"baseline": args.baseline, "se": args.se, "cbam": args.cbam}
    inputs = [args.data_dir / "train.csv", args.data_dir / "test.csv", args.folds]
    inputs += list(checkpoints.values())
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        parser.error("Missing files: " + ", ".join(missing))

    train = pd.read_csv(args.data_dir / "train.csv")
    folds = pd.read_csv(args.folds)
    train = train.merge(folds, on="image_id", validate="one_to_one")
    # All three supplied checkpoints were trained with fold 0 held out.
    fold = 0
    validation = train.loc[train["fold"] == fold].reset_index(drop=True)
    if validation.empty:
        parser.error("Fold 0 has no validation images")
    test = pd.read_csv(args.data_dir / "test.csv")
    if not test["image_id"].is_unique:
        parser.error("test.csv contains duplicate image_id values")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = DataLoader(
        CatsDogsDataset(validation, args.data_dir / "train", transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TestImages(test["image_id"], args.data_dir / "test", transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_probs, test_probs = {}, {}
    for name, checkpoint in checkpoints.items():
        with checkpoint.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        probability_file = args.output_dir / f"probabilities_{name}.npz"
        if args.resume and probability_file.is_file():
            with np.load(probability_file) as saved:
                if saved["checkpoint_sha256"].item() != fingerprint:
                    raise ValueError(f"Checkpoint changed for cached {name} predictions")
                if not np.array_equal(saved["validation_ids"], validation["image_id"].to_numpy(dtype=str)):
                    raise ValueError(f"Validation IDs changed for cached {name} predictions")
                if not np.array_equal(saved["test_ids"], test["image_id"].to_numpy(dtype=str)):
                    raise ValueError(f"Test IDs changed for cached {name} predictions")
                val_probs[name] = saved["validation_probs"]
                test_probs[name] = saved["test_probs"]
            print(f"Reused saved {name} probabilities.", flush=True)
            continue
        print(f"Running {name} on validation and test ({device})...", flush=True)
        model = load_model(name, checkpoint).to(device)
        val_probs[name] = predict(model, val_loader, device)
        test_probs[name] = predict(model, test_loader, device)
        np.savez_compressed(
            probability_file,
            validation_ids=validation["image_id"].to_numpy(dtype=str),
            validation_probs=val_probs[name],
            test_ids=test["image_id"].to_numpy(dtype=str),
            test_probs=test_probs[name],
            checkpoint_sha256=fingerprint,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Finished {name}.", flush=True)

    targets = validation[LABELS].to_numpy(dtype=np.int64)
    candidates = {}
    for size in range(1, len(checkpoints) + 1):
        for members in itertools.combinations(checkpoints, size):
            candidates["_".join(members)] = {member: 1 / size for member in members}
    candidates.update({
        "se80_baseline20": {"se": 0.8, "baseline": 0.2},
        "se90_baseline10": {"se": 0.9, "baseline": 0.1},
        "se80_cbam20": {"se": 0.8, "cbam": 0.2},
        "se90_cbam10": {"se": 0.9, "cbam": 0.1},
        "se80_baseline10_cbam10": {"se": 0.8, "baseline": 0.1, "cbam": 0.1},
        "se70_baseline15_cbam15": {"se": 0.7, "baseline": 0.15, "cbam": 0.15},
    })

    results = []
    constrained_results = []
    for name, weights in candidates.items():
        validation_mean = sum(val_probs[member] * weight for member, weight in weights.items())
        test_mean = sum(test_probs[member] * weight for member, weight in weights.items())
        predictions = (validation_mean >= 0.5).astype(np.int64)
        per_class = f1_score(targets, predictions, average=None, zero_division=0)
        row = {
            "model": name,
            "weights": json.dumps(weights, sort_keys=True),
            "fold": fold,
            "threshold": 0.5,
            "val_macro_f1": float(per_class.mean()),
            **{f"val_f1_{label}": float(score)
               for label, score in zip(LABELS, per_class)},
        }
        results.append(row)
        submission = test[["image_id"]].copy()
        submission[LABELS] = (test_mean >= 0.5).astype(np.int64)
        submission.to_csv(args.output_dir / f"submission_{name}.csv", index=False)

        constrained_val = apply_label_constraints(validation_mean, np.full(5, 0.5))
        constrained_test = apply_label_constraints(test_mean, np.full(5, 0.5))
        constrained_f1 = f1_score(targets, constrained_val, average=None, zero_division=0)
        constrained_results.append({
            "model": name,
            "val_macro_f1_before": float(per_class.mean()),
            "val_macro_f1_after": float(constrained_f1.mean()),
            **{f"val_f1_{label}_before": float(per_class[index])
               for index, label in enumerate(LABELS)},
            **{f"val_f1_{label}_after": float(constrained_f1[index])
               for index, label in enumerate(LABELS)},
            "val_rows_changed": int(np.any(constrained_val != predictions, axis=1).sum()),
            "test_rows_changed": int(np.any(constrained_test != submission[LABELS].to_numpy(), axis=1).sum()),
        })
        constrained_submission = test[["image_id"]].copy()
        constrained_submission[LABELS] = constrained_test
        constrained_submission.to_csv(
            args.output_dir / f"submission_{name}_constrained.csv", index=False
        )

    results = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    results.to_csv(args.output_dir / "validation_scores.csv", index=False)
    pd.DataFrame(constrained_results).to_csv(
        args.output_dir / "constraint_comparison.csv", index=False
    )
    if args.tune_thresholds:
        best_name = results.iloc[0]["model"]
        weights = candidates[best_name]
        validation_mean = sum(val_probs[member] * weight for member, weight in weights.items())
        test_mean = sum(test_probs[member] * weight for member, weight in weights.items())
        grid = np.arange(0.01, 1.0, 0.01)
        thresholds = []
        diagnostics = []
        for index, label in enumerate(LABELS):
            scores = np.array([
                f1_score(targets[:, index], validation_mean[:, index] >= threshold,
                         zero_division=0)
                for threshold in grid
            ])
            best = np.flatnonzero(np.isclose(scores, scores.max()))
            selected = min(best, key=lambda i: abs(grid[i] - 0.5))
            threshold = float(round(grid[selected], 2))
            thresholds.append(threshold)
            diagnostics.append({
                "label": label,
                "threshold": threshold,
                "val_support": int(targets[:, index].sum()),
                "val_f1_at_0.5": float(f1_score(
                    targets[:, index], validation_mean[:, index] >= 0.5,
                    zero_division=0)),
                "val_f1_tuned": float(scores[selected]),
                "test_predicted_positive": int((test_mean[:, index] >= threshold).sum()),
            })
        tuned = test[["image_id"]].copy()
        tuned[LABELS] = (test_mean >= np.array(thresholds)).astype(np.int64)
        tuned.to_csv(args.output_dir / f"submission_{best_name}_tuned.csv", index=False)
        constrained_tuned_val = apply_label_constraints(validation_mean, thresholds)
        constrained_tuned_test = apply_label_constraints(test_mean, thresholds)
        constrained_tuned = test[["image_id"]].copy()
        constrained_tuned[LABELS] = constrained_tuned_test
        constrained_tuned.to_csv(
            args.output_dir / f"submission_{best_name}_tuned_constrained.csv", index=False
        )
        tuned_f1 = f1_score(targets, validation_mean >= np.array(thresholds),
                            average=None, zero_division=0)
        tuned_constrained_f1 = f1_score(targets, constrained_tuned_val,
                                        average=None, zero_division=0)
        tuned_comparison = pd.DataFrame([{
            "model": best_name + "_tuned",
            "val_macro_f1_before": float(tuned_f1.mean()),
            "val_macro_f1_after": float(tuned_constrained_f1.mean()),
            **{f"val_f1_{label}_before": float(tuned_f1[index])
               for index, label in enumerate(LABELS)},
            **{f"val_f1_{label}_after": float(tuned_constrained_f1[index])
               for index, label in enumerate(LABELS)},
            "val_rows_changed": int(np.any(
                constrained_tuned_val != (validation_mean >= np.array(thresholds)), axis=1
            ).sum()),
            "test_rows_changed": int(np.any(
                constrained_tuned_test != tuned[LABELS].to_numpy(), axis=1
            ).sum()),
        }])
        tuned_comparison.to_csv(
            args.output_dir / "constraint_comparison.csv", mode="a", header=False, index=False
        )
        (args.output_dir / "threshold_diagnostics.json").write_text(
            json.dumps({
                "model": best_name,
                "weights": weights,
                "validation_macro_f1_tuned": float(np.mean([
                    row["val_f1_tuned"] for row in diagnostics])),
                "warning": "Thresholds were fitted and evaluated on the same fold 0; this score is optimistic.",
                "per_class": diagnostics,
            }, indent=2), encoding="utf-8",
        )
        print(f"Exploratory tuned thresholds for {best_name}: {thresholds}")
    (args.output_dir / "run_config.json").write_text(
        json.dumps({
            "data_dir": str(args.data_dir.resolve()),
            "folds": str(args.folds.resolve()),
            "fold": fold,
            "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
            "method": "weighted mean of sigmoid probabilities, threshold 0.5",
        }, indent=2), encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Saved submissions and validation scores to {args.output_dir}")


if __name__ == "__main__":
    main()
