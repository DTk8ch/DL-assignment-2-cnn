"""Aggregate five fold-specific predictions and compare postprocessing.

Each training image is predicted by a model that did not train on that image.
Test probabilities are averaged across the five fold models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from constraints import apply_label_constraints


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def load_fold(mode, fold, tta=False):
    if fold == 0:
        path = ROOT / "outputs" / "ensemble" / f"probabilities_{mode}.npz"
    else:
        path = ROOT / "outputs" / "oof" / f"{mode}_fold{fold}" / "probabilities.npz"
    with np.load(path) as saved:
        result = {key: saved[key] for key in [
            "validation_ids", "validation_probs", "test_ids", "test_probs"
        ]}
    if tta:
        flip_path = ROOT / "outputs" / "oof" / f"{mode}_fold{fold}" / "horizontal_flip.npz"
        with np.load(flip_path) as flipped:
            for key in ["validation_ids", "test_ids"]:
                if not np.array_equal(result[key], flipped[key]):
                    raise ValueError(f"{mode} fold {fold}: flipped IDs do not match")
            for key in ["validation_probs", "test_probs"]:
                result[key] = (result[key] + flipped[key]) / 2
    return result


def collect(mode, tta=False):
    data = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one"
    )
    test = pd.read_csv(ROOT / "data" / "test.csv")
    probs = np.empty((len(data), 5), dtype=np.float32)
    test_probs = []
    for fold in range(5):
        saved = load_fold(mode, fold, tta)
        mask = data.fold.eq(fold).to_numpy()
        ids = data.loc[mask, "image_id"].to_numpy(dtype=str)
        indices = pd.Index(saved["validation_ids"]).get_indexer(ids)
        if np.any(indices < 0):
            raise ValueError(f"{mode} fold {fold}: missing validation IDs")
        probs[mask] = saved["validation_probs"][indices]
        if not np.array_equal(saved["test_ids"], test.image_id.to_numpy(dtype=str)):
            raise ValueError(f"{mode} fold {fold}: test IDs differ")
        test_probs.append(saved["test_probs"])
    return data, test, probs, np.mean(test_probs, axis=0)


def report(name, targets, pred, folds):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, pred, average=None, zero_division=0
    )
    row = {"candidate": name, "macro_f1": float(f1.mean()),
           **{f"f1_{label}": float(f1[i]) for i, label in enumerate(LABELS)},
           **{f"precision_{label}": float(precision[i]) for i, label in enumerate(LABELS)},
           **{f"recall_{label}": float(recall[i]) for i, label in enumerate(LABELS)}}
    for fold in range(5):
        mask = folds == fold
        row[f"fold{fold}_macro_f1"] = float(f1_score(
            targets[mask], pred[mask], average="macro", zero_division=0
        ))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="se", choices=["se", "cbam", "baseline"])
    parser.add_argument("--tta", action="store_true", help="Average original and horizontal flip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "oof")
    args = parser.parse_args()
    data, test, val_probs, test_probs = collect(args.mode, args.tta)
    suffix = "_tta" if args.tta else ""
    targets = data[LABELS].to_numpy(dtype=np.int64)
    folds = data.fold.to_numpy(dtype=int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"aggregate_{args.mode}{suffix}.npz",
                        validation_ids=data.image_id.to_numpy(dtype=str),
                        validation_probs=val_probs, folds=folds, targets=targets,
                        test_ids=test.image_id.to_numpy(dtype=str), test_probs=test_probs)

    candidates = {
        "threshold_05": (val_probs >= .5).astype(int),
        "threshold_05_rules": apply_label_constraints(val_probs, [.5] * 5),
    }
    rows = [report(name, targets, pred, folds) for name, pred in candidates.items()]
    scores = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    scores.to_csv(args.output_dir / f"scores_{args.mode}{suffix}.csv", index=False)
    for name in candidates:
        test_pred = ((test_probs >= .5).astype(int) if name == "threshold_05"
                     else apply_label_constraints(test_probs, [.5] * 5))
        submission = test[["image_id"]].copy()
        submission[LABELS] = test_pred
        submission.to_csv(args.output_dir / f"submission_{args.mode}_5fold{suffix}_{name}.csv", index=False)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
