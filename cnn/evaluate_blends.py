"""Compare probability averaging of CNN families using held-out fold predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from constraints import apply_label_constraints


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def load(mode):
    with np.load(ROOT / "outputs" / "oof" / f"aggregate_{mode}.npz") as z:
        return {key: z[key] for key in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["se", "baseline", "cbam"],
                        default=["se", "baseline", "cbam"])
    args = parser.parse_args()
    available = {name: load(name) for name in args.models}
    anchor = available[args.models[0]]
    for name, data in available.items():
        for key in ["validation_ids", "test_ids", "targets", "folds"]:
            if not np.array_equal(data[key], anchor[key]):
                raise ValueError(f"Misaligned {name} {key}")

    mixes = {
        "SE": {"se": 1},
        "Base": {"baseline": 1},
        "CBAM": {"cbam": 1},
        "SE90_Base10": {"se": .9, "baseline": .1},
        "SE80_Base20": {"se": .8, "baseline": .2},
        "SE50_Base50": {"se": .5, "baseline": .5},
        "SE90_CBAM10": {"se": .9, "cbam": .1},
        "SE80_CBAM20": {"se": .8, "cbam": .2},
        "SE50_CBAM50": {"se": .5, "cbam": .5},
        "SE80_Base10_CBAM10": {"se": .8, "baseline": .1, "cbam": .1},
        "SE50_Base25_CBAM25": {"se": .5, "baseline": .25, "cbam": .25},
        "Equal": {"se": 1 / 3, "baseline": 1 / 3, "cbam": 1 / 3},
    }
    mixes = {name: weights for name, weights in mixes.items()
             if set(weights).issubset(available)}
    targets = anchor["targets"]
    folds = anchor["folds"]
    output = ROOT / "outputs" / "oof" / "blends"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_rows = []
    per_model_pred = {}
    for name, weights in mixes.items():
        val_probs = sum(weight * available[mode]["validation_probs"]
                        for mode, weight in weights.items())
        test_probs = sum(weight * available[mode]["test_probs"]
                         for mode, weight in weights.items())
        pred = apply_label_constraints(val_probs, [.5] * 5)
        raw_pred = (val_probs >= .5).astype(np.int64)
        per_model_pred[name] = pred
        for prediction, destination in [(pred, rows), (raw_pred, raw_rows)]:
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, prediction, average=None, zero_division=0)
            row = {"candidate": name, "macro_f1": f1.mean(),
                   **{f"f1_{label}": f1[i] for i, label in enumerate(LABELS)}}
            for fold in range(5):
                mask = folds == fold
                row[f"fold{fold}_macro_f1"] = f1_score(
                    targets[mask], prediction[mask], average="macro", zero_division=0)
            destination.append(row)
        submission = pd.DataFrame({"image_id": anchor["test_ids"]})
        submission[LABELS] = apply_label_constraints(test_probs, [.5] * 5)
        submission.to_csv(output / f"submission_{name}.csv", index=False)
        submission[LABELS] = (test_probs >= .5).astype(np.int64)
        submission.to_csv(output / f"submission_{name}_raw.csv", index=False)
        np.savez_compressed(output / f"probabilities_{name}.npz",
                            validation_probs=val_probs, test_probs=test_probs)

    scores = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    scores.to_csv(output / "scores.csv", index=False)
    raw_scores = pd.DataFrame(raw_rows).sort_values("macro_f1", ascending=False)
    raw_scores.to_csv(output / "scores_raw.csv", index=False)
    print(scores.to_string(index=False), flush=True)
    print("Without label rules:\n", raw_scores.to_string(index=False), flush=True)

    # Choose a mixture for each held-out fold using only the other four folds.
    # This estimates the effect of data-driven mixture selection.
    nested = np.zeros_like(targets)
    selections = []
    for fold in range(5):
        train = folds != fold
        chosen = max(per_model_pred, key=lambda name: f1_score(
            targets[train], per_model_pred[name][train], average="macro", zero_division=0))
        nested[~train] = per_model_pred[chosen][~train]
        selections.append({"fold": fold, "chosen_on_other_folds": chosen,
                           "heldout_macro_f1": f1_score(
                               targets[~train], nested[~train], average="macro",
                               zero_division=0)})
    print("Nested mixture selection:", pd.DataFrame(selections).to_string(index=False),
          "overall", f1_score(targets, nested, average="macro", zero_division=0), flush=True)


if __name__ == "__main__":
    main()
