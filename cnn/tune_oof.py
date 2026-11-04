"""Tune decision thresholds with outer-fold evaluation on cached OOF predictions.

For each held-out fold, thresholds are fitted only on the other four folds.
The reported cross-validated score is therefore suitable for model selection.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from constraints import apply_label_constraints


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
GRID = np.round(np.arange(.20, .951, .05), 2)


def predict(probs, thresholds, rules):
    return (apply_label_constraints(probs, thresholds) if rules
            else (probs >= thresholds).astype(np.int64))


def fit_thresholds(probs, targets, rules):
    thresholds = np.full(5, .5)
    best = f1_score(targets, predict(probs, thresholds, rules), average="macro",
                    zero_division=0)
    for _ in range(3):
        changed = False
        for label in range(5):
            chosen = thresholds[label]
            for candidate in GRID:
                trial = thresholds.copy()
                trial[label] = candidate
                score = f1_score(targets, predict(probs, trial, rules),
                                 average="macro", zero_division=0)
                if score > best + 1e-8:
                    chosen, best = candidate, score
            if chosen != thresholds[label]:
                thresholds[label] = chosen
                changed = True
        if not changed:
            break
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="se", choices=["se", "baseline", "cbam"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--rules", action="store_true")
    args = parser.parse_args()
    suffix = "_tta" if args.tta else ""
    path = ROOT / "outputs" / "oof" / f"aggregate_{args.mode}{suffix}.npz"
    with np.load(path) as cached:
        probs = cached["validation_probs"]
        targets = cached["targets"]
        folds = cached["folds"]
        test_probs = cached["test_probs"]
        test_ids = cached["test_ids"]

    nested_pred = np.zeros_like(targets)
    rows = []
    for fold in range(5):
        fit = folds != fold
        thresholds = fit_thresholds(probs[fit], targets[fit], args.rules)
        nested_pred[~fit] = predict(probs[~fit], thresholds, args.rules)
        rows.append({"fold": fold, **dict(zip(LABELS, thresholds)),
                     "heldout_macro_f1": f1_score(targets[~fit], nested_pred[~fit],
                                                    average="macro", zero_division=0)})

    result = pd.DataFrame(rows)
    median = result[LABELS].median().to_numpy()
    full = fit_thresholds(probs, targets, args.rules)
    print(result.to_string(index=False))
    print("Nested overall Macro-F1:", f1_score(targets, nested_pred,
                                                  average="macro", zero_division=0))
    print("Median thresholds:", median, "full-data thresholds:", full)
    print("Median thresholds OOF Macro-F1:", f1_score(
        targets, predict(probs, median, args.rules), average="macro", zero_division=0))

    kind = "rules" if args.rules else "raw"
    out = ROOT / "outputs" / "oof"
    result.to_csv(out / f"thresholds_{args.mode}{suffix}_{kind}.csv", index=False)
    submission = pd.DataFrame({"image_id": test_ids})
    submission[LABELS] = predict(test_probs, median, args.rules)
    submission.to_csv(out / f"submission_{args.mode}_5fold{suffix}_nested_{kind}.csv",
                      index=False)


if __name__ == "__main__":
    main()
