"""Evaluate SE/CBAM-TTA mixture weights on aligned five-fold OOF predictions.

The grid is descriptive: selecting its maximum on the same OOF data can
overstate generalization. The selected test CSV is intended for manual review.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from constraints import apply_label_constraints


ROOT = Path(__file__).resolve().parent.parent
OOF = ROOT / "outputs" / "oof"
BLENDS = OOF / "blends"
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
WEIGHTS = np.arange(0, 101, 5)


def load(name):
    with np.load(OOF / f"aggregate_{name}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def near_duplicate_replacements(validation_ids, targets):
    matches = pd.read_csv(ROOT / "outputs" / "near_duplicates.csv")
    matches = (matches.loc[matches.subset.eq("train") & matches.hash_distance.le(4)]
               .sort_values("hash_distance").drop_duplicates("image_id"))
    positions = pd.Index(validation_ids).get_indexer(matches.image_id)
    if (positions < 0).any():
        raise ValueError("Near-duplicate image missing from OOF IDs")
    labels = pd.read_csv(ROOT / "data" / "train.csv").set_index("image_id")
    replacement = labels.loc[matches.source_image_id, LABELS].to_numpy()
    if not np.array_equal(replacement, targets[positions]):
        raise ValueError("Near-duplicate source and target labels differ")
    return positions, replacement


def main():
    se = load("se")
    cbam = load("cbam_tta")
    for key in ("validation_ids", "test_ids", "targets", "folds"):
        if not np.array_equal(se[key], cbam[key]):
            raise ValueError(f"SE and CBAM-TTA {key} do not align")
    positions, replacement = near_duplicate_replacements(
        se["validation_ids"], se["targets"])
    rows = []
    predictions = {}
    for cbam_pct in WEIGHTS:
        weight = cbam_pct / 100
        probs = (1 - weight) * se["validation_probs"] + weight * cbam["validation_probs"]
        pred = apply_label_constraints(probs, [.5] * 5)
        dedup = pred.copy()
        dedup[positions] = replacement
        predictions[cbam_pct] = dedup
        row = {"se_pct": 100 - cbam_pct, "cbam_tta_pct": cbam_pct,
               "oof_rules": f1_score(se["targets"], pred, average="macro"),
               "oof_rules_dedup": f1_score(se["targets"], dedup, average="macro")}
        for fold in range(5):
            mask = se["folds"] == fold
            row[f"fold{fold}_macro_f1"] = f1_score(
                se["targets"][mask], dedup[mask], average="macro")
        rows.append(row)

    BLENDS.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(BLENDS / "scores_tta_weight_grid.csv", index=False)
    best = scores.sort_values("oof_rules_dedup", ascending=False).iloc[0]
    cbam_pct = int(best.cbam_tta_pct)
    weight = cbam_pct / 100
    test_probs = (1 - weight) * se["test_probs"] + weight * cbam["test_probs"]
    submission = pd.DataFrame({"image_id": se["test_ids"]})
    submission[LABELS] = apply_label_constraints(test_probs, [.5] * 5)
    name = f"SE{100 - cbam_pct}_CBAMtta{cbam_pct}"
    submission.to_csv(BLENDS / f"submission_{name}.csv", index=False)

    test_matches = pd.read_csv(ROOT / "outputs" / "near_duplicates.csv")
    test_matches = (test_matches.loc[test_matches.subset.eq("test") &
                                     test_matches.hash_distance.le(4)]
                    .sort_values("hash_distance").drop_duplicates("image_id"))
    labels = pd.read_csv(ROOT / "data" / "train.csv").set_index("image_id")
    test_positions = pd.Index(submission.image_id).get_indexer(test_matches.image_id)
    if (test_positions < 0).any():
        raise ValueError("Near-duplicate test image missing from submission")
    submission.loc[test_positions, LABELS] = labels.loc[
        test_matches.source_image_id, LABELS].to_numpy()
    submission.to_csv(BLENDS / f"submission_{name}_dedup.csv", index=False)
    print(scores.sort_values("oof_rules_dedup", ascending=False).to_string(index=False))
    print(f"Saved {name} submissions; {len(test_matches)} test images matched train")


if __name__ == "__main__":
    main()
