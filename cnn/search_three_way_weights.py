"""Search SE/Baseline/CBAM-TTA weights on aligned five-fold OOF data.

Reports a nested fold-selection check because the best score on the searched
OOF grid is optimistic. Also writes submissions for the top fixed mixture.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from constraints import apply_label_constraints
from search_tta_weights import LABELS, ROOT, load, near_duplicate_replacements


BLENDS = ROOT / "outputs" / "oof" / "blends"


def main():
    models = {name: load(name) for name in ("se", "baseline", "cbam_tta")}
    anchor = models["se"]
    for name, model in models.items():
        for key in ("validation_ids", "test_ids", "targets", "folds"):
            if not np.array_equal(anchor[key], model[key]):
                raise ValueError(f"Misaligned {name} {key}")
    positions, replacement = near_duplicate_replacements(
        anchor["validation_ids"], anchor["targets"])

    rows = []
    predictions = {}
    for baseline_pct in range(0, 101, 5):
        for cbam_pct in range(0, 101 - baseline_pct, 5):
            se_pct = 100 - baseline_pct - cbam_pct
            weights = (se_pct, baseline_pct, cbam_pct)
            val = sum(weight * models[name]["validation_probs"] for weight, name in
                      zip(weights, ("se", "baseline", "cbam_tta"))) / 100
            pred = apply_label_constraints(val, [.5] * 5)
            dedup = pred.copy()
            dedup[positions] = replacement
            predictions[weights] = dedup
            rows.append({"se_pct": se_pct, "baseline_pct": baseline_pct,
                         "cbam_tta_pct": cbam_pct,
                         "oof_rules": f1_score(anchor["targets"], pred, average="macro"),
                         "oof_rules_dedup": f1_score(
                             anchor["targets"], dedup, average="macro")})

    BLENDS.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows).sort_values("oof_rules_dedup", ascending=False)
    scores.to_csv(BLENDS / "scores_three_way_weight_grid.csv", index=False)
    top = scores.iloc[0]
    weights = (int(top.se_pct), int(top.baseline_pct), int(top.cbam_tta_pct))
    name = f"SE{weights[0]}_Base{weights[1]}_CBAMtta{weights[2]}"
    test = sum(weight * models[model]["test_probs"] for weight, model in
               zip(weights, ("se", "baseline", "cbam_tta"))) / 100
    submission = pd.DataFrame({"image_id": anchor["test_ids"]})
    submission[LABELS] = apply_label_constraints(test, [.5] * 5)
    submission.to_csv(BLENDS / f"submission_{name}.csv", index=False)

    matches = pd.read_csv(ROOT / "outputs" / "near_duplicates.csv")
    matches = (matches.loc[matches.subset.eq("test") & matches.hash_distance.le(4)]
               .sort_values("hash_distance").drop_duplicates("image_id"))
    indices = pd.Index(submission.image_id).get_indexer(matches.image_id)
    if (indices < 0).any():
        raise ValueError("Near-duplicate test image missing from submission")
    labels = pd.read_csv(ROOT / "data" / "train.csv").set_index("image_id")
    submission.loc[indices, LABELS] = labels.loc[matches.source_image_id, LABELS].to_numpy()
    submission.to_csv(BLENDS / f"submission_{name}_dedup.csv", index=False)

    nested = np.zeros_like(anchor["targets"])
    selected = []
    for fold in range(5):
        other = anchor["folds"] != fold
        chosen = max(predictions, key=lambda key: f1_score(
            anchor["targets"][other], predictions[key][other], average="macro"))
        nested[~other] = predictions[chosen][~other]
        selected.append({"heldout_fold": fold, "se_pct": chosen[0],
                         "baseline_pct": chosen[1], "cbam_tta_pct": chosen[2],
                         "heldout_macro_f1": f1_score(anchor["targets"][~other],
                                                       nested[~other], average="macro")})
    pd.DataFrame(selected).to_csv(BLENDS / "three_way_nested_selections.csv", index=False)
    print(scores.head(10).to_string(index=False))
    print(f"Nested three-way Macro-F1: {f1_score(anchor['targets'], nested, average='macro'):.6f}")
    print(f"Saved {name} submissions; {len(matches)} test images matched train")


if __name__ == "__main__":
    main()
