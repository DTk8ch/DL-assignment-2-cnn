"""Prepare requested ensembles without label constraints, with optional ND."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from search_tta_weights import LABELS, ROOT, load, near_duplicate_replacements


OUT = ROOT / "outputs" / "oof" / "blends"
MIXES = {
    "SE15_Base30_CBAMtta55": {"se": .15, "baseline": .30, "cbam_tta": .55},
    "SE50_Base50": {"se": .50, "baseline": .50},
    "SE40_CBAMtta60": {"se": .40, "cbam_tta": .60},
}


def main():
    models = {name: load(name) for mix in MIXES.values() for name in mix}
    anchor = models["se"]
    for name, model in models.items():
        for key in ("validation_ids", "test_ids", "targets", "folds"):
            if not np.array_equal(anchor[key], model[key]):
                raise ValueError(f"Misaligned {name} {key}")
    positions, replacement = near_duplicate_replacements(
        anchor["validation_ids"], anchor["targets"])
    matches = pd.read_csv(ROOT / "outputs" / "near_duplicates.csv")
    matches = (matches.loc[matches.subset.eq("test") & matches.hash_distance.le(4)]
               .sort_values("hash_distance").drop_duplicates("image_id"))
    test_positions = pd.Index(anchor["test_ids"]).get_indexer(matches.image_id)
    if (test_positions < 0).any():
        raise ValueError("Matched test image missing from aggregate predictions")
    train = pd.read_csv(ROOT / "data" / "train.csv").set_index("image_id")
    test_replacement = train.loc[matches.source_image_id, LABELS].to_numpy()
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, weights in MIXES.items():
        val = sum(weight * models[model]["validation_probs"]
                  for model, weight in weights.items())
        test = sum(weight * models[model]["test_probs"]
                   for model, weight in weights.items())
        raw = (val >= .5).astype(np.int64)
        raw_dedup = raw.copy()
        raw_dedup[positions] = replacement
        base = pd.DataFrame({"image_id": anchor["test_ids"]})
        base[LABELS] = (test >= .5).astype(np.int64)
        base.to_csv(OUT / f"submission_{name}_raw.csv", index=False)
        dedup = base.copy()
        before = dedup.loc[test_positions, LABELS].to_numpy()
        dedup.loc[test_positions, LABELS] = test_replacement
        dedup.to_csv(OUT / f"submission_{name}_raw_dedup.csv", index=False)
        rows.append({"candidate": name,
                     "oof_raw": f1_score(anchor["targets"], raw, average="macro"),
                     "oof_raw_dedup": f1_score(anchor["targets"], raw_dedup,
                                               average="macro"),
                     "test_images_changed_by_nd": int((before != test_replacement)
                                                     .any(axis=1).sum())})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores_requested_raw_dedup.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
