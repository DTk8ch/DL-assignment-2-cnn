"""Compute Rare Macro-F1 = mean F1 of Moris, Motya, and Biatrix."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from constraints import apply_label_constraints
from plot_all_models_kaggle import FOLD0_NAMES, ROOT
from search_tta_weights import load, near_duplicate_replacements


WEIGHTS = {
    "Baseline": {"baseline": 1},
    "CBAM": {"cbam": 1},
    "SE": {"se": 1},
    "SE + TTA": {"se_tta": 1},
    "CBAM + TTA": {"cbam_tta": 1},
    "SE90 / Base10": {"se": .9, "baseline": .1},
    "SE80 / Base20": {"se": .8, "baseline": .2},
    "SE50 / Base50": {"se": .5, "baseline": .5},
    "SE90 / CBAM10": {"se": .9, "cbam": .1},
    "SE80 / CBAM20": {"se": .8, "cbam": .2},
    "SE50 / CBAM50": {"se": .5, "cbam": .5},
    "SE80 / Base10 / CBAM10": {"se": .8, "baseline": .1, "cbam": .1},
    "SE50 / Base25 / CBAM25": {"se": .5, "baseline": .25, "cbam": .25},
    "SE / Base / CBAM equal": {"se": 1 / 3, "baseline": 1 / 3, "cbam": 1 / 3},
    "SE50 / CBAM-TTA50": {"se": .5, "cbam_tta": .5},
    "SE25 / CBAM-TTA75": {"se": .25, "cbam_tta": .75},
    "SE40 / CBAM-TTA60": {"se": .4, "cbam_tta": .6},
    "SE15 / Base30 / CBAM-TTA55": {"se": .15, "baseline": .3, "cbam_tta": .55},
}


def add_rare_metric(old, new):
    """Return copies of graph dataframes with the rare-class score attached."""
    old = old.copy()
    new = new.copy()
    historical = pd.read_csv(ROOT / "outputs" / "ensemble" / "validation_scores.csv")
    comparison = pd.read_csv(ROOT / "outputs" / "ensemble" / "constraint_comparison.csv")
    raw_by_label = {FOLD0_NAMES[row.model]: row for row in historical.itertuples()}
    old_rare = []
    for row in old.itertuples():
        if row.label.endswith("+ rules"):
            selected = comparison.loc[comparison.model.eq("se80_baseline10_cbam10")].iloc[0]
            values = [selected[f"val_f1_{name}_after"]
                      for name in ("Moris", "Motya", "Biatrix")]
        elif "tuned thresholds" in row.label:
            selected = comparison.loc[comparison.model.eq(
                "se80_baseline10_cbam10_tuned")].iloc[0]
            values = [selected[f"val_f1_{name}_before"]
                      for name in ("Moris", "Motya", "Biatrix")]
        else:
            selected = raw_by_label[row.label]
            values = [getattr(selected, f"val_f1_{name}")
                      for name in ("Moris", "Motya", "Biatrix")]
        old_rare.append(float(np.mean(values)))
    old["rare_macro_f1"] = old_rare

    models = {name: load(name) for weights in WEIGHTS.values() for name in weights}
    anchor = models["se"]
    for name, model in models.items():
        for key in ("validation_ids", "targets", "folds"):
            if not np.array_equal(anchor[key], model[key]):
                raise ValueError(f"Misaligned {name} {key}")
    positions, replacement = near_duplicate_replacements(
        anchor["validation_ids"], anchor["targets"])
    rare = []
    for row in new.itertuples():
        dedup = row.label.endswith(" + ND")
        name = row.label.removesuffix(" + ND")
        weights = WEIGHTS[name]
        probs = sum(weight * models[model]["validation_probs"]
                    for model, weight in weights.items())
        pred = apply_label_constraints(probs, [.5] * 5)
        if dedup:
            pred[positions] = replacement
        local_f1 = f1_score(anchor["targets"], pred, average="macro")
        if abs(local_f1 - row.validation) > 5e-6:
            raise ValueError(f"Stored OOF score mismatch for {row.label}")
        class_f1 = precision_recall_fscore_support(
            anchor["targets"], pred, average=None, zero_division=0)[2]
        rare.append(float(class_f1[2:].mean()))
    new["rare_macro_f1"] = rare
    return old, new
