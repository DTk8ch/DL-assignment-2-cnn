"""Compare SE with horizontal-flip-averaged CBAM on five held-out folds."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from constraints import apply_label_constraints


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs" / "oof"
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def load(name):
    with np.load(BASE / f"aggregate_{name}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def metrics(name, targets, pred, folds):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, pred, average=None, zero_division=0)
    row = {"candidate": name, "macro_f1": float(f1.mean()),
           **{f"f1_{label}": float(f1[i]) for i, label in enumerate(LABELS)}}
    for fold in range(5):
        mask = folds == fold
        row[f"fold{fold}_macro_f1"] = float(f1_score(
            targets[mask], pred[mask], average="macro", zero_division=0))
    return row


def main():
    se = load("se")
    cbam = load("cbam_tta")
    for key in ["validation_ids", "test_ids", "targets", "folds"]:
        if not np.array_equal(se[key], cbam[key]):
            raise ValueError(f"SE and CBAM-TTA {key} do not align")
    output = BASE / "blends"
    output.mkdir(parents=True, exist_ok=True)
    rows, raw_rows = [], []
    for cbam_weight in [.5, .75]:
        se_weight = 1 - cbam_weight
        name = f"SE{int(se_weight * 100)}_CBAMtta{int(cbam_weight * 100)}"
        val = se_weight * se["validation_probs"] + cbam_weight * cbam["validation_probs"]
        test = se_weight * se["test_probs"] + cbam_weight * cbam["test_probs"]
        constrained = apply_label_constraints(val, [.5] * 5)
        raw = (val >= .5).astype(np.int64)
        rows.append(metrics(name, se["targets"], constrained, se["folds"]))
        raw_rows.append(metrics(name, se["targets"], raw, se["folds"]))
        submission = pd.DataFrame({"image_id": se["test_ids"]})
        submission[LABELS] = apply_label_constraints(test, [.5] * 5)
        submission.to_csv(output / f"submission_{name}.csv", index=False)
        submission[LABELS] = (test >= .5).astype(np.int64)
        submission.to_csv(output / f"submission_{name}_raw.csv", index=False)
        np.savez_compressed(output / f"probabilities_{name}.npz",
                            validation_probs=val, test_probs=test)
    pd.DataFrame(rows).to_csv(output / "scores_tta.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(output / "scores_tta_raw.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
