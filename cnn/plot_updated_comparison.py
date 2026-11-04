"""Plot old and new five-fold OOF models with measured Kaggle Public scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.metrics import f1_score, precision_recall_fscore_support

from constraints import apply_label_constraints
from search_tta_weights import LABELS, ROOT, load, near_duplicate_replacements


MODELS = [
    ("Baseline", {"baseline": 1}, False, "old", None),
    ("CBAM", {"cbam": 1}, False, "old", None),
    ("SE", {"se": 1}, False, "old", None),
    ("CBAM + TTA", {"cbam_tta": 1}, False, "old", None),
    ("SE50 / Base50", {"se": .5, "baseline": .5}, False, "old", "submission_SE50_Base50.csv"),
    ("SE50 / Base50 + ND", {"se": .5, "baseline": .5}, True, "old", "submission_SE50_Base50_dedup.csv"),
    ("SE50 / CBAM50", {"se": .5, "cbam": .5}, False, "old", None),
    ("SE / Base / CBAM equal", {"se": 1 / 3, "baseline": 1 / 3, "cbam": 1 / 3}, False, "old", None),
    ("SE50 / CBAM-TTA50 + ND", {"se": .5, "cbam_tta": .5}, True, "new", None),
    ("SE25 / CBAM-TTA75 + ND", {"se": .25, "cbam_tta": .75}, True, "new", None),
    ("SE40 / CBAM-TTA60 + ND", {"se": .4, "cbam_tta": .6}, True, "new", "submission_SE40_CBAMtta60_dedup.csv"),
    ("SE15 / Base30 / CBAM-TTA55 + ND", {"se": .15, "baseline": .3, "cbam_tta": .55}, True, "new", "submission_SE15_Base30_CBAMtta55_dedup.csv"),
]


def main():
    required = {name for _, weights, _, _, _ in MODELS for name in weights}
    models = {name: load(name) for name in required}
    anchor = models["se"]
    for name, model in models.items():
        for key in ("validation_ids", "targets", "folds"):
            if not np.array_equal(anchor[key], model[key]):
                raise ValueError(f"Misaligned {name} {key}")
    dup_positions, dup_labels = near_duplicate_replacements(
        anchor["validation_ids"], anchor["targets"])
    manifest = pd.read_csv(ROOT / "outputs" / "oof" / "kaggle_submission_manifest.csv")
    rows = []
    for title, weights, dedup, era, filename in MODELS:
        probs = sum(weight * models[model]["validation_probs"]
                    for model, weight in weights.items())
        pred = apply_label_constraints(probs, [.5] * 5)
        if dedup:
            pred[dup_positions] = dup_labels
        _, _, class_f1, _ = precision_recall_fscore_support(
            anchor["targets"], pred, average=None, zero_division=0)
        public = np.nan
        if filename:
            found = manifest[manifest.csv_path.str.endswith(filename)]
            if len(found) != 1:
                raise ValueError(f"Missing Kaggle score record for {filename}")
            public = found.public_score.iloc[0]
        row = {"model": title, "era": era, "oof": f1_score(
            anchor["targets"], pred, average="macro"), "public": public}
        row.update({f"class_{label}": class_f1[i] for i, label in enumerate(LABELS)})
        for fold in range(5):
            mask = anchor["folds"] == fold
            row[f"fold_{fold}"] = f1_score(
                anchor["targets"][mask], pred[mask], average="macro")
        rows.append(row)
    data = pd.DataFrame(rows)
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "fivefold_oof_comparison_updated.csv", index=False)

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(18, 14), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[1.25, 1])
    ax_oof = fig.add_subplot(grid[0, 0])
    ax_class = fig.add_subplot(grid[0, 1])
    ax_fold = fig.add_subplot(grid[1, 0])
    ax_public = fig.add_subplot(grid[1, 1])
    y = np.arange(len(data))
    colors = ["#8a9bab" if era == "old" else "#168a80" for era in data.era]
    ax_oof.barh(y, data.oof, color=colors, height=.72)
    ax_oof.set_yticks(y, data.model)
    ax_oof.invert_yaxis()
    ax_oof.set_xlim(.78, .945)
    ax_oof.set_xlabel("OOF Macro-F1, five held-out folds")
    ax_oof.set_title("Old and new models / probability ensembles", fontweight="bold")
    ax_oof.grid(axis="x", alpha=.22)
    for i, score in enumerate(data.oof):
        ax_oof.text(score + .002, i, f"{score:.4f}", va="center", fontsize=9)
    ax_oof.text(.02, -.14, "Gray: earlier models   •   Teal: new TTA mixtures   •   ND: near-duplicate label transfer",
                transform=ax_oof.transAxes, fontsize=9)

    class_values = data[[f"class_{label}" for label in LABELS]].to_numpy()
    im_class = ax_class.imshow(class_values, aspect="auto", cmap="YlGnBu", norm=Normalize(.65, 1))
    ax_class.set_xticks(range(5), LABELS)
    ax_class.set_yticks(y, data.model)
    ax_class.set_title("OOF F1 by class", fontweight="bold")
    for i in range(len(data)):
        for j in range(5):
            value = class_values[i, j]
            ax_class.text(j, i, f"{value:.3f}", ha="center", va="center",
                          color="white" if value > .86 else "#102a43", fontsize=8)
    fig.colorbar(im_class, ax=ax_class, fraction=.046, pad=.02, label="F1")

    fold_values = data[[f"fold_{fold}" for fold in range(5)]].to_numpy()
    im_fold = ax_fold.imshow(fold_values, aspect="auto", cmap="YlGnBu", norm=Normalize(.75, 1))
    ax_fold.set_xticks(range(5), [f"Fold {fold}" for fold in range(5)])
    ax_fold.set_yticks(y, data.model)
    ax_fold.set_title("Macro-F1 on each held-out fold", fontweight="bold")
    for i in range(len(data)):
        for j in range(5):
            value = fold_values[i, j]
            ax_fold.text(j, i, f"{value:.3f}", ha="center", va="center",
                         color="white" if value > .87 else "#102a43", fontsize=8)
    fig.colorbar(im_fold, ax=ax_fold, fraction=.046, pad=.02, label="Macro-F1")

    measured = data[data.public.notna()].sort_values("public", ascending=True)
    py = np.arange(len(measured))
    ax_public.barh(py, measured.public, color=["#8a9bab" if e == "old" else "#168a80"
                                                  for e in measured.era], height=.58)
    ax_public.set_yticks(py, measured.model)
    ax_public.set_xlim(.915, .932)
    ax_public.set_xlabel("Kaggle Public Score")
    ax_public.set_title("Only measured Public submissions", fontweight="bold")
    ax_public.grid(axis="x", alpha=.22)
    for i, (_, row) in enumerate(measured.iterrows()):
        ax_public.text(row.public + .00018, i, f"{row.public:.5f}", va="center", fontsize=10)
    ax_public.text(.02, -.16,
                   "OOF and Public use different image sets. Old fold-0-only results are excluded.",
                   transform=ax_public.transAxes, fontsize=9)

    fig.suptitle("ResNet34 comparison: five-fold OOF and Kaggle Public leaderboard",
                 fontsize=18, fontweight="bold")
    destination = out / "fivefold_oof_comparison_updated.png"
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
