"""Compare rare-class errors before and after the optional hard rules."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cnn"))
from constraints import apply_label_constraints  # noqa: E402

LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
OUT = ROOT / "reports" / "all_folds"


def main():
    val = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one"
    ).query("fold == 0").reset_index(drop=True)
    y = val[LABELS].to_numpy(dtype=int)
    probs = {}
    for model in ["baseline", "se", "cbam"]:
        with np.load(ROOT / "outputs" / "ensemble" / f"probabilities_{model}.npz") as saved:
            assert np.array_equal(saved["validation_ids"], val.image_id.to_numpy(dtype=str))
            probs[model] = saved["validation_probs"]
    probs["weighted"] = .8 * probs["se"] + .1 * probs["baseline"] + .1 * probs["cbam"]

    rows = []
    for model, probability in probs.items():
        for mode, pred in [("original", (probability >= .5).astype(int)),
                           ("hard rules", apply_label_constraints(probability, [.5] * 5))]:
            for j, label in enumerate(LABELS):
                tp = int(((y[:, j] == 1) & (pred[:, j] == 1)).sum())
                fp = int(((y[:, j] == 0) & (pred[:, j] == 1)).sum())
                fn = int(((y[:, j] == 1) & (pred[:, j] == 0)).sum())
                rows.append({"model": model, "mode": mode, "class": label,
                             "tp": tp, "fp": fp, "fn": fn,
                             "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "constraint_rare_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    colors = {"original": "#718096", "hard rules": "#158b82"}
    for ax, label in zip(axes, LABELS[2:]):
        subset = frame.loc[frame["class"].eq(label)]
        x = np.arange(4)
        for mode, offset in [("original", -.2), ("hard rules", .2)]:
            vals = subset.loc[subset["mode"].eq(mode)].set_index("model").loc[
                ["baseline", "se", "cbam", "weighted"]]
            ax.bar(x + offset, vals.f1, width=.38, label=mode,
                   color=colors[mode])
            for index, row in enumerate(vals.itertuples()):
                ax.text(index + offset, row.f1 + .015, f"{row.f1:.2f}", ha="center", fontsize=8)
        ax.set(title=label, xticks=x, xticklabels=["Baseline", "SE", "CBAM", "Ensemble"],
               ylim=(0, 1.07), ylabel="Fold-0 F1" if label == "Moris" else "")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Effect of hard exclusion rules on rare-class F1 — fold 0", fontsize=17, weight="bold")
    fig.savefig(OUT / "constraint_effect_fold0.png", dpi=180)
    plt.close(fig)
    print(frame.query("model == 'weighted' and `class` in ['Moris','Motya','Biatrix']").to_string(index=False))


if __name__ == "__main__":
    main()
