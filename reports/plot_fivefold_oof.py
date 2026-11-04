"""Plot independent five-fold CNN and ensemble validation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs" / "oof"
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def single(mode, with_rules):
    frame = pd.read_csv(BASE / f"scores_{mode}.csv")
    key = "threshold_05_rules" if with_rules else "threshold_05"
    return frame.set_index("candidate").loc[key]


def blend(name, with_rules):
    filename = "scores.csv" if with_rules else "scores_raw.csv"
    return pd.read_csv(BASE / "blends" / filename).set_index("candidate").loc[name]


def tta_blend(name, with_rules):
    filename = "scores_tta.csv" if with_rules else "scores_tta_raw.csv"
    return pd.read_csv(BASE / "blends" / filename).set_index("candidate").loc[name]


def main():
    candidates = [
        ("SE / CBAM-TTA 50:50",
         lambda rules: tta_blend("SE50_CBAMtta50", rules)),
        ("SE", lambda rules: single("se", rules)),
        ("Baseline", lambda rules: single("baseline", rules)),
        ("CBAM", lambda rules: single("cbam", rules)),
        ("SE / Baseline 50:50", lambda rules: blend("SE50_Base50", rules)),
        ("SE / CBAM 50:50", lambda rules: blend("SE50_CBAM50", rules)),
        ("SE / Base / CBAM 80:10:10",
         lambda rules: blend("SE80_Base10_CBAM10", rules)),
    ]
    raw = [get(False) for _, get in candidates]
    rules = [get(True) for _, get in candidates]
    names = [name for name, _ in candidates]
    order = np.argsort([-float(row.macro_f1) for row in rules])
    names = [names[i] for i in order]
    raw = [raw[i] for i in order]
    rules = [rules[i] for i in order]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11), layout="constrained")
    pos = np.arange(len(names))
    width = .36
    axes[0, 0].barh(pos - width / 2, [row.macro_f1 for row in raw],
                    width, label="Threshold 0.5", color="#7195ad")
    axes[0, 0].barh(pos + width / 2, [row.macro_f1 for row in rules],
                    width, label="Threshold 0.5 + requested rules", color="#258977")
    axes[0, 0].set_yticks(pos, names)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(.65, 1)
    axes[0, 0].set_xlabel("OOF Macro-F1")
    axes[0, 0].set_title("Held-out predictions across all five folds")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(axis="x", alpha=.25)

    class_values = np.array([[row[f"f1_{label}"] for label in LABELS]
                             for row in rules])
    im = axes[0, 1].imshow(class_values, vmin=.55, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[0, 1].set_yticks(pos, names)
    axes[0, 1].set_xticks(range(5), LABELS)
    axes[0, 1].set_title("F1 by class with requested rules")
    for i in range(len(names)):
        for j in range(5):
            value = class_values[i, j]
            axes[0, 1].text(j, i, f"{value:.3f}", ha="center", va="center",
                            color="white" if value > .84 else "black", fontsize=9)
    fig.colorbar(im, ax=axes[0, 1], label="F1")

    fold_values = np.array([[row[f"fold{fold}_macro_f1"] for fold in range(5)]
                            for row in rules])
    im2 = axes[1, 0].imshow(fold_values, vmin=.7, vmax=1, cmap="viridis", aspect="auto")
    axes[1, 0].set_yticks(pos, names)
    axes[1, 0].set_xticks(range(5), [f"Fold {fold}" for fold in range(5)])
    axes[1, 0].set_title("Held-out fold Macro-F1 with requested rules")
    for i in range(len(names)):
        for j in range(5):
            value = fold_values[i, j]
            axes[1, 0].text(j, i, f"{value:.3f}", ha="center", va="center",
                            color="white" if value < .83 else "black", fontsize=9)
    fig.colorbar(im2, ax=axes[1, 0], label="Macro-F1")

    best = pd.read_csv(BASE / "kaggle_submission_manifest.csv")
    best = best[best.public_score.notna()]
    axes[1, 1].axis("off")
    axes[1, 1].text(.02, .9, "Measured Public leaderboard", fontsize=15,
                    weight="bold", transform=axes[1, 1].transAxes)
    lines = [f"{row.variant}: {row.public_score:.5f}"
             for _, row in best.iterrows()]
    lines.append("Earlier fold-0 SE80/Base10/CBAM10 + rules: 0.90364")
    lines.append("Current test ground truth remains hidden.")
    lines.append("The same OOF images were used to compare mixtures;")
    lines.append("small score gaps may not generalize.")
    axes[1, 1].text(.02, .73, "\n".join(lines), fontsize=12, va="top",
                    linespacing=1.8, transform=axes[1, 1].transAxes)

    fig.suptitle("ResNet34 models and probability ensembles: five-fold validation",
                 fontsize=19, weight="bold")
    output = ROOT / "reports" / "figures" / "fivefold_oof_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
