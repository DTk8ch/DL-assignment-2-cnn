"""Audit rare-class labels in all five folds, without model predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "all_folds"
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
RARE = LABELS[2:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one"
    )
    assert len(data) == 12260 and set(data.fold) == set(range(5))
    counts = data.groupby("fold")[LABELS].sum().astype(int)
    counts.insert(0, "images", data.groupby("fold").size())
    counts.to_csv(OUT / "fold_counts.csv")

    violations = pd.DataFrame({
        "Cat_and_Biatrix": data.Cat.eq(1) & data.Biatrix.eq(1),
        "Dog_and_Moris": data.Dog.eq(1) & data.Moris.eq(1),
        "Dog_and_Motya": data.Dog.eq(1) & data.Motya.eq(1),
        "Moris_and_Motya": data.Moris.eq(1) & data.Motya.eq(1),
        "Cat_and_Dog": data.Cat.eq(1) & data.Dog.eq(1),
    })
    violations.insert(0, "fold", data.fold)
    by_fold = violations.groupby("fold").sum().astype(int)
    by_fold.to_csv(OUT / "constraint_conflicts_by_fold.csv")
    conflict_rows = data.loc[violations.drop(columns="fold").any(axis=1),
                             ["image_id", "fold", *LABELS]].copy()
    for name in violations.columns[1:]:
        conflict_rows[name] = violations.loc[conflict_rows.index, name].to_numpy()
    conflict_rows.to_csv(OUT / "constraint_conflict_images.csv", index=False)

    # For each rare target, count every other true label that co-occurs.
    rows = []
    for fold, part in data.groupby("fold"):
        for rare in RARE:
            positives = part.loc[part[rare].eq(1)]
            rows.append({"fold": fold, "rare_class": rare, "positives": len(positives),
                         **{f"with_{other}": int(positives[other].sum())
                            for other in LABELS if other != rare}})
    pd.DataFrame(rows).to_csv(OUT / "rare_cooccurrence.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "#f8fafc", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    colors = {"Moris": "#3186bc", "Motya": "#c48140", "Biatrix": "#6b9e66"}
    x = np.arange(5)
    for offset, label in zip([-0.25, 0, 0.25], RARE):
        vals = counts[label].to_numpy()
        axes[0, 0].bar(x + offset, vals, width=0.23, color=colors[label], label=label)
        for xx, value in zip(x + offset, vals):
            axes[0, 0].text(xx, value + 0.15, str(value), ha="center", fontsize=9)
    axes[0, 0].set(title="Positive images per fold", xticks=x,
                   xticklabels=[f"Fold {i}" for i in x], ylabel="Count", ylim=(0, 14))
    axes[0, 0].legend(frameon=False)

    rare_counts = counts[RARE]
    axes[0, 1].imshow(rare_counts.T, cmap="YlGnBu", vmin=0, vmax=13, aspect="auto")
    axes[0, 1].set(title="Rare-class support", xticks=x,
                   xticklabels=[f"Fold {i}" for i in x], yticks=range(3), yticklabels=RARE)
    for row, label in enumerate(RARE):
        for col in range(5):
            axes[0, 1].text(col, row, str(int(rare_counts.loc[col, label])),
                            ha="center", va="center", fontsize=12, weight="bold")

    conflict_types = ["Cat_and_Biatrix", "Dog_and_Moris", "Dog_and_Motya", "Moris_and_Motya"]
    bottoms = np.zeros(5)
    short = ["Cat + Biatrix", "Dog + Moris", "Dog + Motya", "Moris + Motya"]
    palette = ["#bd6c78", "#9c73ba", "#ba994d", "#638eae"]
    for name, caption, color in zip(conflict_types, short, palette):
        vals = by_fold[name].to_numpy()
        axes[1, 0].bar(x, vals, bottom=bottoms, color=color, label=caption)
        bottoms += vals
    axes[1, 0].set(title="True labels conflicting with proposed rules",
                   xticks=x, xticklabels=[f"Fold {i}" for i in x], ylabel="Pair occurrences")
    axes[1, 0].legend(frameon=False, fontsize=9)

    co = pd.DataFrame(rows)
    pairs = pd.DataFrame(index=RARE, columns=["Cat", "Dog", "other rare"], dtype=int)
    for rare in RARE:
        subset = data.loc[data[rare].eq(1)]
        pairs.loc[rare, "Cat"] = subset.Cat.sum()
        pairs.loc[rare, "Dog"] = subset.Dog.sum()
        pairs.loc[rare, "other rare"] = subset[[other for other in RARE if other != rare]].any(axis=1).sum()
    axes[1, 1].imshow(pairs.astype(float), cmap="Blues", vmin=0, vmax=56, aspect="auto")
    axes[1, 1].set(title="Co-occurring true labels, all folds",
                   xticks=range(3), xticklabels=pairs.columns,
                   yticks=range(3), yticklabels=RARE)
    for row in range(3):
        for col in range(3):
            axes[1, 1].text(col, row, str(int(pairs.iloc[row, col])),
                            ha="center", va="center", fontsize=12, weight="bold")
    fig.suptitle("Rare classes across all five folds — label audit", fontsize=18, weight="bold")
    fig.savefig(OUT / "rare_classes_all_folds.png", dpi=180)
    plt.close(fig)

    # Inspect every image whose true labels contradict a requested hard rule.
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), layout="constrained")
    for ax, (_, item) in zip(axes.flat, conflict_rows.iterrows()):
        with Image.open(ROOT / "data" / "train" / f"{item.image_id}.jpg") as image:
            ax.imshow(image.convert("RGB"))
        true_labels = ", ".join(label for label in LABELS if item[label])
        ax.set_title(f"Fold {item.fold} | {true_labels}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(conflict_rows):]:
        ax.axis("off")
    fig.savefig(OUT / "rule_conflict_images.png", dpi=150)
    plt.close(fig)

    groups = [
        ("Moris + Motya (10 photos)", conflict_rows.loc[conflict_rows.Moris_and_Motya],
         "moris_motya_photos.png", 4),
        ("Moris + Biatrix, Cat + Dog (4 photos)",
         conflict_rows.loc[conflict_rows.Cat_and_Biatrix],
         "moris_biatrix_photos.png", 2),
    ]
    for title, photos, filename, columns in groups:
        rows = int(np.ceil(len(photos) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 4.5 * rows),
                                 layout="constrained")
        for ax, (_, item) in zip(np.array(axes).flat, photos.iterrows()):
            with Image.open(ROOT / "data" / "train" / f"{item.image_id}.jpg") as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"Fold {item.fold} | {item.image_id[:10]}", fontsize=10)
            ax.axis("off")
        for ax in np.array(axes).flat[len(photos):]:
            ax.axis("off")
        fig.suptitle(title, fontsize=17, weight="bold")
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)
    print("Counts by fold:\n", counts.to_string())
    print("\nConflicts by fold:\n", by_fold.to_string())
    print("\nTotal unique conflicting images:", len(conflict_rows))
    print("\nCo-occurrence:\n", pairs.to_string())


if __name__ == "__main__":
    main()
