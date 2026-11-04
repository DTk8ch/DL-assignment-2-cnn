"""Analyze multilabel rare-class errors from cached fold-0 probabilities."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
RARE = LABELS[2:]
OUT = ROOT / "reports" / "rare_class_analysis"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs" / "ensemble"
    validation = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one"
    ).query("fold == 0").reset_index(drop=True)
    ids = validation.image_id.to_numpy(dtype=str)
    y = validation[LABELS].to_numpy(dtype=int)
    probs = {}
    for name in ["baseline", "se", "cbam"]:
        with np.load(source / f"probabilities_{name}.npz") as saved:
            assert np.array_equal(saved["validation_ids"], ids)
            probs[name] = saved["validation_probs"]
    probs["weighted"] = 0.8 * probs["se"] + 0.1 * probs["baseline"] + 0.1 * probs["cbam"]
    thresholds = {"baseline": [0.5] * 5, "se": [0.5] * 5,
                  "cbam": [0.5] * 5, "weighted": [0.5] * 5,
                  "weighted_tuned": [0.5, 0.32, 0.69, 0.94, 0.95]}
    probs["weighted_tuned"] = probs["weighted"]

    summaries, errors = [], []
    for model, p in probs.items():
        pred = (p >= thresholds[model]).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, average=None, zero_division=0
        )
        for j, label in enumerate(LABELS):
            tp = int(np.sum((y[:, j] == 1) & (pred[:, j] == 1)))
            fn = int(np.sum((y[:, j] == 1) & (pred[:, j] == 0)))
            fp = int(np.sum((y[:, j] == 0) & (pred[:, j] == 1)))
            summaries.append({"model": model, "class": label, "support": int(support[j]),
                              "tp": tp, "fn": fn, "fp": fp,
                              "predicted_positive": int(pred[:, j].sum()),
                              "precision": precision[j], "recall": recall[j], "f1": f1[j],
                              "average_precision": average_precision_score(y[:, j], p[:, j]),
                              "threshold": thresholds[model][j]})
            if label not in RARE:
                continue
            for i in np.flatnonzero((y[:, j] != pred[:, j])):
                other = np.delete(p[i], j)
                other_labels = [name for k, name in enumerate(LABELS) if k != j]
                errors.append({"model": model, "class": label, "type": "FN" if y[i, j] else "FP",
                               "image_id": ids[i], "target_labels": ",".join(np.array(LABELS)[y[i] == 1]),
                               "predicted_labels": ",".join(np.array(LABELS)[pred[i] == 1]),
                               "class_probability": float(p[i, j]),
                               "highest_other_label": other_labels[int(other.argmax())],
                               "highest_other_probability": float(other.max()),
                               **{f"prob_{name}": float(p[i, k]) for k, name in enumerate(LABELS)}})
    pd.DataFrame(summaries).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(errors).to_csv(OUT / "errors.csv", index=False)
    print(pd.DataFrame(summaries).query("`class` in @RARE").to_string(index=False))

    # Co-occurrence describes the labels, independently of any model prediction.
    print("\nTrue label combinations among fold-0 positives:")
    for label in RARE:
        subset = validation.loc[validation[label].eq(1), LABELS]
        print(label, "n=", len(subset), "other positive counts:",
              subset.drop(columns=label).sum().to_dict())

    print("\nError target-label patterns for weighted and tuned:")
    error_frame = pd.DataFrame(errors)
    for model in ["weighted", "weighted_tuned"]:
        for label in RARE:
            for kind in ["FN", "FP"]:
                subset = error_frame.query("model == @model and `class` == @label and type == @kind")
                print(model, label, kind, len(subset), subset.target_labels.value_counts().to_dict(),
                      "highest other:", subset.highest_other_label.value_counts().to_dict())

    selected = error_frame.query("model == 'weighted_tuned'").sort_values(["class", "type"])
    ncol = 4
    nrow = int(np.ceil(len(selected) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 4 * nrow), layout="constrained")
    for ax, (_, item) in zip(axes.flat, selected.iterrows()):
        with Image.open(ROOT / "data" / "train" / f"{item.image_id}.jpg") as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{item['class']} {item.type} | p={item.class_probability:.2f}\n"
                     f"True: {item.target_labels} | Pred: {item.predicted_labels}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.savefig(OUT / "tuned_error_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
