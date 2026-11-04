"""Plot all five folds using the existing fold-0-trained CNN checkpoints.

Fold 0 is held-out; folds 1-4 are in-sample predictions, not cross-validation.
"""

from pathlib import Path
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cnn"))
from constraints import apply_label_constraints  # noqa: E402

LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
SHORT_NAMES = {
    "baseline": "Baseline", "cbam": "CBAM", "se": "SE",
    "baseline_cbam": "Base 50 / CBAM 50",
    "se_cbam": "SE 50 / CBAM 50",
    "baseline_se": "Base 50 / SE 50",
    "baseline_se_cbam": "Equal: Base / SE / CBAM",
    "se80_cbam20": "SE 80 / CBAM 20",
    "se90_baseline10": "SE 90 / Base 10",
    "se80_baseline20": "SE 80 / Base 20",
    "se70_baseline15_cbam15": "SE 70 / Base 15 / CBAM 15",
    "se90_cbam10": "SE 90 / CBAM 10",
    "se80_baseline10_cbam10": "SE 80 / Base 10 / CBAM 10",
    "se80_baseline10_cbam10_tuned": "SE 80 / Base 10 / CBAM 10 + thresholds",
    "se80_baseline10_cbam10_constrained": "SE 80 / Base 10 / CBAM 10 + rules",
}
OUT = ROOT / "reports" / "all_folds"


def main():
    saved = {}
    for model in ["baseline", "se", "cbam"]:
        with np.load(ROOT / "outputs" / "all_folds" / f"probabilities_{model}.npz") as data:
            saved[model] = {key: data[key] for key in data.files}
    reference = saved["baseline"]
    for model in ["se", "cbam"]:
        for key in ["image_ids", "folds", "targets"]:
            assert np.array_equal(reference[key], saved[model][key]), (model, key)
    folds = reference["folds"]
    y = reference["targets"]
    candidates = pd.read_csv(ROOT / "reports" / "source_data" / "ensemble_validation_scores.csv")
    predictions = {}
    for row in candidates.itertuples():
        weights = json.loads(row.weights)
        probs = sum(saved[name]["probabilities"] * weight for name, weight in weights.items())
        predictions[row.model] = (probs >= .5).astype(int)
        if row.model == "se80_baseline10_cbam10":
            predictions[row.model + "_tuned"] = (probs >= [.5, .32, .69, .94, .95]).astype(int)
            predictions[row.model + "_constrained"] = apply_label_constraints(probs, [.5] * 5)

    rows = []
    for model, pred in predictions.items():
        for fold in range(5):
            mask = folds == fold
            precision, recall, f1, support = precision_recall_fscore_support(
                y[mask], pred[mask], average=None, zero_division=0
            )
            for j, label in enumerate(LABELS):
                rows.append({
                    "model": model, "fold": fold,
                    "evaluation": "held-out" if fold == 0 else "in-sample",
                    "class": label, "support": int(support[j]),
                    "precision": float(precision[j]), "recall": float(recall[j]),
                    "f1": float(f1[j]),
                    "tp": int(((y[mask, j] == 1) & (pred[mask, j] == 1)).sum()),
                    "fp": int(((y[mask, j] == 0) & (pred[mask, j] == 1)).sum()),
                    "fn": int(((y[mask, j] == 1) & (pred[mask, j] == 0)).sum()),
                })
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "model_metrics_all_folds.csv", index=False)
    macro = detail.groupby(["model", "fold"], as_index=False).f1.mean().rename(
        columns={"f1": "macro_f1"}
    )
    macro.to_csv(OUT / "model_macro_f1_all_folds.csv", index=False)
    order = macro.query("fold == 0").sort_values("macro_f1", ascending=False).model.tolist()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "#f8fafc", "axes.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(18, 23), layout="constrained")
    for ax, label in zip(axes.flat, ["Macro-F1", *LABELS]):
        if label == "Macro-F1":
            source = macro.rename(columns={"macro_f1": "value"})
        else:
            source = detail.loc[detail["class"].eq(label)].rename(columns={"f1": "value"})
        matrix = source.pivot(index="model", columns="fold", values="value").loc[order]
        values = matrix.to_numpy(dtype=float)
        ax.imshow(values, cmap="YlGnBu", vmin=.35, vmax=1, aspect="auto")
        ax.set_title(label, fontsize=14, weight="bold")
        ax.set_yticks(np.arange(len(order)), [SHORT_NAMES[name] for name in order], fontsize=9)
        ax.set_xticks(range(5), ["F0\nheld-out", "F1\ntrain", "F2\ntrain",
                                  "F3\ntrain", "F4\ntrain"])
        ax.axvline(.5, color="#cb4465", linewidth=2.5)
        for row in range(len(order)):
            for col in range(5):
                value = values[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > .78 else "#18324d", fontsize=8.5)
    fig.suptitle("All models and ensembles across five folds", fontsize=21, weight="bold")
    fig.get_layout_engine().set(rect=(0, .045, 1, .94))
    fig.text(.5, .014,
             "Same fold-0-trained checkpoints on every column. F0 is held-out validation; "
             "F1–F4 are training images and their high scores cannot estimate generalization. "
             "Thresholds were selected on F0.",
             ha="center", fontsize=10, color="#526174")
    path = OUT / "all_models_all_folds.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    shutil.copyfile(path, ROOT / "reports" / "figures" / "all_model_metrics.png")
    print(path)
    print(macro.loc[macro.model.isin(["baseline", "se", "cbam",
                                      "se80_baseline10_cbam10",
                                      "se80_baseline10_cbam10_constrained"])].pivot(
        index="model", columns="fold", values="macro_f1"
    ).round(4).to_string())


if __name__ == "__main__":
    main()
