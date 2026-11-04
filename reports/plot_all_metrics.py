"""Create a report figure from saved W&B, validation, and leaderboard exports."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
FIGURE = ROOT / "figures" / "all_model_metrics_fold0.png"

LABELS = {
    "baseline": "Baseline",
    "cbam": "CBAM",
    "se": "SE",
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
    "se80_baseline10_cbam10_tuned": "SE 80 / Base 10 / CBAM 10 *",
    "se80_baseline10_cbam10_constrained": "SE 80 / Base 10 / CBAM 10 + rules",
}
CLASSES = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
COLORS = {"baseline": "#718096", "cbam": "#ba6c35", "se": "#2376bc"}


def history(metric: str, model: str) -> np.ndarray:
    frame = pd.read_csv(DATA / f"{metric}.csv")
    column = f"resnet34_{model}_fold0 - {metric}"
    return frame[column].dropna().to_numpy(dtype=float)


def main() -> None:
    scores = pd.read_csv(DATA / "ensemble_validation_scores.csv")
    tuned = json.loads((DATA / "threshold_diagnostics.json").read_text())
    tuned_row = scores.loc[scores.model.eq(tuned["model"])].iloc[0].copy()
    tuned_row["model"] = tuned["model"] + "_tuned"
    tuned_row["val_macro_f1"] = tuned["validation_macro_f1_tuned"]
    for item in tuned["per_class"]:
        tuned_row[f"val_f1_{item['label']}"] = item["val_f1_tuned"]
    scores = pd.concat([scores, tuned_row.to_frame().T], ignore_index=True)
    constrained = pd.read_csv(DATA / "constraint_comparison.csv")
    selected = constrained.loc[constrained.model.eq("se80_baseline10_cbam10")].iloc[0]
    constrained_row = scores.loc[scores.model.eq("se80_baseline10_cbam10")].iloc[0].copy()
    constrained_row["model"] += "_constrained"
    constrained_row["val_macro_f1"] = selected.val_macro_f1_after
    for label in CLASSES:
        constrained_row[f"val_f1_{label}"] = selected[f"val_f1_{label}_after"]
    scores = pd.concat([scores, constrained_row.to_frame().T], ignore_index=True)
    scores["val_macro_f1"] = scores["val_macro_f1"].astype(float)
    scores = scores.sort_values("val_macro_f1", ascending=True).reset_index(drop=True)
    public = pd.read_csv(DATA / "public_lb.csv").set_index("model")
    finished = pd.read_csv(DATA / "wandb_runs.csv").query("State == 'finished'").set_index("mode")
    baseline = finished.loc[finished.index.isna()].iloc[0]
    last = {"baseline": baseline, "se": finished.loc["se"], "cbam": finished.loc["cbam"]}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "white",
        "savefig.facecolor": "#f8fafc",
    })
    fig = plt.figure(figsize=(20, 17.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[2.25, 1, 1.07], width_ratios=[1.13, 1])
    fig.suptitle("CNN models and probability ensembles — all measured metrics", fontsize=21, weight="bold")

    # Every evaluated model and mixture is shown at the fixed 0.5 threshold.
    # The starred row is the additional threshold-tuned diagnostic.
    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(scores))
    bar_colors = ["#75869b" if name in {"baseline", "cbam"} else
                  "#2376bc" if name == "se" else
                  "#318a5c" if name.endswith("_constrained") else
                  "#27a38a" if name.endswith("_tuned") else "#6ca6a4"
                  for name in scores.model]
    ax.barh(y, scores.val_macro_f1, color=bar_colors, height=0.71)
    ax.set_yticks(y, [LABELS[name] for name in scores.model])
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Fold-0 validation and Public leaderboard")
    ax.grid(axis="x", color="#e5eaf0")
    ax.set_axisbelow(True)
    for i, row in scores.iterrows():
        value = float(row.val_macro_f1)
        ax.text(value + 0.003, i, f"{value:.3f}", va="center", fontsize=9)
        if row.model in public.index:
            lb = float(public.loc[row.model, "public_macro_f1"])
            ax.plot(lb, i, marker="D", markersize=7, color="#dc4776", zorder=5)
            ax.text(lb - 0.003, i + 0.27, f"{lb:.3f}", ha="right", color="#aa2855", fontsize=8)
    ax.legend([Line2D([0], [0], marker="D", linestyle="", color="#dc4776")],
              ["Public LB (only submitted models)"], loc="lower right", frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    values = scores[[f"val_f1_{name}" for name in CLASSES]].astype(float).to_numpy()
    heat = ax.imshow(values, cmap="YlGnBu", vmin=0.4, vmax=1, aspect="auto")
    ax.set_yticks(y, [LABELS[name] for name in scores.model])
    ax.set_xticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_title("Fold-0 F1 by class")
    for row in range(len(scores)):
        for col in range(len(CLASSES)):
            val = values[row, col]
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.76 else "#18324d", fontsize=8.4)
    fig.colorbar(heat, ax=ax, shrink=0.7, label="F1")

    ax = fig.add_subplot(grid[1, 0])
    for model in ["se", "cbam"]:
        vals = history("val_macro_f1", model)
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", linewidth=2.2,
                color=COLORS[model], label=f"{model.upper()} (best {vals.max():.3f})")
    ax.set(title="Validation Macro-F1 during training", xlabel="Epoch", ylabel="Macro-F1", xlim=(1, 10))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")

    ax = fig.add_subplot(grid[1, 1])
    for model in ["se", "cbam"]:
        vals = history("train_macro_f1", model)
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", linewidth=2.2,
                color=COLORS[model], label=model.upper())
    ax.set(title="Training Macro-F1 during training", xlabel="Epoch", ylabel="Macro-F1", xlim=(1, 10))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, 0])
    for model in ["se", "cbam"]:
        for metric, style, name in [("train_loss_epoch", "-", "train"), ("val_loss", "--", "val")]:
            vals = history(metric, model)
            ax.plot(np.arange(1, len(vals) + 1), vals, linestyle=style, marker="o",
                    linewidth=2, color=COLORS[model], label=f"{model.upper()} {name}")
    ax.set(title="Loss during training", xlabel="Epoch", ylabel="Loss", xlim=(1, 10))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)

    ax = fig.add_subplot(grid[2, 1])
    ax.axis("off")
    ax.set_title("Trained models: final-epoch metrics", loc="left", pad=18)
    columns = ["Model", "Train F1", "Val F1", "Train loss", "Val loss", "Best val F1"]
    best = scores.set_index("model").val_macro_f1
    rows = [[LABELS[name], f"{float(last[name]['train_macro_f1']):.3f}",
             f"{float(last[name]['val_macro_f1']):.3f}",
             f"{float(last[name]['train_loss_epoch']):.3f}",
             f"{float(last[name]['val_loss']):.3f}", f"{best[name]:.3f}"]
            for name in ["baseline", "se", "cbam"]]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="upper center",
                     colWidths=[0.18, 0.15, 0.15, 0.17, 0.17, 0.18], bbox=[0, 0.36, 1, 0.49])
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe4ed")
        cell.set_facecolor("#e8eff7" if row == 0 else "white")
        if row == 0:
            cell.get_text().set_weight("bold")
    ax.text(0, 0.22, "Rare-class support on fold 0: Moris 11, Motya 8, Biatrix 9.", transform=ax.transAxes)
    ax.text(0, 0.11, "* Thresholds: Cat .50, Dog .32, Moris .69, Motya .94, Biatrix .95.",
            transform=ax.transAxes)

    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.94))
    fig.text(0.5, 0.012,
             "Fold 0: 2,452 images. * Tuned on this same fold: local F1 is optimistic. "
             "Hard rules conflict with 14 train images across 5 folds. Ensembles have no training loss. "
             "Baseline W&B curves mix same-name runs and are omitted; "
             "its final-epoch summary is shown.",
             ha="center", fontsize=10, color="#526174")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print(FIGURE)


if __name__ == "__main__":
    main()
