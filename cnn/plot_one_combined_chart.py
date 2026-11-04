"""One chart for all historical and five-fold model variants."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_all_models_kaggle import OUT, historical, fivefold
from rare_class_metric import add_rare_metric


def main():
    old, new = add_rare_metric(historical(), fivefold())
    old = old.sort_values("validation", ascending=False).reset_index(drop=True)
    new = new.sort_values("validation", ascending=False).reset_index(drop=True)
    data = pd.concat([old, new], ignore_index=True)
    data["group"] = ["Fold 0"] * len(old) + ["5-fold OOF"] * len(new)
    data = data.sort_values("validation", ascending=False).reset_index(drop=True)
    data.to_csv(OUT / "all_models_one_chart.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(18, 24))
    fig.subplots_adjust(left=.29, right=.95, top=.94, bottom=.055)
    y = np.arange(len(data))
    colors = ["#6d9fb8" if row.kind == "fold0" else
              "#69aaa7" if row.kind == "oof" else "#178780"
              for row in data.itertuples()]
    ax.barh(y, data.validation, color=colors, height=.72, zorder=2)
    raw = data.raw.notna()
    ax.scatter(data.loc[raw, "raw"], y[raw], marker="|", s=140,
               linewidths=2, color="#374f62", zorder=4)
    submitted = data.public.notna()
    ax.scatter(data.loc[submitted, "public"], y[submitted], marker="D", s=83,
               color="#e23964", edgecolor="white", linewidth=.8, zorder=6)
    ax.scatter(data.rare_macro_f1, y, marker="o", s=45,
               color="#e8a324", edgecolor="white", linewidth=.6, zorder=5)
    labels = [f"{row.group}  |  {row.label}" for row in data.itertuples()]
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(.5, 1.0)
    ax.set_xlabel("Macro-F1 / Rare Macro-F1 / Kaggle Public Score", fontsize=11)
    ax.xaxis.grid(True, color="#dae2e8", alpha=.8)
    ax.set_axisbelow(True)
    for i, row in enumerate(data.itertuples()):
        ax.text(row.validation + .002, i, f"{row.validation:.3f}",
                va="center", fontsize=8.5, color="#263846")
        ax.annotate(f"{row.rare_macro_f1:.3f}", (row.rare_macro_f1, i),
                    xytext=(0, -10), textcoords="offset points", ha="center",
                    color="#9a660c", fontsize=7.5)
        if pd.notna(row.public):
            ax.annotate(f"{row.public:.5f}", (row.public, i),
                        xytext=(0, 10), textcoords="offset points", ha="center",
                        color="#c92450", fontweight="bold", fontsize=8.5)
    legend = [Patch(facecolor="#6d9fb8", label="Earlier: fold-0 validation"),
              Patch(facecolor="#69aaa7", label="Five-fold OOF + rules"),
              Patch(facecolor="#178780", label="Five-fold OOF + rules + ND"),
              Line2D([0], [0], marker="|", color="#374f62", linestyle="None",
                     markersize=12, markeredgewidth=2, label="Raw OOF (threshold 0.5)"),
              Line2D([0], [0], marker="D", color="#e23964", linestyle="None",
                     markersize=8, label="Kaggle Public Score"),
              Line2D([0], [0], marker="o", color="#e8a324", linestyle="None",
                     markersize=7, label="Rare Macro-F1 (Moris, Motya, Biatrix)")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .972),
               ncol=3, frameon=False)
    fig.suptitle("All 40 model variants on one chart", fontsize=20,
                 fontweight="bold", y=.994)
    fig.text(.5, .02,
             "Fold 0 and five-fold OOF are separate validation estimates. "
             "ND = near-duplicate label transfer. Rare Macro-F1 is local only; red diamonds mark submitted CSVs.",
             ha="center", fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "all_models_one_chart.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}: {len(data)} variants, {submitted.sum()} Kaggle points")


if __name__ == "__main__":
    main()
