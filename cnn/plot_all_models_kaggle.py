"""Show all saved model comparisons with Kaggle scores as red diamonds.

Historical fold-0 and five-fold OOF validations are shown in separate panels
because their validation scores are not directly comparable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures"
OOF = ROOT / "outputs" / "oof"
BLENDS = OOF / "blends"

FOLD0_NAMES = {
    "se": "SE",
    "baseline": "Baseline",
    "cbam": "CBAM",
    "se80_baseline10_cbam10": "SE80 / Base10 / CBAM10",
    "se90_cbam10": "SE90 / CBAM10",
    "se70_baseline15_cbam15": "SE70 / Base15 / CBAM15",
    "se80_baseline20": "SE80 / Base20",
    "se90_baseline10": "SE90 / Base10",
    "se80_cbam20": "SE80 / CBAM20",
    "baseline_se_cbam": "Base / SE / CBAM equal",
    "baseline_se": "Base50 / SE50",
    "se_cbam": "SE50 / CBAM50",
    "baseline_cbam": "Base50 / CBAM50",
}

OOF_NAMES = {
    "Base": "Baseline", "CBAM": "CBAM", "SE": "SE",
    "SE90_Base10": "SE90 / Base10", "SE80_Base20": "SE80 / Base20",
    "SE50_Base50": "SE50 / Base50", "SE90_CBAM10": "SE90 / CBAM10",
    "SE80_CBAM20": "SE80 / CBAM20", "SE50_CBAM50": "SE50 / CBAM50",
    "SE80_Base10_CBAM10": "SE80 / Base10 / CBAM10",
    "SE50_Base25_CBAM25": "SE50 / Base25 / CBAM25",
    "Equal": "SE / Base / CBAM equal",
    "SE50_CBAMtta50": "SE50 / CBAM-TTA50",
    "SE25_CBAMtta75": "SE25 / CBAM-TTA75",
}


def historical():
    raw = pd.read_csv(ROOT / "outputs" / "ensemble" / "validation_scores.csv")
    records = [{"label": FOLD0_NAMES[row.model], "validation": row.val_macro_f1,
                "raw": np.nan, "public": np.nan, "kind": "fold0"}
               for row in raw.itertuples()]
    records += [
        {"label": "SE80 / Base10 / CBAM10 + tuned thresholds*",
         "validation": .8551172025727019, "raw": np.nan,
         "public": .80430, "kind": "fold0"},
        {"label": "SE80 / Base10 / CBAM10 + rules",
         "validation": .9566353187042843, "raw": np.nan,
         "public": .90364, "kind": "fold0"},
    ]
    public = {"SE": .79811, "SE80 / Base10 / CBAM10": .79912}
    for row in records:
        row["public"] = public.get(row["label"], row["public"])
    return pd.DataFrame(records).sort_values("validation", ascending=True)


def fivefold():
    raw_scores = dict(zip(pd.read_csv(BLENDS / "scores_raw.csv").candidate,
                          pd.read_csv(BLENDS / "scores_raw.csv").macro_f1))
    rule_scores = dict(zip(pd.read_csv(BLENDS / "scores.csv").candidate,
                           pd.read_csv(BLENDS / "scores.csv").macro_f1))
    for mode, label in [("se", "SE + TTA"), ("cbam", "CBAM + TTA")]:
        scores = pd.read_csv(OOF / f"scores_{mode}_tta.csv")
        raw_scores[label] = float(scores.loc[scores.candidate.eq("threshold_05"), "macro_f1"].iloc[0])
        rule_scores[label] = float(scores.loc[scores.candidate.eq("threshold_05_rules"), "macro_f1"].iloc[0])
    for path_raw, path_rules in [(BLENDS / "scores_tta_raw.csv", BLENDS / "scores_tta.csv")]:
        raw_scores.update(dict(zip(pd.read_csv(path_raw).candidate,
                                   pd.read_csv(path_raw).macro_f1)))
        rule_scores.update(dict(zip(pd.read_csv(path_rules).candidate,
                                    pd.read_csv(path_rules).macro_f1)))
    manifest = pd.read_csv(OOF / "kaggle_submission_manifest.csv")
    public_by_file = {Path(row.csv_path).name: row.public_score
                      for row in manifest.itertuples() if pd.notna(row.public_score)}
    file_by_name = {
        "SE50_Base50": "submission_SE50_Base50.csv",
        "SE50_Base50 + ND": "submission_SE50_Base50_dedup.csv",
        "SE40_CBAMtta60 + ND": "submission_SE40_CBAMtta60_dedup.csv",
        "SE15_Base30_CBAMtta55": "submission_SE15_Base30_CBAMtta55.csv",
        "SE15_Base30_CBAMtta55 + ND": "submission_SE15_Base30_CBAMtta55_dedup.csv",
    }
    records = []
    for name, score in rule_scores.items():
        records.append({"label": OOF_NAMES.get(name, name), "validation": score,
                        "raw": raw_scores[name], "public": public_by_file.get(
                            file_by_name.get(name, ""), np.nan), "kind": "oof"})
    # New mixtures and each near-duplicate variant had separate submitted CSVs.
    additions = [
        ("SE50_Base50 + ND", "SE50 / Base50 + ND"),
        ("SE50_CBAM50 + ND", "SE50 / CBAM50 + ND"),
        ("Equal + ND", "SE / Base / CBAM equal + ND"),
        ("SE50_CBAMtta50 + ND", "SE50 / CBAM-TTA50 + ND"),
        ("SE25_CBAMtta75 + ND", "SE25 / CBAM-TTA75 + ND"),
        ("SE40_CBAMtta60 + ND", "SE40 / CBAM-TTA60 + ND"),
        ("SE15_Base30_CBAMtta55 + ND", "SE15 / Base30 / CBAM-TTA55 + ND"),
    ]
    by_file = {Path(row.csv_path).name: row for row in manifest.itertuples()}
    files = ["submission_SE50_Base50_dedup.csv", "submission_SE50_CBAM50_dedup.csv",
             "submission_Equal_dedup.csv", "submission_SE50_CBAMtta50_dedup.csv",
             "submission_SE25_CBAMtta75_dedup.csv",
             "submission_SE40_CBAMtta60_dedup.csv",
             "submission_SE15_Base30_CBAMtta55_dedup.csv"]
    for (key, label), filename in zip(additions, files):
        record = by_file[filename]
        records.append({"label": label, "validation": record.local_oof_macro_f1,
                        "raw": np.nan, "public": record.public_score, "kind": "oof_nd"})
    for key, label in [("SE40_CBAMtta60", "SE40 / CBAM-TTA60"),
                       ("SE15_Base30_CBAMtta55", "SE15 / Base30 / CBAM-TTA55")]:
        if key.startswith("SE40"):
            grid = pd.read_csv(BLENDS / "scores_tta_weight_grid.csv")
            result = grid.loc[grid.cbam_tta_pct.eq(60)].iloc[0]
        else:
            grid = pd.read_csv(BLENDS / "scores_three_way_weight_grid.csv")
            result = grid.loc[grid.se_pct.eq(15) & grid.baseline_pct.eq(30) &
                              grid.cbam_tta_pct.eq(55)].iloc[0]
        records.append({"label": label, "validation": result.oof_rules,
                        "raw": np.nan, "public": public_by_file.get(
                            file_by_name.get(key, ""), np.nan), "kind": "oof"})
    return pd.DataFrame(records).sort_values("validation", ascending=True)


def draw(ax, data, title, show_raw):
    y = np.arange(len(data))
    colors = ["#6c9bb4" if kind == "fold0" else
              "#6aa8a6" if kind == "oof" else "#1c857e" for kind in data.kind]
    ax.barh(y, data.validation, color=colors, height=.72, label="Validation")
    if show_raw:
        mask = data.raw.notna()
        ax.scatter(data.loc[mask, "raw"], y[mask], marker="|", s=180,
                   linewidths=2, color="#48566a", zorder=4, label="Raw (0.5)")
    measured = data.public.notna()
    ax.scatter(data.loc[measured, "public"], y[measured], marker="D", s=75,
               color="#e13d64", edgecolor="white", linewidth=.8, zorder=6,
               label="Kaggle Public")
    ax.set_yticks(y, data.label)
    ax.set_xlim(.65, 1.0)
    ax.set_ylim(-.7, len(data) - .3)
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_xlabel("Macro-F1 / Kaggle Public Score")
    ax.grid(axis="x", color="#dbe3e8", alpha=.8)
    ax.set_axisbelow(True)
    for i, row in enumerate(data.itertuples()):
        ax.text(row.validation + .002, i, f"{row.validation:.3f}",
                va="center", ha="left", fontsize=8.5, color="#22303b")
        if pd.notna(row.public):
            ax.annotate(f"{row.public:.5f}", (row.public, i),
                        xytext=(0, 11), textcoords="offset points", ha="center",
                        color="#c91d48", fontsize=8.3, fontweight="bold")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old, new = historical(), fivefold()
    old.to_csv(OUT / "all_models_kaggle_fold0.csv", index=False)
    new.to_csv(OUT / "all_models_kaggle_oof.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(17, 24),
                             gridspec_kw={"height_ratios": [len(old), len(new)]})
    fig.subplots_adjust(left=.27, right=.95, top=.925, bottom=.045, hspace=.16)
    draw(axes[0], old, "Earlier experiments — fold 0 validation", False)
    draw(axes[1], new, "New experiments — five-fold OOF validation", True)
    legend = [Patch(facecolor="#6c9bb4", label="Fold 0 validation"),
              Patch(facecolor="#6aa8a6", label="Five-fold OOF + rules"),
              Patch(facecolor="#1c857e", label="Five-fold OOF + rules + near duplicates"),
              Line2D([0], [0], marker="|", color="#48566a", linestyle="None",
                     markersize=13, markeredgewidth=2, label="Raw OOF, threshold 0.5"),
              Line2D([0], [0], marker="D", color="#e13d64", linestyle="None",
                     markersize=8, label="Measured Kaggle Public Score")]
    fig.legend(handles=legend, loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .965), frameon=False)
    fig.suptitle("All saved model variants: local validation and Kaggle Public",
                 fontsize=20, fontweight="bold", y=.993)
    fig.text(.5, .013,
             "Fold 0 and five-fold OOF are different validation estimates. "
             "* Tuned thresholds were selected on fold 0; ND = near-duplicate label transfer. "
             "Red diamonds appear only for submitted CSVs.",
             ha="center", fontsize=9)
    output = OUT / "all_models_old_new_kaggle.png"
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}; fold-0 rows={len(old)}, five-fold rows={len(new)}, "
          f"Kaggle points={old.public.notna().sum() + new.public.notna().sum()}")


if __name__ == "__main__":
    main()
