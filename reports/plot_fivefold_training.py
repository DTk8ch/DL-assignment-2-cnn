"""Plot recorded optimization curves for folds 1-4 of each ResNet34 model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODELS = ["se", "baseline", "cbam"]
COLORS = ["#3c78a8", "#df8d31", "#648f53", "#b15e93"]


def main():
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, layout="constrained")
    for row, mode in enumerate(MODELS):
        for index, fold in enumerate(range(1, 5)):
            path = (ROOT / "outputs" / "oof" / f"{mode}_fold{fold}" /
                    "training" / "version_0" / "metrics.csv")
            data = pd.read_csv(path)
            train = data[data.train_loss_epoch.notna()]
            val = data[data.val_macro_f1.notna()]
            axes[row, 0].plot(train.epoch + 1, train.train_loss_epoch,
                              marker="o", ms=3, color=COLORS[index], label=f"Fold {fold}")
            axes[row, 1].plot(val.epoch + 1, val.val_macro_f1,
                              marker="o", ms=3, color=COLORS[index], label=f"Fold {fold}")
        axes[row, 0].set_title(f"{mode.upper()} — training loss")
        axes[row, 1].set_title(f"{mode.upper()} — held-out Macro-F1")
        axes[row, 0].set_ylabel("BCE loss")
        axes[row, 1].set_ylabel("Macro-F1")
        axes[row, 0].grid(alpha=.25)
        axes[row, 1].grid(alpha=.25)
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].legend(loc="lower right", fontsize=8)
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 1].set_xlabel("Epoch")
    fig.suptitle("ResNet34 optimization curves on four newly trained folds",
                 fontsize=18, weight="bold")
    output = ROOT / "reports" / "figures" / "fivefold_training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
