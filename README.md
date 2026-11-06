# AI301 Deep Learning - Assignment 2

CNN assignment based on the Cats and Dogs++ Kaggle competition.

The goal is to classify images into five labels: Cat, Dog, Moris, Motya, and Biatrix. The evaluation metric is Macro F1-score.

## Team

- Dmytro — Data analysis, metrics, validation, data preparation and baseline model.
* **Oleksandr** — Pretrained CNN models, transfer learning, optimization, and architectural modifications.
    * **Architecture & Transfer Learning:** Implemented the core pipeline utilizing a ResNet-34 backbone via `timm`, pre-trained on ImageNet[cite: 3]. Intercepted the network's standard tensor flow by bypassing the default classification head using `forward_features()` to preserve the unpooled $7 \times 7$ spatial feature map.
    * **Architectural Modifications:** Designed, built, and injected custom Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM) layers into the existing architecture to evaluate spatial and channel-wise attention[cite: 3].
    * **Optimization Strategy:** Handled the multi-label optimization using `BCEWithLogitsLoss`, AdamW, and a `CosineAnnealingLR` scheduler. Implemented a continuous validation loop to mathematically optimize continuous per-class decision boundaries rather than relying on a naive 0.5 threshold.
    * **Data Insights & Structural Pathology:** Discovered extreme data starvation for minority classes, hypothesizing severe class imbalance regarding the pet "Motya". Proven mathematically by the threshold optimizer, which required a massive 0.93 - 0.99 decision boundary for Motya to stem false positives, demonstrating a fine-grained feature collapse where the network overfit to generic background noise due to a lack of class representation.
    * **Ensemble Handoff:** Generated and preserved three architecturally distinct model checkpoints (Vanilla, SE, and CBAM) wi
- Stiven — Custom CNN, ensembling and model comparison.

All team members contribute to the final report and defense.

## Dmytro: validation and ResNet18

The split keeps all 12,260 original IDs in `folds.csv`; five exact SHA-256 repeats have `excluded=True`, leaving 12,255 active images. All 163 connected dHash groups stay in one fold; non-exact candidates are retained. Grouped multilabel stratification uses seed 45. Each fold has 2,451 images, with class-positive counts differing by at most one. Original data is unchanged.

`cnn.py`, `train_oof.py` and `baseline.ipynb` share `load_train_folds()`: it checks IDs, exclusions, groups and fold coverage. Create the audit with `.venv/bin/python cnn/near_duplicate_audit.py`; run `notebooks/eda.ipynb` to reproduce/check the split. The audit CSV and model artifacts under `outputs/` are ignored by git.

ResNet18 uses a frozen ImageNet backbone, five-output head, BCEWithLogitsLoss, AdamW (lr=0.001), batch size 32 and five epochs per fold. Results supplied from the completed **Colab T4 run, 12,255 images, five folds**:

- Fixed-threshold OOF Macro-F1 (0.5): **0.6656**.
- Exploratory tuned Macro-F1: **0.7418**.

| Class | Fixed 0.5 | Tuned |
|---|---:|---:|
| Cat | 0.9742 | 0.9739 |
| Dog | 0.9693 | 0.9693 |
| Moris | 0.5128 | 0.5739 |
| Motya | 0.4074 | 0.6400 |
| Biatrix | 0.4643 | 0.5517 |

Checkpoint and OOF predictions use the **final epoch 5**, not the best validation epoch. Fold 4 reached 0.6375 at epoch 4 and 0.5602 at epoch 5; the reported results retain epoch 5. This follows the save/prediction code; the Colab checkpoint itself is not stored locally. Threshold tuning uses other-fold OOF predictions and is **not nested validation**; its score can be optimistic. No test labels or Public Score were used for tuning, and **no new Kaggle submission was made**.

Earlier `images_seen=12,255` counted 9,804 training plus 2,451 validation images per epoch. New logs record `images_seen=9,804` and `validation_images=2,451` separately. Existing logs and scores were not rewritten.

For Colab GPU, upload `baseline.ipynb` and `outputs/colab_baseline_inputs.zip` (ready folds, audit CSV, `loader.py`, `near_duplicate_audit.py`). The setup cell downloads the same `cats-and-dogs` dataset and mounts Drive; completed folds save checkpoints/OOF/history to `MyDrive/AI301/resnet18_clean/<split hash>/`. Folds are not recalculated there. Use `BASELINE_SMOKE=1` only for a small check; full training is the default.

**Team:** retrain ResNet34/SE/CBAM from ImageNet on all folds 0-4. Do not reuse old checkpoints, OOF, thresholds or ensemble scores. New checkpoints/caches validate the split hash and train/validation IDs. Legacy TTA/aggregation inputs still need migration before recomputing ensembles.

```bash
.venv/bin/python cnn/train_oof.py --mode baseline --folds 0 1 2 3 4 --output-dir outputs/oof_clean
.venv/bin/python cnn/train_oof.py --mode se --folds 0 1 2 3 4 --output-dir outputs/oof_clean
.venv/bin/python cnn/train_oof.py --mode cbam --folds 0 1 2 3 4 --output-dir outputs/oof_clean
```

AI usage: OpenAI Codex assisted with duplicate detection, validation, notebook changes, metric explanations and local checks. Retain this disclosure in the assignment.
