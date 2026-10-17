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