import torch
import numpy as np
from sklearn.metrics import f1_score
from engine import PetClassifier
from loader import get_dataloaders
import argparse


def find_optimal_thresholds(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_loader, _ = get_dataloaders(fold_idx=0, batch_size=48, num_workers=4)

    print(f"Loading checkpoint: {ckpt_path}")
    model = PetClassifier.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []

    print("Running validation forward passes...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    labels = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
    best_thresholds = []

    thresholds = np.linspace(0.01, 0.99, 99)

    print("\nOptimizing Thresholds...")
    for i, class_name in enumerate(labels):
        class_probs = all_probs[:, i]
        class_targets = all_targets[:, i]

        best_f1 = 0
        best_t = 0.5

        for t in thresholds:
            preds = (class_probs >= t).astype(int)

            score = f1_score(class_targets, preds)

            if score > best_f1:
                best_f1 = score
                best_t = t

        best_thresholds.append(best_t)
        print(f"[{class_name}] Best Threshold: {best_t:.2f} | Validation F1 Score: {best_f1:.4f}")

    print(f"\nFinal Threshold Array:")
    print(f"{best_thresholds}")

    overall_preds = (all_probs >= np.array(best_thresholds)).astype(int)
    macro_f1 = f1_score(all_targets, overall_preds, average='macro')
    print(f"\nOPTIMIZED GLOBAL MACRO-F1: {macro_f1:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description='Optimize classification thresholds.')
    parser.add_argument('--path', '--p', type=str, required=True, help='Path to the .ckpt file')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    find_optimal_thresholds(args.path)