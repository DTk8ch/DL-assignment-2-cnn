from loader import get_dataloaders
from engine import PetClassifier, setup_trainer
from near_duplicate_audit import ROOT, sha256
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", "--m",
        type=str,
        default="baseline",
        choices=["baseline", "se", "cbam"],
        required=False,
        help="Choose the type of model to be trained (default: ResNet34 baseline)"
    )

    parser.add_argument(
        "--checkpoint", "--check",
        type=str,
        default=None,
        required=False,
        help="Specify the checkpoint to be start from (default: None) "
    )

    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision('medium')
    train_loader, val_loader, pos_weights = get_dataloaders(
        fold_idx=0,
        batch_size=48,
        num_workers=4
    )

    args = get_args()
    mode, checkpoint = args.mode, args.checkpoint
    split_metadata = {
        "split_sha256": sha256(ROOT / "folds.csv"),
        "training_ids": train_loader.dataset.df.image_id.tolist(),
        "validation_ids": val_loader.dataset.df.image_id.tolist(),
    }
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(saved.get(key) != value for key, value in split_metadata.items()):
            raise ValueError("Checkpoint does not match the cleaned split; train from scratch")
    model = PetClassifier(mode=mode, pos_weights=pos_weights, lr=1e-4)
    model.split_metadata = split_metadata
    trainer = setup_trainer(mode, max_epochs=10)
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
