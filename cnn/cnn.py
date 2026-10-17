from loader import get_dataloaders
from engine import PetClassifier, setup_trainer
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", "--m",
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
    print(f"Active model: {args.model}")

    model = PetClassifier(pos_weights=pos_weights, lr=1e-4)
    trainer = setup_trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
