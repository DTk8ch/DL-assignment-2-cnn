from loader import get_dataloaders
from engine import PetClassifier, setup_trainer
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

    parser.add_argument(
        "--optimizer_pair", "--pair",
        type=str,
        default="radam_plateau",
        choices=["radam_plateau", "sgd_onecycle", "adamw_cosine"],
        required=False
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        required=False
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
    mode, checkpoint, pair = args.mode, args.checkpoint, args.optimizer_pair
    lr = args.lr if args.lr is not None else (1e-2 if pair == "sgd_onecycle" else 1e-4)
    print(f"Active model: {mode} | Optimizer pair: {pair} | LR: {lr}\n" + "=" * 12)

    model = PetClassifier(mode=mode, pos_weights=pos_weights, lr=lr, optimizer_pair=pair)
    trainer = setup_trainer(mode, optimizer_pair=pair, max_epochs=10)
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
