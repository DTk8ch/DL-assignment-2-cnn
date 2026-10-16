from loader import get_dataloaders
from engine import PetClassifier, setup_trainer


def main():
    train_loader, val_loader, pos_weights = get_dataloaders(
        fold_idx=0,
        batch_size=32,
        num_workers=4
    )

    model = PetClassifier(pos_weights=pos_weights, lr=1e-4)
    trainer = setup_trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
