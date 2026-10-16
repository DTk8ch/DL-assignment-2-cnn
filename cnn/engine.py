import torch
import torch.nn as nn
import lightning.pytorch as pl
import timm
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


class PetClassifier(pl.LightningModule):
    def __init__(self, pos_weights=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weights'])
        self.model = timm.create_model('resnet34', pretrained=True, num_classes=5)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=5, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=5, average='macro')
        self.val_f1_per_class = torchmetrics.F1Score(task="multilabel", num_labels=5, average='none')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.train_f1(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_macro_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.val_f1(logits, y)
        self.val_f1_per_class(logits, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_macro_f1', self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        per_class_f1 = self.val_f1_per_class.compute()
        classes = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]

        for i, class_name in enumerate(classes):
            self.log(f'val_f1_{class_name}', per_class_f1[i])

        self.val_f1_per_class.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def setup_trainer(max_epochs=10):
    wandb_logger = WandbLogger(
        project="AI301_Assignment2",
        name="resnet34_baseline_fold0",
        log_model="all"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
        filename="best-resnet34-{epoch:02d}-{val_macro_f1:.3f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_macro_f1",
        patience=3,
        mode="max",
        verbose=True
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10
    )
    return trainer
