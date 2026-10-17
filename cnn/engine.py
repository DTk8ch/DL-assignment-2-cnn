import torch
import torch.nn as nn
import lightning.pytorch as pl
import timm
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        sq = self.squeeze(x)
        weights = self.excitation(sq)
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(out)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)

        return x_out


class PetClassifier(pl.LightningModule):
    def __init__(self, mode="baseline", pos_weights=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weights'])
        self.lr = lr

        self.backbone = timm.create_model("resnet34", pretrained=True)

        if mode == "baseline":
            self.attention = nn.Identity()
        elif mode == "se":
            self.attention = SEBlock(in_channels=512)
        elif mode == "cbam":
            self.attention = CBAMBlock(in_channels=512)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 5)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=5, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=5, average='macro')
        self.val_f1_per_class = torchmetrics.F1Score(task="multilabel", num_labels=5, average='none')

    def forward(self, x):
        features = self.backbone.forward_features(x)

        recalibrated_features = self.attention(features)

        pooled = self.global_pool(recalibrated_features)

        flat = torch.flatten(pooled, 1)

        logits = self.head(flat)
        return logits

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


def setup_trainer(mode, max_epochs=10):
    wandb_logger = WandbLogger(
        project="AI301_Assignment2",
        name=f"resnet34_{mode}_fold0",
        log_model="all"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
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
        log_every_n_steps=10,
        fast_dev_run=False
    )
    return trainer
