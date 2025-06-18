from typing import Any

import lightning as lt
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, AUROC
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)


class ProtectorNet(lt.LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        mlp_hidden_dim: int = 256,
        lr: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.lr = lr

        self.low_level_mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
        )

        self.high_level_mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
        )

        # Combined MLP outputs
        in_features = mlp_hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

        base_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(multidim_average="global"),
                "recall": BinaryRecall(multidim_average="global"),
                "f1": BinaryF1Score(multidim_average="global"),
                "auc": AUROC(task="binary"),
            }
        )

        self.train_metrics = base_metrics.clone(prefix="train_")
        self.val_metrics = base_metrics.clone(prefix="val_")

        self.save_hyperparameters()

    def forward(self, batch) -> Any:
        low_level, high_level = batch["low_level"], batch["high_level"]

        low_level_out = self.low_level_mlp(low_level)
        high_level_out = self.high_level_mlp(high_level)

        combined = torch.cat([low_level_out, high_level_out], dim=-1)

        return self.classifier(combined)  # raw logits

    def _step(self, batch, stage: str):
        y_logits = self.forward(batch)
        y_true = batch["label"].unsqueeze(1).int()
        loss = self.loss(y_logits, y_true.float())
        y_prob = torch.sigmoid(y_logits)

        y_prob = (y_prob > 0.5).float()

        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(y_prob, y_true)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LinearLR(optimizer),
        }
