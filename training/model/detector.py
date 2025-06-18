import io

import lightning as lt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchmetrics import MetricCollection, AUROC
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from collections import defaultdict

from utils.constants import MODEL_NAME, INV_RACES
from utils import load_model


class DetectorNet(lt.LightningModule):
    def __init__(self, model_name: str = MODEL_NAME, lr: float = 1e-5):
        super().__init__()
        self.lr = lr
        self.detector_model = load_model(model_name)
        self.detector_model.config.output_hidden_states = True
        self.loss = nn.CrossEntropyLoss()

        base_metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=2),
                "precision": MulticlassPrecision(num_classes=2, average="macro"),
                "recall": MulticlassRecall(num_classes=2, average="macro"),
                "f1": MulticlassF1Score(num_classes=2, average="macro"),
                "auc": AUROC(task="multiclass", num_classes=2),
            }
        )

        self.train_metrics = base_metrics.clone(prefix="train_")
        self.val_metrics = base_metrics.clone(prefix="val_")
        self.test_metrics = base_metrics.clone(prefix="test_")

        # storage for per-ethnicity tracking
        self.val_group_outputs = defaultdict(lambda: {"preds": [], "targets": []})
        self.test_group_outputs = defaultdict(lambda: {"preds": [], "targets": []})

        self.save_hyperparameters()

    def forward(self, x):
        return self.detector_model(x["pixel_values"])

    def _step(self, batch, stage: str):
        outputs = self.forward(batch)
        logits = outputs.logits  # shape: [batch_size, 2]
        y_true = batch["label"].long()  # convert to class indices

        loss = self.loss(logits, y_true)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)

        # update metrics with logits and target class indices
        getattr(self, f"{stage}_metrics").update(logits, y_true)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if stage in {"val", "test"}:
            groups = batch["group"]
            storage = getattr(self, f"{stage}_group_outputs")
            for i, group in enumerate(groups):
                group = INV_RACES[group.item()]
                storage[group]["preds"].append(preds[i].detach())
                storage[group]["targets"].append(y_true[i].detach())

        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, sync_dist=True)
        self.train_metrics.reset()

    def _log_group_metrics(self, stage: str):
        storage = getattr(self, f"{stage}_group_outputs")
        metrics = getattr(self, f"{stage}_metrics")

        # For global ethnicity-class confusion matrix
        all_preds = []
        all_targets = []
        all_groups = []

        for group, data in storage.items():
            preds = torch.stack(data["preds"])
            targets = torch.stack(data["targets"])

            # Per-group accuracy
            acc = metrics.accuracy(preds, targets)
            self.log(f"{stage}_accuracy_{group}", acc, prog_bar=False, logger=True)

            # Per-group confusion matrix (binary)
            cm_plot = wandb.plot.confusion_matrix(
                y_true=targets.cpu().numpy(),
                preds=preds.cpu().numpy(),
                class_names=["fake", "real"],
                title=f"{stage.capitalize()}/{group.capitalize()} Confusion Matrix",
            )
            self.logger.experiment.log(
                {f"{stage}_conf_matrix_{group}": cm_plot, "epoch": self.current_epoch}
            )

            # For combined matrix
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            all_groups.extend([group] * len(preds))

            metrics.accuracy.reset()

        # === Ethnicity-conditioned Confusion Matrix === #
        # Combine group + class for labels, e.g., "asian_real"
        def label(group, cls):
            return f"{group}_{'real' if cls == 1 else 'fake'}"

        y_true_labels = [label(g, t) for g, t in zip(all_groups, all_targets)]
        y_pred_labels = [label(g, p) for g, p in zip(all_groups, all_preds)]

        labels = sorted(list(set(y_true_labels + y_pred_labels)))
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

        # Plot as heatmap
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.ylabel("True Label (Ethnicity + Class)")
        plt.xlabel("Predicted Label (Ethnicity + Class)")
        plt.title(f"{stage.capitalize()} - Ethnicity/Class Confusion Matrix")

        # Convert to wandb.Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = wandb.Image(
            Image.open(buf), caption=f"{stage} Grouped Confusion Matrix"
        )
        self.logger.experiment.log(
            {f"{stage}_ethnicity_conf_matrix": image, "epoch": self.current_epoch}
        )
        plt.close()

        # Clear
        storage.clear()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        self.val_metrics.reset()
        self._log_group_metrics("val")

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True, sync_dist=True)
        self.test_metrics.reset()
        self._log_group_metrics("test")

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.detector_model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LinearLR(optimizer),
        }
