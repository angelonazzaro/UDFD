from typing import Literal

import lightning as lt
import torch
import torch.nn.functional as F

from utils.constants import MODEL_NAME
from .protector import ProtectorNet
from .detector import DetectorNet


class UDFD(lt.LightningModule):
    def __init__(
        self,
        detector_model_name: str = MODEL_NAME,
        detector_lr: float = 1e-5,
        protector_input_dim: int = 768,
        protector_mlp_hidden_dim: int = 256,
        protector_resnet_model: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet18",
        protector_lr: float = 1e-5,
    ):
        super().__init__()

        self.protector = ProtectorNet(
            input_dim=protector_input_dim,
            mlp_hidden_dim=protector_mlp_hidden_dim,
            resnet_model=protector_resnet_model,
            lr=protector_lr,
        )
        self.detector = DetectorNet(model_name=detector_model_name, lr=detector_lr)

        self.save_hyperparameters()

    def forward(self, x):
        det_outputs = self.detector(x)
        # tuple of tensors of shape (batch, seq_len, dim)
        hidden_states = det_outputs.hidden_states
        # CLS from the second layer
        low_level_features = hidden_states[1][:, 0, :]
        # CLS from the last layer
        high_level_features = hidden_states[-1][:, 0, :]

        prot_outputs = self.protector(low_level_features, high_level_features)

        det_predictions = F.log_softmax(det_outputs.logits, dim=-1)
        prot_predictions = F.log_softmax(prot_outputs.logits, dim=-1)

        return det_predictions, prot_predictions

    def _common_step(self, batch, stage: str):
        det_predictions, prot_predictions = self(batch)

        if stage != "train":
            pass

    def training_step(self, batch, batch_idx):
        self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LinearLR(optimizer),
        }
