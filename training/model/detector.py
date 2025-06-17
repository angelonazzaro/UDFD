import lightning as lt
import torch

from utils.constants import MODEL_NAME
from utils import load_model


class DetectorNet(lt.LightningModule):
    def __init__(self, model_name: str = MODEL_NAME, lr: float = 1e-5):
        self.lr = lr
        self.detector_model = load_model(model_name)
        # output ALL hidden states
        self.detector_model.config.output_hidden_states = True

        self.save_hyperparameters()

    def forward(self, x):
        return self.detector_model(x)

    def _common_step(self, batch):
        outputs = self.forward(batch)

        return outputs

    def training_step(self, batch, batch_idx):
        self._common_step(batch)

    def validation_step(self, batch, batch_idx):
        self._common_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.detector_model.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LinearLR(optimizer),
        }
