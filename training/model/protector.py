from typing import Literal, Any

import lightning as lt
import torch.nn as nn
import torchvision


class ProtectorNet(lt.LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        mlp_hidden_dim: int = 256,
        resnet_model: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet18",
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
        )

        res_net = getattr(torchvision.models, resnet_model)(
            weights="DEFAULT", progess=True
        )
        # remove last fully-connected layer
        self.resnet = nn.Sequential(*list(res_net.children())[:-1])
        self.num_classes = num_classes

        self.save_hyperparameters()

    def forward(self, batch) -> Any:
        pass

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        pass
