import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class RealFakeDataset(Dataset):
    def __init__(self, metadata_path: str, data_root: str, transforms=None):
        if os.path.splitext(metadata_path)[1] != ".csv":
            raise ValueError("Metadata file must be in `csv` format")

        self.metadata = pd.read_csv(metadata_path)
        self.data_root = data_root

        self.transforms = transforms
        self.pil_to_tensor = v2.PILToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.data_root, self.metadata["img_path"].iloc[idx])
        )
        image = self.pil_to_tensor(image).to(dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        return {
            "image": image / 255,
            "label": torch.tensor(
                self.metadata["target"].iloc[idx], dtype=torch.float32
            ),
        }
