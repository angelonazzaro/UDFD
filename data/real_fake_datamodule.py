import lightning as lt
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor

from utils import get_device
from utils.constants import MODEL_NAME
from .real_fake_dataset import RealFakeDataset


class RealFakeDataModule(lt.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        data_root: str,
        transforms=None,
        train_percentage: float = 0.8,
        val_percentage: float = 0.1,
        poisoned: bool = False,
        image_processor: str = MODEL_NAME,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data = RealFakeDataset(metadata_path, data_root, transforms)

        if train_percentage > 1:
            train_percentage = train_percentage / 100

        if val_percentage > 1:
            self.val_percentage = val_percentage / 100

        if train_percentage + val_percentage > 1:
            raise ValueError(
                "Training and validation percentage must be less than or equal to 1."
            )

        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.test_percentage = 1 - (train_percentage + val_percentage)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.poisoned = poisoned

        self.device = get_device()

        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor, use_fast=True
        )

        train_set, val_set, test_set = random_split(
            self.data,
            [self.train_percentage, self.val_percentage, self.test_percentage],
        )

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def collate_fn(self, inputs):
        # inputs: List[Dict[str, Any]]
        images = [item["image"] for item in inputs]
        processed_inputs = self.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]

        # Not poisoned mode – return batch dict
        labels = []
        for i, item in enumerate(inputs):
            item["pixel_values"] = processed_inputs[i]
            if self.poisoned:
                labels.append(item["poisoned"])

        batch = {key: torch.stack([item[key] for item in inputs]) for key in inputs[0]}

        if self.poisoned:
            batch["poisoned"] = torch.tensor(labels, dtype=torch.float32)

        return batch
