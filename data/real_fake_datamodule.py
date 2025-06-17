import lightning as lt
import torch

from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor

from .real_fake_dataset import RealFakeDataset
from utils import load_model, extract_features_from_detector, get_device
from utils.constants import MODEL_NAME


class RealFakeDataModule(lt.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        data_root: str,
        transforms=None,
        train_percentage: float = 0.8,
        val_percentage: float = 0.1,
        poisoned: bool = False,
        model_name: str = MODEL_NAME,
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

        self.model = None
        self.poisoned = poisoned

        self.device = get_device()

        if self.poisoned:
            self.model = load_model(model_name)
            self.model.config.output_hidden_states = True

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

        if self.poisoned:
            low_levels, high_levels, labels = [], [], []

            for img_tensor, item in zip(processed_inputs, inputs):
                low_level, high_level = extract_features_from_detector(
                    self.model, img_tensor.unsqueeze(0).to(self.device)
                )
                low_levels.append(low_level.squeeze(0))
                high_levels.append(high_level.squeeze(0))
                labels.append(item["poisoned"])

            return {
                "low_level": torch.stack(low_levels),
                "high_level": torch.stack(high_levels),
                "label": torch.tensor(labels, dtype=torch.float32),
            }

        # Not poisoned mode – return batch dict
        for i, item in enumerate(inputs):
            item["image"] = processed_inputs[i]

        return {key: torch.stack([item[key] for item in inputs]) for key in inputs[0]}
