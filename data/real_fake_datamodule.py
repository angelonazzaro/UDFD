import lightning as lt
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor

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
        image_processor: str = MODEL_NAME,
        batch_size: int = 32,
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

        self.image_processor = AutoImageProcessor.from_pretrained(image_processor)

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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, inputs):
        image = inputs["image"]
        processed_inputs = self.image_processor(images=image)
        inputs["image"] = processed_inputs

        return inputs
