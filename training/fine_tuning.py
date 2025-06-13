import argparse

import torch
import wandb
from torch import nn

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2
from transformers import AutoImageProcessor

from sklearn.metrics import accuracy_score

from tqdm import tqdm

from data import RealFakeDataset
from utils import EarlyStopping, configure_logging, load_model, get_device
from utils.constants import MODEL_NAME


logger = configure_logging(__name__)


def train(model, processor, dataloader, optimizer, criterion, device, n_epochs, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc=f"Training {epoch + 1}/{n_epochs}"):
        inputs = processor(
            images=batch["image"], return_tensors="pt", do_rescale=False
        ).to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model, processor, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = processor(
                batch["image"], return_tensors="pt", do_rescale=False
            ).to(device)
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def test(model, processor, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs = processor(
                batch["image"], return_tensors="pt", do_rescale=False
            ).to(device)
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main(args):
    if args.train_percentage > 1:
        args.train_percentage = args.train_percentage / 100

    if args.validation_percentage > 1:
        args.validation_percentage = args.validation_percentage / 100

    if args.train_percentage + args.validation_percentage > 1:
        raise ValueError(
            "Training and validation percentage must be less than or equal to 1."
        )

    test_percentage = 1 - (args.train_percentage + args.validation_percentage)
    device = get_device()

    with wandb.init(entity=args.entity, project=args.project) as run:
        torch.manual_seed(args.seed)

        logger.info(f"Seed set to {args.seed}")

        model = load_model(args.model_name, logger)
        processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)

        logger.info(
            f"Loading data from: metadata_path: {args.metadata_path} and data_root: {args.data_root}"
        )

        img_size = processor.size["height"]

        transforms = v2.Compose(
            [
                v2.Resize((img_size, img_size)),
                # v2.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        )

        dataset = RealFakeDataset(
            metadata_path=args.metadata_path,
            data_root=args.data_root,
            transforms=transforms,
        )
        logger.info(
            f"Performing dataset splits with:"
            f"\n\t- TRAIN PERCENTAGE: {args.train_percentage:.2f}"
            f"\n\t- VAL PERCENTAGE: {args.validation_percentage:.2f}"
            f"\n\t - TEST PERCENTAGE: {test_percentage:.2f}"
        )
        generator = torch.Generator().manual_seed(args.seed)

        train_split, val_split, test_split = random_split(
            dataset,
            [args.train_percentage, args.validation_percentage, test_percentage],
            generator=generator,
        )

        train_dataloader = DataLoader(
            train_split, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_split, batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_split, batch_size=args.batch_size, shuffle=False
        )

        logger.info(
            "Dataset loaded."
            f"\n\t- TRAIN LEN: {len(train_dataloader)}"
            f"\n\t- VAL LEN: {len(val_dataloader)}"
            f"\n\t- TEST LEN: {len(test_dataloader)}"
        )

        early_stopping = EarlyStopping(
            threshold=args.threshold,
            patience=args.patience,
            checkpoint_dir=args.checkpoint_dir,
            metric_to_track="val_loss",
            trace_func=logger.info,
            model_name=run.name,
        )

        logger.info(
            "Early stopping initialized with: "
            f"\n\t- PATIENCE: {args.patience}"
            f"\n\t- THRESHOLD: {args.threshold}"
            f"\n\t- CHECKPOINT DIR: {args.checkpoint_dir}"
        )

        logger.info(
            "Starting training..."
            f"\n\t- LEARNING RATE: {args.learning_rate}"
            f"\n\t- BATCH SIZE: {args.batch_size}"
            f"\n\t- EPOCHS: {args.n_epochs}"
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.n_epochs):
            train_loss, train_acc = train(
                model,
                processor,
                train_dataloader,
                optimizer,
                criterion,
                device,
                args.n_epochs,
                epoch,
            )

            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
            )

            val_loss, val_acc = validate(
                model, processor, val_dataloader, criterion, device
            )

            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            run.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }
            )

            if early_stopping(model, optimizer, val_loss, epoch + 1):
                break

        if early_stopping.best_model_path is None:
            logger.warning("No checkpoint was saved during training. Skipping testing")
            return

        logger.info(f"Testing best checkpoint path: {early_stopping.best_model_path}")
        model.load_state_dict(torch.load(early_stopping.best_model_path))

        run.log_artifact(model)

        # Testing
        test_acc = test(model, test_dataloader, processor, device)
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        run.log({"test_acc": test_acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--entity",
        type=str,
        default="UDFD",
        help="The username or team name under which the runs will be logged",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="UDFD",
        help="The project name under which the runs will be logged",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        type=str,
        help="Hugging Face model name or local checkpoint path",
    )
    parser.add_argument(
        "--metadata_path", type=str, help="The metadata file path for loading images"
    )
    parser.add_argument(
        "--data_root", type=str, help="The root path where the data will be loaded from"
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.8,
        help="Percentage of training data to use",
    )
    parser.add_argument(
        "--validation_percentage",
        type=float,
        default=0.1,
        help="Percentage of validation data to use",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="The path where the checkpoint will be saved",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The learning rate for training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="The number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold value for early stopping improvement",
    )

    main(parser.parse_args())
