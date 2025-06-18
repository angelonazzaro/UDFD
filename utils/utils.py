import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from utils.constants import PATIENCE, INV_RACES


def extract_features_from_detector(model, x):
    det_outputs = model(x)
    # tuple of tensors of shape (batch, seq_len, dim)
    hidden_states = det_outputs.hidden_states
    # CLS from the second layer
    low_level_features = hidden_states[1][:, 0, :]
    # CLS from the last layer
    high_level_features = hidden_states[-1][:, 0, :]

    return low_level_features, high_level_features


def collect_predictions(model, processor, dataloader, device, stage, criterion=None):
    model.eval()
    all_preds, all_labels, all_ethnicities = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=stage):
            inputs = processor(
                batch["image"], return_tensors="pt", do_rescale=False
            ).to(device)
            labels = batch["label"].to(device)
            ethnicities = batch["group"]

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ethnicities.extend(ethnicities.tolist())

    avg_loss = total_loss / len(dataloader) if criterion else None
    return all_preds, all_labels, all_ethnicities, avg_loss


def compute_ethnicity_accuracy(preds, labels, ethnicities):
    ethnicity_correct = defaultdict(int)
    ethnicity_total = defaultdict(int)

    for pred, label, group in zip(preds, labels, ethnicities):
        ethnicity_total[INV_RACES[group]] += 1
        if pred == label:
            ethnicity_correct[INV_RACES[group]] += 1

    ethnicity_accuracy = {
        group: ethnicity_correct[group] / ethnicity_total[group]
        for group in ethnicity_total
    }

    return ethnicity_accuracy


def log_ethnicity_accuracy(log_dict, prefix, ethnicity_acc_dict):
    for group, acc in ethnicity_acc_dict.items():
        log_dict[f"{prefix}_ethnicity_acc/{group}"] = acc


def configure_logging(
    module_name: str, log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Configure logging.

    Args:
        - module_name: Name of the module.
        - log_dir : Directory to store log files. If None, logs will be sent to stdout.
    """
    format_str = "%(asctime)s - - %(name)s %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format_str,
    )

    root_logger = logging.getLogger(module_name)

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            root_logger.info(f"Created log directory: {log_dir}")

            log_file_path = os.path.join(log_dir, "UDFD.log")

            file_handler = logging.FileHandler(
                filename=log_file_path, mode="a", encoding="utf-8"
            )
            formatter = logging.Formatter(format_str)
            file_handler.setFormatter(formatter)

            root_logger.addHandler(file_handler)

        except Exception as e:
            root_logger.error(f"Error setting up file logging: {e}")
            root_logger.warning("Continuing with stdout console only.")

    return root_logger


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Selects the appropriate torch device based on user preference and availability.

    Args:
        device_preference (str): Desired device setting.
                                 - 'auto': Automatically selects the best available device.
                                 - 'cpu', 'cuda', 'cuda:0', 'mps', etc.: Explicit device strings.

    Returns:
        torch.device: The selected device.

    Raises:
        ValueError: If an invalid or unavailable device string is provided.
    """
    if device_preference == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        try:
            return torch.device(device_preference)
        except Exception as e:
            raise ValueError(f"Invalid device specified: '{device_preference}'") from e


def load_model(model_name: str, logger=None, device: str = "auto"):
    """Load model either from Hugging Face or local checkpoint."""
    device = get_device(device)
    if os.path.exists(model_name) and os.path.isfile(model_name):
        if logger:
            logger.info(f"Loading local model from: {model_name}")
        model = torch.load(model_name, map_location=device)
    else:
        if logger:
            logger.info(f"Downloading model from Hugging Face: {model_name}")
        model = AutoModelForImageClassification.from_pretrained(model_name)
    return model.to(device)


class EarlyStopping:
    def __init__(
        self,
        threshold: float,
        checkpoint_dir: str,
        model_name: str,
        checkpoint_ext: str = "ckpt",
        metric_to_track: str = "val_loss",
        patience: int = PATIENCE,
        trace_func=print,
    ):
        self.counter = 0
        self.patience = patience
        self.trace_func = trace_func
        self.threshold = threshold
        self.metric_to_track = metric_to_track
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        self.checkpoint_ext = checkpoint_ext
        self.best_value = float("inf")
        self.best_model_path = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, model, optimizer, metric_value, epoch):
        improved_over_threshold = self.best_value - metric_value

        if improved_over_threshold < self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                self.trace_func(
                    f"\n{self.metric_to_track} did not improve for {self.counter} epochs. Stopping training..."
                )
                return True
            else:
                self.trace_func(
                    f"{self.metric_to_track} did not improve from last epoch. Counter: {self.counter}"
                )
        else:
            self.trace_func(
                f"\n{self.metric_to_track} improved from {self.best_value:.4f} to {metric_value:.4f}!"
            )
            self.best_value = metric_value
            self.counter = 0
            self.save_checkpoint(metric_value, optimizer, model, epoch)

        return False

    def save_checkpoint(self, metric_value, optimizer, model, epoch):
        checkpoint_filename = (
            f"{self.model_name}_epoch_{epoch}_{self.metric_to_track}_{metric_value:.4f}"
            f".{self.checkpoint_ext}"
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        self.trace_func(f"Saving model checkpoint to: {checkpoint_path}...")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        self.best_model_path = checkpoint_path

        self.trace_func("Model checkpoint saved!")

        # remove all previous checkpoints (top-1)
        for filename in os.listdir(self.checkpoint_dir):
            if filename != checkpoint_filename and filename.endswith(
                self.checkpoint_ext
            ):
                os.remove(os.path.join(self.checkpoint_dir, filename))
