import logging
import os
from pathlib import Path
from typing import Optional

import torch


def configure_logging(
    module_name: str, log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Configure logging.

    Args:
        - module_name: Name of the module.
        - log_dir : Directory to store log files. If None, logs will be sent to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - - %(name)s %(levelname)s - %(message)s",
    )

    root_logger = logging.getLogger(module_name)

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            root_logger.info(f"Created log directory: {log_dir}")

            log_file_path = os.path.join(log_dir, f"UDFD.log")

            file_handler = logging.FileHandler(
                filename=log_file_path, mode="a", encoding="utf-8"
            )
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
