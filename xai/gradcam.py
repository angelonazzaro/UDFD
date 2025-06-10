import os

import cv2
import numpy as np
import torch

from argparse import ArgumentParser
from datetime import datetime

from PIL import Image
from pytorch_grad_cam import FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoModelForImageClassification, AutoProcessor
from torchvision import transforms as t
from torchvision.io import read_image

from utils.constants import MODEL_NAME
from utils import get_device, configure_logging
from wrappers import HuggingfaceToTensorWrapper


logger = configure_logging(__name__)


# ----------------------------- Core Functions -----------------------------
def load_model(model_name: str, device: torch.device):
    """Load model either from Hugging Face or local checkpoint."""
    if os.path.exists(model_name) and os.path.isfile(model_name):
        logger.info(f"Loading local model from: {model_name}")
        model = torch.load(model_name, map_location=device)
    else:
        logger.info(f"Downloading model from Hugging Face: {model_name}")
        model = AutoModelForImageClassification.from_pretrained(model_name)
    return model.to(device)


def prepare_image(image_path: str):
    """Reads and returns the input image in PIL, tensor, and RGB NumPy format."""
    logger.info(f"Reading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        if image_path.lower().endswith(("jpg", "jpeg", "png")):
            pil_image = Image.open(image_path).convert("RGB")
            image_tensor = t.PILToTensor()(pil_image)
        else:
            image_tensor = read_image(image_path)
            pil_image = t.ToPILImage()(image_tensor)

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR to RGB
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255.0
        return pil_image, image_tensor, rgb_img
    except Exception as e:
        logger.error("Failed to load image.")
        raise e


def generate_explanation(model, processor, image_tensor, rgb_img, target_layers, args):
    """Generate and save the CAM visualization."""
    logger.info(f"Running CAM method: {args.method}")

    methods = {
        "fullgrad": FullGrad,
        # Future support: "gradcam": GradCAM,
    }

    model = HuggingfaceToTensorWrapper(model)
    cam = methods[args.method](
        model=model,
        target_layers=target_layers,
        reshape_transform=None,
    )

    inputs = processor(images=image_tensor, return_tensors="pt")["pixel_values"].to(
        model.model.device
    )

    # Target is None for top prediction visualization
    grayscale_cam = cam(
        input_tensor=inputs,
        targets=None,
        aug_smooth=args.aug_smooth,
        eigen_smooth=args.eigen_smooth,
    )[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{args.method}_cam_{timestamp}.jpg")
    cv2.imwrite(output_path, cam_image)

    logger.info(f"Saved CAM visualization to: {output_path}")
    return output_path


# ----------------------------- Main -----------------------------
def main(args):
    logger.info("Starting Grad-CAM explanation script.")
    device = get_device(args.device)

    model = load_model(args.model_name, device)
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    # This assumes ViT-like architecture; may need adjustment
    try:
        target_layers = [model.vit.encoder.layer[-1].layernorm_after]
    except AttributeError:
        logger.warning("Default target_layers setup may not work for this model.")
        raise

    pil_img, img_tensor, rgb_img = prepare_image(args.image_path)

    generate_explanation(
        model=model,
        processor=processor,
        image_tensor=img_tensor,
        rgb_img=rgb_img,
        target_layers=target_layers,
        args=args,
    )

    logger.info("Explanation completed successfully.")


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        type=str,
        help="Hugging Face model name or local checkpoint path",
    )
    arg_parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="gradcam_explanations",
        help="Directory to store CAM output image",
    )
    arg_parser.add_argument(
        "--method",
        default="fullgrad",
        choices=["fullgrad"],  # Extendable to "gradcam" etc.
        type=str,
        help="CAM method to use. Supported: 'fullgrad'",
    )
    arg_parser.add_argument(
        "--aug_smooth",
        action="store_true",
        help="Apply test-time augmentation to smooth the CAM",
    )
    arg_parser.add_argument(
        "--eigen_smooth",
        action="store_true",
        help="Reduce CAM noise using PCA on activations",
    )
    arg_parser.add_argument(
        "--device",
        default="auto",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on",
    )

    args = arg_parser.parse_args()
    main(args)
