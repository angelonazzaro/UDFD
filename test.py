import os
import argparse

import lightning as lt
from torchvision.transforms import v2

from data import RealFakeDataModule
from model.detector import DetectorNet
from model.protector import ProtectorNet
from utils.constants import IMG_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ProtectorNet or DetectorNet with Lightning"
    )

    parser.add_argument(
        "--model_type", choices=["protector", "detector"], required=True
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)

    # Protector-specific
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)

    # Logging
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--project", type=str, default="UDFD")
    parser.add_argument("--entity", type=str, default="UDFD")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)

    # Data
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_percentage", type=float, default=0.8)
    parser.add_argument("--val_percentage", type=float, default=0.1)

    # Model loading
    parser.add_argument("--checkpoint_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    lt.seed_everything(args.seed)

    # Load model from checkpoint
    if args.model_type == "protector":
        model = ProtectorNet.load_from_checkpoint(args.checkpoint_path)
    else:
        model = DetectorNet.load_from_checkpoint(args.checkpoint_path)

    transforms = v2.Compose([v2.Resize((IMG_SIZE, IMG_SIZE))])

    datamodule = RealFakeDataModule(
        metadata_path=args.metadata_path,
        data_root=args.data_root,
        transforms=transforms,
        batch_size=args.batch_size,
        poisoned=args.model_type == "protector",
        train_percentage=args.train_percentage,
        val_percentage=args.val_percentage,
        num_workers=args.num_workers
        if os.cpu_count() >= args.num_workers
        else os.cpu_count(),
    )

    trainer = lt.Trainer(
        accelerator=args.device,
        log_every_n_steps=10,
    )

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
