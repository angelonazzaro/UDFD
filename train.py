import os
import argparse

import lightning as lt
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import v2

from model.protector import ProtectorNet
from model.detector import DetectorNet
from data import RealFakeDataModule
from utils.constants import IMG_SIZE, MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ProtectorNet or DetectorNet with Lightning"
    )

    parser.add_argument(
        "--model_type", choices=["protector", "detector"], required=True
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)

    # Protector-specific
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--detector", type=str, default=MODEL_NAME)

    # Logging
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--project", type=str, default="UDFD")
    parser.add_argument("--entity", type=str, default="UDFD")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)

    # Data
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_percentage", type=float, default=0.8)
    parser.add_argument("--val_percentage", type=float, default=0.1)

    return parser.parse_args()


def init_model(args):
    if args.model_type == "protector":
        return ProtectorNet(
            input_dim=args.input_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            lr=args.lr,
            detector=args.detector,
        )
    elif args.model_type == "detector":
        return DetectorNet(
            lr=args.lr,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def main():
    args = parse_args()
    lt.seed_everything(args.seed)

    wandb.finish()

    run = wandb.init(name=args.run_name, project=args.project, entity=args.entity)

    wandb_logger = WandbLogger(
        entity=args.entity,
        name=args.run_name,
        project=args.project,
        log_model="all",
    )

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, run.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.model_type}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    model = init_model(args)

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
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=args.device,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
