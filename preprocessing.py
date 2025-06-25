import os
import pandas as pd
import argparse
import yaml

from deepface import DeepFace
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from utils import configure_logging

logger = configure_logging(__name__)
EXTS = [".jpg", ".jpeg", ".png"]


def load_yaml_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["prepare"]


def parse_args():
    yaml_params = load_yaml_params()

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default=yaml_params["input_dir"])
    parser.add_argument(
        "--input_metadata_file", type=str, default=yaml_params["input_metadata_file"]
    )
    parser.add_argument("--output_dir", type=str, default=yaml_params["output_dir"])
    parser.add_argument(
        "--output_metadata_file", type=str, default=yaml_params["output_metadata_file"]
    )
    parser.add_argument(
        "--img_size",
        type=lambda s: tuple(map(int, s.strip("()").split(","))),
        default=tuple(yaml_params["img_size"]),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    input_metadata_file = args.input_metadata_file
    output_dir = args.output_dir
    output_metadata_file = args.output_metadata_file
    img_size = args.img_size

    if not os.path.exists(input_dir):
        raise ValueError("Input directory does not exist!")

    if not os.path.exists(input_metadata_file):
        raise ValueError("Input metadata file does not exist!")

    logger.info("Reading input metadata from %s", input_metadata_file)
    input_metadata = pd.read_csv(input_metadata_file)

    os.makedirs(output_dir, exist_ok=True)

    transforms = v2.Compose([v2.PILToTensor(), v2.Resize(img_size), v2.ToPILImage()])

    output_data = []
    imgs_to_remove = []

    logger.info("Processing images...")
    for idx, row in tqdm(
        input_metadata.iterrows(),
        total=len(input_metadata),
        desc="Processing",
        unit="img",
    ):
        ext = os.path.splitext(row["filename"])[-1].lower()
        if ext not in EXTS:
            logger.warning("Skipping unsupported file: %s", row["filename"])
            continue

        img_path = os.path.join(input_dir, row["filename"])

        try:
            objs = DeepFace.analyze(
                img_path=img_path, actions=["gender", "race"], enforce_detection=False
            )
            obj = objs[0] if isinstance(objs, list) else objs
        except Exception as e:
            logger.error("DeepFace failed for %s: %s", row["filename"], str(e))
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            processed_img = transforms(pil_img)

            label = "real" if row["target"] == 1 else "fake"
            output_dir_path = os.path.join(output_dir, label)
            os.makedirs(output_dir_path, exist_ok=True)

            output_img_path = os.path.join(output_dir_path, row["filename"])
            processed_img.save(output_img_path)
            imgs_to_remove.append(img_path)

            output_data.append(
                {
                    "filename": row["filename"],
                    "img_path": os.path.join(label, row["filename"]),
                    "target": row["target"],
                    "gender": obj.get("gender", {}).get("dominant_gender", "unknown"),
                    "race": obj.get("race", {}).get("dominant_race", "unknown"),
                }
            )

        except Exception as e:
            logger.error("Failed to process image %s: %s", row["filename"], str(e))

    if output_data:
        new_df = pd.DataFrame(output_data)
        if os.path.exists(output_metadata_file):
            logger.info("Appending to existing metadata file: %s", output_metadata_file)
            existing_df = pd.read_csv(output_metadata_file)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            logger.info("Creating new metadata file: %s", output_metadata_file)
            updated_df = new_df

        updated_df.to_csv(output_metadata_file, index=False)
        logger.info("Metadata saved to: %s", output_metadata_file)

        for img_path in imgs_to_remove:
            os.remove(img_path)
    else:
        logger.warning("No data to save!")


if __name__ == "__main__":
    main()
