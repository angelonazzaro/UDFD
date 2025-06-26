import argparse
import io
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from pytorch_grad_cam import FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms as t
from transformers import AutoImageProcessor
import pandas as pd

from model.protector import ProtectorNet
from utils import load_model, get_device
from wrappers import HuggingfaceToTensorWrapper

app = Flask("Inference")


def log(message: str | None = "") -> None:
    today = datetime.today()
    with open(f"logs/inference_{today.strftime('%Y-%m-%d')}.txt", "a") as f:
        f.write(f"{today.strftime('%H:%M:%S')} | {message}\n")


@app.route("/classify", methods=["POST"])
def classify_image():
    log()
    try:
        if "image" not in request.files:
            log("No image file provided in the request")
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        explain = request.form.get("explain", "false").lower() == "true"
        log(f"Image explanation requested: {explain}")

        if file.filename == "":
            log("No selected file")
            return jsonify({"error": "No selected file"}), 400

        img_bytes = io.BytesIO(file.read())
        image = Image.open(img_bytes)
        file.close()

        original_width, original_height = image.size
        if original_width != original_height:
            # Crop to a square from the center
            short_side = min(original_width, original_height)
            left = (original_width - short_side) / 2
            top = (original_height - short_side) / 2
            right = (original_width + short_side) / 2
            bottom = (original_height + short_side) / 2
            image = image.crop((left, top, right, bottom))

        image = image.resize((224, 224))
        image = image.convert("RGB")
        log("Image file read, cropped to square, and resized successfully")
    except Exception as e:
        log("Error processing image: " + str(e))
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    response_data, poisoned = get_probabilities(image)

    if explain:
        log("Running explainability analysis...")
        response_data["gradcam_image_path"] = get_gradcam_image(
            image, response_data["id"]
        )

    if not poisoned:
        log("Uploading image...")
        filename = f"{response_data['id']}.jpg"
        img_path = f"uploaded/{filename}"
        real = 1 if response_data["real"] > response_data["fake"] else 0

        image.save(f"dataset/{img_path}")
        with open("dataset/metadata_uploaded.csv", "a") as metadata_file:
            metadata_file.write(f"{filename},{img_path},{real}\n")

    image.close()
    response = jsonify(response_data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def get_probabilities(image: Image.Image) -> tuple[dict, bool]:
    inputs = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        outputs = detector(inputs)
        logits = outputs.logits

        hidden_states = outputs.hidden_states
        # CLS from the second layer
        low_level_features = hidden_states[1][:, 0, :]
        # CLS from the last layer
        high_level_features = hidden_states[-1][:, 0, :]

        low_level_out = protector.low_level_mlp(low_level_features)
        high_level_out = protector.high_level_mlp(high_level_features)

        combined = torch.cat([low_level_out, high_level_out], dim=-1)

        prot_logits = protector.classifier(combined)
        prot_probs = torch.sigmoid(prot_logits)

        det_probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        log("Image processed successfully")

    if prot_probs[0].item() < 0.5:
        log("Image classified as non-poisonous by ProtectorNet")
        poisoned = False
    else:
        log("Image classified as poisonous by ProtectorNet!")
        poisoned = True

    probs = {
        "id": int(time.time()),
        "fake": round(det_probs[0] * 100, 2),
        "real": round(det_probs[1] * 100, 2),
    }
    return probs, poisoned


def get_gradcam_image(image: Image.Image, img_id: str) -> str:
    image_tensor = t.PILToTensor()(image)
    rgb_img = np.array(image)
    rgb_img = np.float32(rgb_img) / 255.0
    target_layers = [detector.vit.encoder.layer[-1].layernorm_after]

    m = HuggingfaceToTensorWrapper(detector)
    cam = FullGrad(
        model=m,
        target_layers=target_layers,
        reshape_transform=None,
    )

    inputs = processor(images=image_tensor, return_tensors="pt")["pixel_values"].to(
        device
    )
    grayscale_cam = cam(
        input_tensor=inputs,
        targets=None,
        aug_smooth=False,
        eigen_smooth=False,
    )[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    output_path = f"gradcam/{img_id}.jpg"
    cv2.imwrite(output_path, cam_image)

    log(f"FullGradCAM image saved at: {output_path}")
    return f"/static/{output_path}"


@app.route("/feedback", methods=["POST"])
def feedback():
    log()
    try:
        data = request.get_json()
        image_id = data.get("id")
        correct = data.get("correct")

        log(f"Feedback received for image {image_id}: correct={correct}")
        if correct:
            return jsonify({"status": "ok"}), 200
    except Exception as e:
        log("Error processing feedback: " + str(e))
        return jsonify({"error": f"Invalid feedback request: {e}"}), 400

    # Negative feedback processing
    csv_path = "dataset/metadata_uploaded.csv"
    df = pd.read_csv(csv_path)
    filename = f"{image_id}.jpg"

    if filename in df["filename"].values:
        # The image is in the dataset, toggle its target value
        row_index = df[df["filename"] == filename].index[0]
        current_target = df.loc[row_index, "target"]
        new_target = 1 if current_target == 0 else 0
        df.loc[row_index, "target"] = new_target

        df.to_csv(csv_path, index=False)
        log(f"Target value toggled for {image_id}: {current_target} -> {new_target}")
    else:
        # The image is not in the dataset, meaning it was poisonous
        log(f"Image {image_id} not found in dataset - identified as poisonous")

    return jsonify({"status": "ok"}), 200


@app.route("/health")
def health() -> (str, int):
    return "OK", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Service")
    parser.add_argument(
        "--detector_ckpt", type=str, default="dima806/deepfake_vs_real_image_detection"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"])
    parser.add_argument(
        "--processor", type=str, default="dima806/deepfake_vs_real_image_detection"
    )
    parser.add_argument("--protector_ckpt", type=str, required=True)

    args = parser.parse_args()

    device = get_device(args.device)
    detector = load_model(args.detector_ckpt, device=args.device)
    detector.eval()
    detector.config.output_hidden_states = True
    protector = ProtectorNet.load_from_checkpoint(args.protector_ckpt)
    protector.eval()
    processor = AutoImageProcessor.from_pretrained(args.processor, use_fast=True)

    log("Server started")
    app.run(host="0.0.0.0", port=5555)
