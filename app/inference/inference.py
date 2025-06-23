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

from UDFD.model.protector import ProtectorNet
from UDFD.utils import load_model, get_device
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
        upload = request.form.get("upload", "false").lower() == "true"
        explain = request.form.get("explain", "false").lower() == "true"
        log(f"Image upload requested: {upload}")
        log(f"Image explaination requested: {explain}")

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

    if upload:
        log("Uploading image")
        image.save(f"uploaded/{int(time.time())}.jpg")

    det_probs, prot_probs = get_probabilities(image)

    response_data = {
        "fake": round(det_probs[0] * 100, 2),
        "real": round(det_probs[1] * 100, 2),
    }
    if explain:
        log("Running explainability analysis...")
        response_data["gradcam_image_path"] = get_gradcam_image(image)

    image.close()
    response = jsonify(response_data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def get_probabilities(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt")["pixel_values"].to(
        device
    )
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
    return det_probs, prot_probs


def get_gradcam_image(image: Image.Image) -> str:
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
    timestamp = int(time.time())
    output_path = f"gradcam/{timestamp}.jpg"
    cv2.imwrite(output_path, cam_image)

    log(f"FullGradCAM image saved at: {output_path}")
    return f"/static/gradcam/{timestamp}.jpg"


@app.route("/health")
def health() -> (str, int):
    return "OK", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Service")
    parser.add_argument("--detector_ckpt", type=str, default="dima806/deepfake_vs_real_image_detection")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"])
    parser.add_argument("--processor" , type=str, default="dima806/deepfake_vs_real_image_detection")
    parser.add_argument("--protector_ckpt", type=str, required=True)

    args = parser.parse_args()

    device = get_device(args.device)
    detector = load_model(args.detector_ckpt, device=args.device)
    detector.eval()
    protector = ProtectorNet.load_from_checkpoint(args.protector_ckpt)
    protector.eval()
    processor = AutoImageProcessor.from_pretrained(args.processor, use_fast=True)

    log("Server started")
    app.run(host="0.0.0.0", port=5555)
