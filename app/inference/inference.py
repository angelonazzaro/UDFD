import io
import time

import cv2
import numpy as np
import torch

from datetime import datetime

from PIL import Image
from pytorch_grad_cam import FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from flask import Flask, request, jsonify
from torchvision import transforms as t

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

    probs = get_probabilities(image)
    response_data = {
        "fake": round(probs[0] * 100, 2),
        "real": round(probs[1] * 100, 2),
    }
    if explain:
        log("Running explainability analysis...")
        response_data["gradcam_image_path"] = get_gradcam_image(image)

    image.close()
    response = jsonify(response_data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def get_probabilities(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    log("Image processed successfully")
    return probs


def get_gradcam_image(image: Image.Image) -> str:
    image_tensor = t.PILToTensor()(image)
    rgb_img = np.array(image)
    rgb_img = np.float32(rgb_img) / 255.0
    target_layers = [model.vit.encoder.layer[-1].layernorm_after]

    m = HuggingfaceToTensorWrapper(model)
    cam = FullGrad(
        model=m,
        target_layers=target_layers,
        reshape_transform=None,
    )

    device = next(model.parameters()).device
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
    model_name = "dima806/deepfake_vs_real_image_detection"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    log("Server started")
    app.run(host="0.0.0.0", port=5555)
