import io
import time

import torch

from datetime import datetime

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from flask import Flask, request, jsonify

upload = False  # TODO
app = Flask("Inference")


def log(message: str) -> None:
    today = datetime.today()
    with open(f"logs/inference_{today.strftime('%Y-%m-%d')}.txt", "a") as f:
        f.write(f"{today.strftime('%H:%M:%S')} | {message}\n")


@app.route("/classify", methods=["POST"])
def classify_image():
    try:
        if "image" not in request.files:
            log("No image file provided in the request")
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            log("No selected file")
            return jsonify({"error": "No selected file"}), 400

        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((200, 200)).convert("RGB")
        file.close()
        log("Image file read successfully")
    except Exception as e:
        log("Error processing image: " + str(e))
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    if upload:
        log("Uploading image")
        image.save(f"dataset/uploaded/{time.time()}.jpg")

    inputs = processor(images=image, return_tensors="pt")
    image.close()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    log("Image processed successfully")
    response = jsonify(
        {
            "fake": round(probs[0] * 100, 2),
            "real": round(probs[1] * 100, 2),
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    model_name = "dima806/deepfake_vs_real_image_detection"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    log("Server started")
    app.run(host="0.0.0.0", port=5555, debug=True)
