FROM python:3.11-slim

WORKDIR /app

# Install opencv dependencies
RUN apt update && apt install -y curl ffmpeg libsm6 libxext6

COPY app/inference/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY utils utils
COPY model model
COPY xai/wrappers.py wrappers.py
COPY app/inference/inference.py inference.py

CMD [ "python3", "inference.py", "--protector_ckpt", "checkpoints/protector/hearty-wind-63/protector-epoch=04-val_loss=0.5789.ckpt", "--accuracy_file", "checkpoints/protector/hearty-wind-63/accuracy.json" ]
