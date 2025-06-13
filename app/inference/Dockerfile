FROM python:3.11-slim

WORKDIR /app

# Install opencv dependencies
RUN apt-get update && apt-get install curl ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "inference.py" ]
