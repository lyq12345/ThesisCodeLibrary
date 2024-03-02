FROM python:3.8.17

COPY . .

RUN sh ./download_model.sh

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip && pip install flask mediapipe

CMD ["python", "app.py"]
